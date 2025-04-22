import torch
import math
import numpy as np
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from dataclasses import dataclass
from typing import Union

from tesi.dial_mpc.utils.utils import *
from tesi.dial_mpc.config.base_env_config import BaseEnvConfig
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver


@dataclass
class UnitreeGo2EnvConfig(BaseEnvConfig):
    kp: Union[float, torch.Tensor] = 30.0
    kd: Union[float, torch.Tensor] = 0.0
    default_vx: float = 1.0
    default_vy: float = 0.0
    default_vyaw: float = 0.0
    ramp_up_time: float = 2.0
    gait: str = "trot"


class UnitreeGo2Env:
    def __init__(self, num_envs, env_cfg, show_viewer=False):
        self.device = torch.device(env_cfg.backend if env_cfg.backend == "mps" and torch.backends.mps.is_available() else "cpu")
        
        self._foot_radius = 0.0175

        # Environment parameters
        self.num_envs = num_envs
        self.num_obs = 55  # Base observations + joint states + commands
        self.num_actions = 12  # Number of joints (hip, thigh, calf for each leg)
        
        # Control parameters
        self.dt = env_cfg.dt
        self.timestep = env_cfg.timestep
        self.leg_control = env_cfg.leg_control
        self.action_scale = env_cfg.action_scale
        self.kp = env_cfg.kp
        self.kd = env_cfg.kd

        # Command parameters
        self.default_vx = env_cfg.default_vx
        self.default_vy = env_cfg.default_vy
        self.default_vyaw = env_cfg.default_vyaw
        self.ramp_up_time = env_cfg.ramp_up_time
        self.randomize_tasks = env_cfg.randomize_tasks
        
        # Gait parameters
        self._gait = env_cfg.gait
        self._gait_phase = {
            "stand": torch.zeros(4, device=self.device),
            "walk": torch.tensor([0.0, 0.5, 0.75, 0.25], device=self.device),
            "trot": torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device),
            "canter": torch.tensor([0.0, 0.33, 0.33, 0.66], device=self.device),
            "gallop": torch.tensor([0.0, 0.05, 0.4, 0.35], device=self.device),
        }
        self._gait_params = {
            #                  ratio, cadence, amplitude
            "stand": torch.tensor([1.0, 1.0, 0.0], device=self.device),
            "walk": torch.tensor([0.75, 1.0, 0.08], device=self.device),
            "trot": torch.tensor([0.45, 2, 0.08], device=self.device),
            "canter": torch.tensor([0.4, 4, 0.06], device=self.device),
            "gallop": torch.tensor([0.3, 3.5, 0.10], device=self.device),
        }
                
        # Create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=1),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            show_FPS=False,
        )

        # Get rigid solver
        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver

        # Add plane
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # Add robot
        self.base_init_pos = torch.tensor([0.0, 0.0, 0.36], device=self.device)
        self.base_init_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                merge_fixed_links=True,
                links_to_keep=['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot',],
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
            visualize_contact=True,
        )

        # Build scene
        self.scene.build(n_envs=num_envs)

        # Define joint names and indices
        self.dof_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.dof_names]
        
        def find_link_indices(names):
            link_indices = list()
            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices
        
        self.termination_contact_link_indices = find_link_indices(['base'])
        self.feet_link_indices = find_link_indices(['foot'])
        
        assert len(self.termination_contact_link_indices) > 0
        assert len(self.feet_link_indices) > 0
        self.feet_link_indices_world_frame = [i+1 for i in self.feet_link_indices]
        
        # Default joint angles
        self.default_dof_pos = torch.tensor([
            0.0,  # FR_hip_joint
            0.8,  # FR_thigh_joint
            -1.5, # FR_calf_joint
            0.0,  # FL_hip_joint
            0.8,  # FL_thigh_joint
            -1.5, # FL_calf_joint
            0.0,  # RR_hip_joint
            1.0,  # RR_thigh_joint
            -1.5, # RR_calf_joint
            0.0,  # RL_hip_joint
            1.0,  # RL_thigh_joint
            -1.5, # RL_calf_joint
        ], device=self.device)
        
        # Get joint limits from the robot model
        self.joint_range = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)
        
        # Get torque limits from the robot model
        self.joint_torque_range = torch.stack([
            -self.robot.get_dofs_force_range(self.motor_dofs)[1],
            self.robot.get_dofs_force_range(self.motor_dofs)[1]
        ], dim=1)

        # PD control parameters
        self.robot.set_dofs_kp([self.kp] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.kd] * self.num_actions, self.motor_dofs)
        
        
        # Initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        
        # Initialize foot position buffers
        self.foot_positions = torch.ones(
            self.num_envs, len(self.feet_link_indices), 3, device=self.device, dtype=gs.tc_float,
        )
        
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        
        # Command and state buffers
        self.commands = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.commands[:, 0] = self.default_vx
        self.commands[:, 1] = self.default_vy
        self.commands[:, 2] = self.default_vyaw
        
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.dof_torque = torch.zeros_like(self.actions)
        
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        
        # State tracking
        self.feet_air_time = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.last_contacts = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.bool)
        self.target_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.target_pos[:, 2] = 0.3  # Target height
        self.target_yaw = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        
        # Log data
        self.extras = {}

        # For state restoration during MPC rollouts
        self._saved_state = None

        # For tracking reset reasons
        self.last_reset_reason = None

    def step(self, actions):
        self.actions = torch.clamp(actions, -1.0, 1.0)
        
        # Apply actions to robot
        if self.leg_control == "position":
            target_dof_pos = self.actions * self.action_scale + self.default_dof_pos
            self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        elif self.leg_control == "torque":
            # Get current joint positions and velocities
            self.dof_pos = self.robot.get_dofs_position(self.motor_dofs)
            self.dof_vel = self.robot.get_dofs_velocity(self.motor_dofs)
            
            # Calculate joint targets (same as act2joint)
            target_dof_pos = self.actions * self.action_scale + self.default_dof_pos
            
            # Apply PD control (same as act2tau)
            position_error = target_dof_pos - self.dof_pos
            torques = self.kp * position_error - self.kd * self.dof_vel
            
            # Clip torques to joint limits
            torques = torch.clamp(
                torques,
                self.joint_torque_range[:, 0],
                self.joint_torque_range[:, 1]
            )
            
            # Apply torques
            self.robot.control_dofs_force(torques, self.motor_dofs)
            self.dof_torque = torques
        
        # Step the simulation
        self.scene.step()

        # Update state information
        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        
        # Calculate base orientation in Euler angles
        self.base_euler = gs_quat2euler(self.base_quat)
        
        # Get velocities in local frame
        inv_base_quat = gs_quat_conjugate(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        
        # Update joint positions and velocities
        self.dof_pos = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel = self.robot.get_dofs_velocity(self.motor_dofs)
        
        # Update target position and orientation based on command
        progress = torch.clamp(
            self.episode_length_buf * self.dt / self.ramp_up_time, 
            max=1.0
        ).unsqueeze(-1)
        
        cur_commands = self.commands * progress
        self.target_pos += torch.cat([cur_commands[:, :2], torch.zeros((self.num_envs, 1), device=self.device)], dim=1) * self.dt
        self.target_yaw += cur_commands[:, 2] * self.dt
        
        # Update commands to match unitree_go2_env.py structure
        # commands[:, 0:2] = linear velocity (vx, vy)
        # commands[:, 2] = angular velocity (yaw rate)
        self.commands[:, 0] = cur_commands[:, 0]  # Forward velocity
        self.commands[:, 1] = cur_commands[:, 1]  # Lateral velocity
        self.commands[:, 2] = cur_commands[:, 2]  # Yaw rate
        
        # Get foot position information for contact detection and gait calculation
        self.foot_positions[:] = self.rigid_solver.get_links_pos(self.feet_link_indices_world_frame)
        
        # Get contact forces for proper contact detection
        link_contact_forces = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=torch.float32,
        )

        # Check termination conditions
        self.reset_buf = torch.any(
            torch.norm(
                link_contact_forces[:, self.termination_contact_link_indices, :],
                dim=-1,
            )
            > 1.0,
            dim=1,
        )

        # Foot contact detection
        foot_contact = torch.zeros((self.num_envs, len(self.feet_link_indices)), dtype=torch.bool, device=self.device)
        for i, link_idx in enumerate(self.feet_link_indices):
            # Primary method: detect contact using contact forces
            contact_force = link_contact_forces[:, link_idx, 2]  # vertical force component
            force_contact = contact_force > 1.0
            
            # Backup method: use height-based detection
            foot_height = self.foot_positions[:, i, 2]
            height_contact = foot_height < 1e-3  # 1mm threshold for contact detection
            
            # Combine both methods for more robust detection
            foot_contact[:, i] = torch.logical_or(force_contact, height_contact)
        
        # Filter contacts to reduce noise
        contact_filt = torch.logical_or(foot_contact, self.last_contacts)
        
        # Detect first contact events for reward calculations
        first_contact = (~self.last_contacts) & contact_filt
        
        # Update air time tracking with vectorized operations
        self.feet_air_time += self.dt
        self.feet_air_time = torch.where(
            contact_filt, 
            torch.zeros_like(self.feet_air_time), 
            self.feet_air_time
        )
        self.last_contacts = foot_contact
        
        # Calculate rewards
        duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]
        
        # Target foot height based on gait
        z_feet_tar = get_foot_step(duty_ratio, cadence, amplitude, phases, self.episode_length_buf[0] * self.dt)
        z_feet = self.foot_positions[:, :, 2] - self._foot_radius
        
        # Compute rewards
        reward_gaits = -torch.sum(((z_feet_tar - z_feet[0]) / 0.05) ** 2)
        
        # Air time reward - only reward on first contact with the ground
        # Detect first contact events
        first_contact = (self.feet_air_time > 0.) * contact_filt
        # Reward air time only on first contact and when command velocity is non-zero
        reward_air_time = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        # Only apply reward when command velocity is non-zero
        reward_air_time *= torch.norm(self.commands[:, :2], dim=1) > 0.1
        
        # Position reward - calculate distance between head position and target position
        head_vec = torch.tensor([0.285, 0.0, 0.0], device=self.device)  # Head offset from base

        # The properly vectorized calculation for head position without using bmm
        # First, get the rotation matrices for each environment in the batch
        R = transform_by_quat(
            torch.eye(3, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1), 
            self.base_quat
        )

        # Properly reshape head_vec for broadcasting
        head_vec = head_vec.unsqueeze(0)  # Shape: [1, 3]

        # Use broadcasting to rotate the vector (avoiding bmm which requires careful reshaping)
        # This computes the matrix-vector product for each batch item
        rotated_head_vecs = torch.zeros((self.num_envs, 3), device=self.device)
        for i in range(self.num_envs):
            rotated_head_vecs[i] = R[i] @ head_vec[0]

        # Calculate head position by adding the rotated vector to the base position
        head_pos = self.base_pos + rotated_head_vecs

        # Calculate the reward as negative sum of squared differences
        reward_pos = -torch.sum((head_pos - self.target_pos) ** 2, dim=1)
        
        # Upright reward
        up_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        current_up = transform_by_quat(up_vec.unsqueeze(0).repeat(self.num_envs, 1), self.base_quat)
        reward_upright = -torch.sum((current_up - up_vec.unsqueeze(0)) ** 2, dim=1)
        
        # Yaw orientation reward
        yaw = self.base_euler[:, 2]
        d_yaw = yaw - self.target_yaw
        reward_yaw = -torch.square(torch.atan2(torch.sin(d_yaw), torch.cos(d_yaw)))
        
        # Velocity reward
        reward_vel = -torch.sum((self.base_lin_vel[:, :2] - cur_commands[:, :2]) ** 2, dim=1)
        reward_ang_vel = -torch.sum((self.base_ang_vel[:, 2:] - cur_commands[:, 2:]) ** 2, dim=1)
        
        # Height reward
        reward_height = -torch.sum((self.base_pos[:, 2:] - self.target_pos[:, 2:]) ** 2, dim=1)
        
        # Energy reward (only if using torque control)
        if self.leg_control == "torque":
            reward_energy = -torch.sum(torch.maximum(self.dof_torque * self.dof_vel / 160.0, torch.zeros_like(self.dof_torque)) ** 2, dim=1)
        else:
            reward_energy = torch.zeros(self.num_envs, device=self.device)
        
        # Alive reward
        reward_alive = 1.0 - self.reset_buf.float()
        
        # Combine rewards with weighting
        self.rew_buf = (
            reward_gaits * 0.1     
            + reward_air_time * 0.0 
            + reward_pos * 0.0      
            + reward_upright * 0.5  
            + reward_yaw * 0.3      
            + reward_vel * 1.0     
            + reward_ang_vel * 1.0 
            + reward_height * 1.0  
            + reward_energy * 0.0   
            + reward_alive * 0.0    
        )
        
        # Check termination conditions
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Check if base has fallen over - vectorized
        base_up = transform_by_quat(up.unsqueeze(0).repeat(self.num_envs, 1), self.base_quat)
        orientation_done = (base_up[:, 2] < 0)  # Z component should be positive for upright
        
        # Check if joints are out of range - vectorized
        joint_min_done = torch.any(self.dof_pos < self.joint_range[:, 0], dim=1)
        joint_max_done = torch.any(self.dof_pos > self.joint_range[:, 1], dim=1)
        
        # Check if base is too low
        height_done = (self.base_pos[:, 2] < 0.18)
        
        # Combine all termination conditions
        done = done | orientation_done | joint_min_done | joint_max_done | height_done
        
        # Convert bool to int for compatibility
        self.reset_buf = done.int()
        
        # Reset environments that are done
        if torch.any(done):
            reset_idx = torch.where(done)[0]
            print(f"[DEBUG] Resetting {torch.sum(done).item()}/{self.num_envs} environments at step {self.episode_length_buf[0].item()}")
            print(f"[DEBUG] Base height: {self.base_pos[0, 2]:.4f}, Orientation Z: {base_up[0, 2]:.4f}")
            print(f"[DEBUG] Joint positions - min: {self.dof_pos.min().item():.4f}, max: {self.dof_pos.max().item():.4f}")
            
            # Detailed termination reason
            for idx in reset_idx:
                reasons = []
                if orientation_done[idx]:
                    reasons.append(f"orientation={base_up[idx, 2]:.4f}")
                if joint_min_done[idx]:
                    min_val, min_idx = torch.min(self.dof_pos[idx] - self.joint_range[:, 0], dim=0)
                    reasons.append(f"min_joint={min_idx.item()} (violated by {-min_val.item():.4f})")
                if joint_max_done[idx]:
                    max_val, max_idx = torch.max(self.dof_pos[idx] - self.joint_range[:, 1], dim=0)
                    reasons.append(f"max_joint={max_idx.item()} (violated by {max_val.item():.4f})")
                if height_done[idx]:
                    reasons.append(f"height={self.base_pos[idx, 2]:.4f}")
                reason_str = ", ".join(reasons)
                print(f"[DEBUG] Reset reason for env {idx}: {reason_str}")
            
            self.reset_idx(reset_idx)
        
        # Update buffers
        self.last_actions = self.actions.clone()
        self.last_dof_vel = self.dof_vel.clone()
        
        # Get observations
        self.obs_buf = self.get_obs()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
        
    def get_obs(self):
        """Get the current observation"""
        # Transform global velocities to body frame
        vb = global_to_body_velocity(self.base_lin_vel, self.base_quat)
        ab = global_to_body_velocity(self.base_ang_vel * math.pi / 180.0, self.base_quat)  # Added angle conversion
        
        # Create observations - Match unitree_go2_env.py structure
        obs = torch.cat([
            # Target velocities (vel_tar)
            self.commands[:, :3],        # Linear velocity targets
            
            # Target angular velocities (ang_vel_tar)
            self.commands[:, 2:],        # Angular velocity targets (yaw rate)
            
            # Control signals (ctrl)
            self.actions,                # Current control actions
            
            # Position states (qpos)
            torch.cat([                  # Combine base and joint positions
                self.base_pos,           # Base position (3)
                self.base_quat,          # Base orientation (4)
                self.dof_pos,            # Joint positions (12)
            ], dim=1),
            
            # Body-frame velocities (vb)
            vb,                          # Linear velocity in body frame
            
            # Body-frame angular velocities (ab)
            ab,                          # Angular velocity in body frame
            
            # Joint velocities (qvel[6:])
            self.dof_vel,                # Joint velocities (12)
        ], dim=1)
        
        return obs

    def reset(self):
        """Reset all environments"""
        self.reset_buf[:] = 1
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        
        # Get observations after reset
        obs = self.get_obs()
        return obs, {}
    
    def reset_idx(self, env_idx):
        """Reset specified environments"""
        if len(env_idx) == 0:
            return

        print(f"[DEBUG] Performing reset on environments: {env_idx.tolist()}")
        
        # Reset DOFs
        self.dof_pos[env_idx] = self.default_dof_pos
        self.dof_vel[env_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[env_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=env_idx,
        )
        
        # Reset base
        self.base_pos[env_idx] = self.base_init_pos
        self.base_quat[env_idx] = self.base_init_quat.unsqueeze(0)
        self.robot.set_pos(self.base_pos[env_idx], zero_velocity=True, envs_idx=env_idx)
        self.robot.set_quat(self.base_quat[env_idx], zero_velocity=True, envs_idx=env_idx)
        
        # Reset velocities
        self.base_lin_vel[env_idx] = 0.0
        self.base_ang_vel[env_idx] = 0.0
        
        # Reset buffers
        self.last_actions[env_idx] = 0.0
        self.last_dof_vel[env_idx] = 0.0
        self.feet_air_time[env_idx] = 0.0
        self.last_contacts[env_idx] = False
        self.episode_length_buf[env_idx] = 0
        
        # Reset target positions
        self.target_pos[env_idx, 0] = 0.282
        self.target_pos[env_idx, 1] = 0.0
        self.target_pos[env_idx, 2] = 0.3
        self.target_yaw[env_idx] = 0.0
                
        # Resample commands if randomize_tasks is true
        if self.randomize_tasks:
            new_lin_vel_cmd, new_ang_vel_cmd = self.sample_command(len(env_idx))
            self.commands[env_idx, 0] = new_lin_vel_cmd[:, 0]  # x velocity
            self.commands[env_idx, 1] = new_lin_vel_cmd[:, 1]  # y velocity
            self.commands[env_idx, 2] = new_ang_vel_cmd[:, 2]  # yaw rate
        else:
            self.commands[env_idx, 0] = self.default_vx
            self.commands[env_idx, 1] = self.default_vy
            self.commands[env_idx, 2] = self.default_vyaw
            
        # Update last_reset_reason
        self.last_reset_reason = f"Reset due to termination conditions. Reset environments: {env_idx.tolist()}"
        
    def sample_command(self, num_samples=None):
        """Sample a random command if randomize_tasks is enabled."""
        if num_samples is None:
            num_samples = self.num_envs
            
        lin_vel_x = gs_rand_float(-1.5, 1.5, (num_samples,), self.device)
        lin_vel_y = gs_rand_float(-0.5, 0.5, (num_samples,), self.device)
        ang_vel_yaw = gs_rand_float(-1.5, 1.5, (num_samples,), self.device)
        
        new_lin_vel_cmd = torch.zeros((num_samples, 3), device=self.device)
        new_lin_vel_cmd[:, 0] = lin_vel_x
        new_lin_vel_cmd[:, 1] = lin_vel_y
        
        new_ang_vel_cmd = torch.zeros((num_samples, 3), device=self.device)
        new_ang_vel_cmd[:, 2] = ang_vel_yaw
        
        return new_lin_vel_cmd, new_ang_vel_cmd

    def save_state(self):
        """Save the current environment state for later restoration."""
        # print(f"[DEBUG RESTORE] Saving environment state")
        self._saved_state = {
            # Robot state
            'dof_pos': self.dof_pos.clone(),
            'dof_vel': self.dof_vel.clone(),
            'base_pos': self.base_pos.clone(),
            'base_quat': self.base_quat.clone(),
            'base_lin_vel': self.base_lin_vel.clone(),
            'base_ang_vel': self.base_ang_vel.clone(),
            
            # Environment state
            'episode_length_buf': self.episode_length_buf.clone(),
            'last_actions': self.last_actions.clone(),
            'last_dof_vel': self.last_dof_vel.clone(),
            'feet_air_time': self.feet_air_time.clone(),
            'last_contacts': self.last_contacts.clone(),
            'target_pos': self.target_pos.clone(),
            'target_yaw': self.target_yaw.clone(),
        }
        
    def restore_state(self):
        """Restore the environment to the previously saved state."""
        if self._saved_state is None:
            print("[DEBUG RESTORE] Warning: No saved state to restore!")
            return
            
        # print(f"[DEBUG RESTORE] Restoring environment state")
        
        # Restore robot state
        self.dof_pos = self._saved_state['dof_pos'].clone()
        self.dof_vel = self._saved_state['dof_vel'].clone()
        self.base_pos = self._saved_state['base_pos'].clone()
        self.base_quat = self._saved_state['base_quat'].clone()
        self.base_lin_vel = self._saved_state['base_lin_vel'].clone()
        self.base_ang_vel = self._saved_state['base_ang_vel'].clone()
        
        # Set positions in physics engine
        self.robot.set_dofs_position(
            position=self.dof_pos,
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True
        )
        self.robot.set_pos(self.base_pos, zero_velocity=True)
        self.robot.set_quat(self.base_quat, zero_velocity=True)
        
        # Restore environment state
        self.episode_length_buf = self._saved_state['episode_length_buf'].clone()
        self.last_actions = self._saved_state['last_actions'].clone()
        self.last_dof_vel = self._saved_state['last_dof_vel'].clone()
        self.feet_air_time = self._saved_state['feet_air_time'].clone()
        self.last_contacts = self._saved_state['last_contacts'].clone()
        self.target_pos = self._saved_state['target_pos'].clone()
        self.target_yaw = self._saved_state['target_yaw'].clone()
