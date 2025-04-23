import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from dataclasses import dataclass
from typing import Union, Any, Dict
import hashlib


from ..utils.utils import *
from ..utils.function_utils import *
from ..config.base_env_config import BaseEnvConfig
from .base_env import BaseEnv


@dataclass
class UnitreeGo2EnvConfig(BaseEnvConfig):
    kp: Union[float, torch.Tensor] = 30.0
    kd: Union[float, torch.Tensor] = 0.0
    default_vx: float = 1.0
    default_vy: float = 0.0
    default_vyaw: float = 0.0
    ramp_up_time: float = 2.0
    gait: str = "trot"
    n_envs: int = 1

class UnitreeGo2Env(BaseEnv):
    def __init__(self, config: UnitreeGo2EnvConfig):
        self._init_base_pos = torch.tensor([0.0, 0.0, 0.27], dtype=torch.float32)
        self._init_base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        self._init_joint_pos = torch.tensor([
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8
        ], dtype=torch.float32)
        self.device = torch.device(config.backend if config.backend != "mps" or torch.backends.mps.is_available() else "cpu")
        super().__init__(config)
        
        self._foot_radius = 0.0175

        self._gait = config.gait
        self._gait_phase = {
            "stand": torch.zeros(4),
            "walk": torch.tensor([0.0, 0.5, 0.75, 0.25]),
            "trot": torch.tensor([0.0, 0.5, 0.5, 0.0]),
            "canter": torch.tensor([0.0, 0.33, 0.33, 0.66]),
            "gallop": torch.tensor([0.0, 0.05, 0.4, 0.35]),
        }
        self._gait_params = {
            #                  ratio, cadence, amplitude
            "stand": torch.tensor([1.0, 1.0, 0.0]),
            "walk": torch.tensor([0.75, 1.0, 0.08]),
            "trot": torch.tensor([0.45, 2, 0.08]),
            "canter": torch.tensor([0.4, 4, 0.06]),
            "gallop": torch.tensor([0.3, 3.5, 0.10]),
        }

        # move gait definitions to the correct device
        for k, v in self._gait_phase.items():
            self._gait_phase[k] = v.to(self.device)
            
        for k, v in self._gait_params.items():
            self._gait_params[k] = v.to(self.device)

        self._init_q = self.robot.get_qpos()
        self._default_pose = self.robot.get_qpos()[7:]

        print(self._init_q)
        print(self._default_pose)

        self.joint_range = torch.tensor(
            [
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -0.85],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -0.85],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -1.3],
                [-0.5, 0.5],
                [0.4, 1.4],
                [-2.3, -1.3],
            ],
            device = self.device
        )

        self.setup_feet_links()
        
        # number of control inputs for the policy (one per motor DOF)
        self.action_size = len(self.motor_dofs)
        
        # number of parallel environments
        self.n_envs = config.n_envs
        
    def create_robot(self):
        robot = self.scene.add_entity(
            gs.morphs.URDF(
                file= 'urdf/go2/urdf/go2.urdf',
                merge_fixed_links=True,
                links_to_keep= ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot',],
                pos=self._init_base_pos.cpu().numpy(),
                quat=self._init_base_quat.cpu().numpy(),
            ),
        )
        return robot

    # Set up references to foot links (equivalent to MuJoCo's feet_site)
    def setup_feet_links(self):
        """
        Store references to foot links for easier access
        This is equivalent to MuJoCo's feet_site_id
        """
        # Define the foot link names (same as in links_to_keep)
        self._feet_names = [
            "FL_foot",
            "FR_foot",
            "RL_foot",
            "RR_foot",
        ]
        
        # Get references to the foot links
        self._feet_links = []
        for foot_name in self._feet_names:
            foot_link = self.robot.get_link(foot_name)
            assert foot_link is not None, f"Link {foot_name} not found."
            self._feet_links.append(foot_link)
            
    def reset(self) -> dict:
        # broadcast home pose across all envs
        n = self.n_envs
        pos_batch = self._init_base_pos.unsqueeze(0).repeat(n, 1)
        quat_batch = self._init_base_quat.unsqueeze(0).repeat(n, 1)
        joint_batch = self._init_joint_pos.unsqueeze(0).repeat(n, 1)
        self.robot.set_pos(pos_batch, zero_velocity=True)
        self.robot.set_quat(quat_batch, zero_velocity=True)
        self.robot.set_dofs_position(
            position=joint_batch,
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True
        )

        # Get current state from Genesis scene
        state = self.scene.get_state()

        # initialize batched state_info
        state_info = {
            "pos_tar": torch.tensor([0.282, 0.0, 0.3], device=self.device),
            "vel_tar": torch.zeros(3, device=self.device),
            "ang_vel_tar": torch.zeros(3, device=self.device),
            "yaw_tar": torch.tensor(0.0, device=self.device),
            "step": torch.tensor(0, device=self.device),
            "z_feet": torch.zeros(4, device=self.device),
            "z_feet_tar": torch.zeros(4, device=self.device),
            "randomize_target": self._config.randomize_tasks,
            "last_contact": torch.zeros(4, dtype=torch.bool, device=self.device),
            "feet_air_time": torch.zeros(4, device=self.device),
        }

        # batch each entry in state_info
        for k, v in list(state_info.items()):
            if torch.is_tensor(v):
                if v.ndim == 0:
                    state_info[k] = v.repeat(n)
                else:
                    state_info[k] = v.unsqueeze(0).repeat(n, *([1]*v.dim()))
        # initialize RNG key for command randomization
        state_info["rng"] = torch.tensor(0, dtype=torch.int64, device=self.device)
        # initialize batched control targets
        self._current_targets = torch.zeros(n, len(self.motor_dofs), device=self.device)
        # Get observation
        obs = self._get_obs(state, state_info)  # [n_envs, obs_dim]

        # Return batched state information
        return {
            "state": state,                   # Genesis sim state holds n_envs copies
            "obs": obs,                       # [n_envs, obs_dim]
            "state_info": state_info,         # each entry [n_envs,...]
            "reward": torch.zeros(n, device=self.device),
            "done": torch.zeros(n, dtype=torch.float32, device=self.device),
            "metrics": {}
        }
    
    def step(self, state: dict, action: torch.Tensor) -> dict:
        # 1. Broadcast and apply control -------------------------------------------------
        n_envs = self._config.n_envs

        # Convert actions to joint targets/torques (broadcast inside helper)
        joint_targets = self.act2joint(action)  # [n_envs, nu]

        if self._config.leg_control == "position":
            ctrl = joint_targets
            self.robot.set_dofs_position(ctrl, dofs_idx_local=self.motor_dofs)
        else:
            ctrl = self.act2tau(action)  # [n_envs, nu]
            self.robot.control_dofs_force(force=ctrl, dofs_idx_local=self.motor_dofs)

        # keep for observation
        self._current_targets = ctrl  # [n_envs, nu]

        # step simulation ----------------------------------------------------------------
        self.scene.step()
        new_state = self.scene.get_state()

        # -------------------------------------------------------------------------------
        # OBSERVATION (must use updated state_info for targets)
        # -------------------------------------------------------------------------------
        obs = self._get_obs(new_state, state["state_info"])  # [n_envs, obs_dim]

        # -------------------------------------------------------------------------------
        # COMMAND TARGETS (velocity commands)
        # -------------------------------------------------------------------------------
        step_count = state["state_info"]["step"]  # [n_envs]

        # mask of envs that should receive new random command this step
        randomize_mask = state["state_info"]["randomize_target"] & ((step_count % 500) == 0)

        # default command tensors repeated for all envs
        default_lin_cmd = torch.tensor([
            self._config.default_vx,
            self._config.default_vy,
            0.0,
        ], device=self.device).expand(n_envs, -1)
        default_ang_cmd = torch.tensor([0.0, 0.0, self._config.default_vyaw], device=self.device).expand(n_envs, -1)

        # sample random commands for all envs (then we will mask)
        lin_vel_x = torch.rand(n_envs, device=self.device) * (1.5 - (-1.5)) + (-1.5)
        lin_vel_y = torch.rand(n_envs, device=self.device) * (0.5 - (-0.5)) + (-0.5)
        ang_vel_yaw = torch.rand(n_envs, device=self.device) * (1.5 - (-1.5)) + (-1.5)

        rand_lin_cmd = torch.stack([lin_vel_x, lin_vel_y, torch.zeros_like(lin_vel_x)], dim=1)
        rand_ang_cmd = torch.stack([torch.zeros_like(ang_vel_yaw), torch.zeros_like(ang_vel_yaw), ang_vel_yaw], dim=1)

        # choose between default and random based on mask
        vel_tar = torch.where(randomize_mask.unsqueeze(1), rand_lin_cmd, default_lin_cmd)  # [n_envs,3]
        ang_vel_tar = torch.where(randomize_mask.unsqueeze(1), rand_ang_cmd, default_ang_cmd)  # [n_envs,3]

        # ramp up targets ---------------------------------------------------------------
        ramp = (step_count.float() * self.dt / self._config.ramp_up_time).unsqueeze(1)  # [n_envs,1]
        state["state_info"]["vel_tar"] = torch.minimum(vel_tar * ramp, vel_tar)
        state["state_info"]["ang_vel_tar"] = torch.minimum(ang_vel_tar * ramp, ang_vel_tar)
        # -------------------------------------------------------------------------------
        # REWARD CALCULATIONS (vectorised over envs) ------------------------------------
        # -------------------------------------------------------------------------------

        # --- Gait reward ---------------------------------------------------------------
        z_feet = torch.stack([link.get_pos()[:, 2] for link in self._feet_links], dim=1)  # [n_envs,4]

        duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]

        time_step = step_count.float() * self.dt  # [n_envs]
        z_feet_tar = get_foot_step(duty_ratio, cadence, amplitude, phases, time_step)  # [n_envs,4]

        reward_gaits = -torch.sum(((z_feet_tar - z_feet) / 0.05) ** 2, dim=1)  # [n_envs]

        # --- Foot contact / air time reward -------------------------------------------
        foot_contact_z = z_feet - self._foot_radius
        contact = foot_contact_z < 1e-3  # [n_envs,4]
        contact_filt_mm = contact | state["state_info"]["last_contact"]
        first_contact = (state["state_info"]["feet_air_time"] > 0.0) * contact_filt_mm

        state["state_info"]["feet_air_time"] += self.dt
        reward_air_time = torch.sum((state["state_info"]["feet_air_time"] - 0.1) * first_contact.float(), dim=1)

        # --- Position reward -----------------------------------------------------------
        pos_tar = state["state_info"]["pos_tar"] + state["state_info"]["vel_tar"] * self.dt * step_count.unsqueeze(1)

        torso_link = self.robot.get_link("base")
        pos = torso_link.get_pos().to(self.device)  # [n_envs,3]
        quat = torso_link.get_quat().to(self.device)  # [n_envs,4]

        R = gs_quat_to_3x3(quat)  # [n_envs,3,3]
        head_vec = torch.tensor([0.285, 0.0, 0.0], device=self.device)
        head_pos = pos + torch.matmul(R, head_vec)  # [n_envs,3]

        reward_pos = -torch.sum((head_pos - pos_tar) ** 2, dim=1)

        # --- Upright reward ------------------------------------------------------------
        vec_tar = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(n_envs, -1)  # [n_envs,3]
        vec = gs_rotate(vec_tar, quat)  # [n_envs,3]
        reward_upright = -torch.sum((vec - vec_tar) ** 2, dim=1)

        # --- Yaw orientation reward ----------------------------------------------------
        yaw_tar = state["state_info"]["yaw_tar"] + state["state_info"]["ang_vel_tar"][:, 2] * self.dt * step_count  # [n_envs]
        yaw = gs_quat2euler(quat)[:, 2]
        d_yaw = yaw - yaw_tar
        reward_yaw = -torch.atan2(torch.sin(d_yaw), torch.cos(d_yaw)) ** 2  # [n_envs]

        # --- Velocity reward -----------------------------------------------------------
        vb = global_to_body_velocity(torso_link.get_vel().to(self.device), quat)  # [n_envs,3]
        ab = global_to_body_velocity(torso_link.get_ang().to(self.device) * (math.pi / 180.0), quat)
        reward_vel = -torch.sum((vb[:, :2] - state["state_info"]["vel_tar"][:, :2]) ** 2, dim=1)
        # For angular velocity we already have a [n_envs] vector; extra sum would raise dim error
        reward_ang_vel = -((ab[:, 2] - state["state_info"]["ang_vel_tar"][:, 2]) ** 2)

        # --- Height reward -------------------------------------------------------------
        reward_height = -torch.square(pos[:, 2] - state["state_info"]["pos_tar"][:, 2])

        # --- Energy reward -------------------------------------------------------------
        joint_vel = self.robot.get_dofs_velocity(self.motor_dofs).to(self.device)  # [n_envs,nu]
        reward_energy = -torch.sum(torch.maximum(ctrl * joint_vel / 160.0, torch.tensor(0.0, device=self.device)) ** 2, dim=1)

        # --- Alive reward --------------------------------------------------------------
        reward_alive = 1.0 - state["done"].float()

        # combine -----------------------------------------------------------------------
        reward = (
            reward_gaits * 0.1
            + reward_air_time * 0.0
            + reward_pos * 0.0
            + reward_upright * 0.5
            + reward_yaw * 0.3
            + reward_vel * 1.0
            + reward_ang_vel * 1.0
            + reward_height * 1.0
            + reward_energy * 0.00
            + reward_alive * 0.0
        )  # [n_envs]

        # -------------------------------------------------------------------------------
        # DONE --------------------------------------------------------------------------
        # -------------------------------------------------------------------------------
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(n_envs, -1)
        u_vec = gs_rotate(up, quat)  # [n_envs,3]
        done_cond = torch.sum(u_vec * up, dim=1) < 0  # [n_envs]

        # joint limits
        joint_angles = self.robot.get_dofs_position(self.motor_dofs).to(self.device)  # [n_envs,nu]
        done_cond |= torch.any(joint_angles < self.joint_range[:, 0].unsqueeze(0), dim=1)
        done_cond |= torch.any(joint_angles > self.joint_range[:, 1].unsqueeze(0), dim=1)

        # too low
        done_cond |= pos[:, 2] < 0.18

        done = done_cond.float()  # [n_envs]

        # -------------------------------------------------------------------------------
        # STATE BOOKKEEPING -------------------------------------------------------------
        # -------------------------------------------------------------------------------
        state["state_info"]["step"] += 1  # incr for all envs
        state["state_info"]["z_feet"] = z_feet
        state["state_info"]["z_feet_tar"] = z_feet_tar
        state["state_info"]["feet_air_time"] *= (~contact_filt_mm).float()
        state["state_info"]["last_contact"] = contact

        # -------------------------------------------------------------------------------
        return {
            "state": new_state,
            "obs": obs,
            "state_info": state["state_info"],
            "reward": reward,
            "done": done,
            "metrics": {},
        }
    
    def _get_obs(
        self,
        state,
        state_info: dict[str, Any],
    ) -> torch.Tensor:
        # ensure state_info tensors and current_targets are on the correct device
        for key, val in state_info.items():
            if isinstance(val, torch.Tensor):
                state_info[key] = val.to(self.device)
        self._current_targets = self._current_targets.to(self.device)
        # each of these returns shape [n_envs, dim] – move to configured device
        qpos   = self.robot.get_link('base').get_pos()
        torso_quat = self.robot.get_link("base").get_quat().to(self.device)            # [n, 4]
        torso_vel = self.robot.get_link("base").get_vel().to(self.device)              # [n, 3]
        torso_ang_vel = self.robot.get_link("base").get_ang().to(self.device)          # [n, 3]
        
        vb = global_to_body_velocity(torso_vel, torso_quat)           # [n,3]
        ab = global_to_body_velocity(torso_ang_vel * (torch.pi / 180.0), torso_quat)
        
        joint_pos = self.robot.get_dofs_position(self.motor_dofs).to(self.device)  # [n, nu]
        joint_vel = self.robot.get_dofs_velocity(self.motor_dofs).to(self.device)  # [n, nu]
        
        ctrl = self._current_targets.to(self.device)                             # [n, nu]
        
        # Make sure target values are on correct device
        vel_tar = state_info["vel_tar"].to(self.device)                          # [n,3]
        ang_vel_tar = state_info["ang_vel_tar"].to(self.device)                  # [n,3]
        
        # concatenate along feature dimension (dim=1) to form [n, obs_dim]
        obs = torch.cat([
            vel_tar,        # [n,3]
            ang_vel_tar,    # [n,3]
            ctrl,           # [n,nu]
            qpos,           # [n,nu]
            vb,             # [n,3]
            ab,             # [n,3]
            joint_vel       # [n,nu]
        ], dim=1)
        
        return obs
        
    def sample_command(self, generator: torch.Generator = None) -> tuple[torch.Tensor, torch.Tensor]:
        lin_vel_x = [-1.5, 1.5]  # min max [m/s]
        lin_vel_y = [-0.5, 0.5]  # min max [m/s]
        ang_vel_yaw = [-1.5, 1.5]  # min max [rad/s]
        
        # ensure reproducibility via provided generator
        if generator is None:
            generator = torch.Generator(device=self.device)
        # Sample random values within the specified ranges
        lin_vel_x_value = torch.rand(1, device=self.device, generator=generator) * (lin_vel_x[1] - lin_vel_x[0]) + lin_vel_x[0]
        lin_vel_y_value = torch.rand(1, device=self.device, generator=generator) * (lin_vel_y[1] - lin_vel_y[0]) + lin_vel_y[0]
        ang_vel_yaw_value = torch.rand(1, device=self.device, generator=generator) * (ang_vel_yaw[1] - ang_vel_yaw[0]) + ang_vel_yaw[0]
        
        # Create command tensors
        new_lin_vel_cmd = torch.tensor([lin_vel_x_value.item(), lin_vel_y_value.item(), 0.0], device=self.device)
        new_ang_vel_cmd = torch.tensor([0.0, 0.0, ang_vel_yaw_value.item()], device=self.device)
        
        return new_lin_vel_cmd, new_ang_vel_cmd