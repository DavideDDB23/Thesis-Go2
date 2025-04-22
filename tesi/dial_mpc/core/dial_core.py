import os
import time
from dataclasses import dataclass
import importlib
import sys
import math
import numpy as np
import torch
import yaml
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

# Add project root to path for imports to work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import genesis as gs

# Import our own configuration and environment using relative imports
from tesi.dial_mpc.core.dial_config import DialConfig
from tesi.dial_mpc.envs.go2_env import UnitreeGo2Env, UnitreeGo2EnvConfig
from tesi.dial_mpc.utils.utils import global_to_body_velocity, body_to_global_velocity

# Try to use science plots if available
try:
    import scienceplots
    plt.style.use("science")
except ImportError:
    pass

try:
    import art
    import emoji
except ImportError:
    pass


def rollout_us(step_env, state, us, device):
    """Rollout a sequence of actions and return rewards and states."""
    rewards = []
    states = []
    
    obs = state['obs']
    for u in us:
        obs, reward, done, info = step_env(u)
        rewards.append(reward)
        states.append(info)
    
    return torch.tensor(rewards, device=device), states


def softmax_update(weights, Y0s, sigma, mu_0t):
    """Update the mean of the distribution using softmax weights."""
    # Weighted mean using softmax weights
    mu_0tm1 = torch.einsum("n,nij->ij", weights, Y0s)
    return mu_0tm1, sigma


class MBDPI:
    def __init__(self, args: DialConfig, env):
        self.args = args
        self.env = env
        self.device = env.device
        self.nu = env.num_actions  # Number of actions

        # Select update method
        self.update_fn = {
            "mppi": softmax_update,
        }[args.update_method]

        # Setup diffusion schedule
        sigma0 = 1e-2
        sigma1 = 1.0
        A = sigma0
        B = math.log(sigma1 / sigma0) / args.Ndiffuse
        self.sigmas = A * torch.exp(torch.tensor([B * i for i in range(args.Ndiffuse)], device=self.device))
        
        # Setup control noise schedule
        self.sigma_control = (
            args.horizon_diffuse_factor ** torch.arange(args.Hnode + 1, device=self.device)
        ).flip(0)  # Reverse the tensor using flip instead of [::-1]

        # Setup time discretization
        self.ctrl_dt = 0.02  # Control timestep
        self.step_us = torch.linspace(0, self.ctrl_dt * args.Hsample, args.Hsample + 1, device=self.device)
        self.step_nodes = torch.linspace(0, self.ctrl_dt * args.Hsample, args.Hnode + 1, device=self.device)
        self.node_dt = self.ctrl_dt * (args.Hsample) / (args.Hnode)

    def node2u(self, nodes):
        """Convert control nodes to dense control sequence using interpolation."""
        # Use scipy for interpolation since we're working with PyTorch tensors
        nodes_np = nodes.cpu().numpy()
        step_nodes_np = self.step_nodes.cpu().numpy()
        step_us_np = self.step_us.cpu().numpy()
        
        # Create interpolation function for each output dimension
        us_np = np.zeros((step_us_np.shape[0], nodes_np.shape[1]))
        for i in range(nodes_np.shape[1]):
            # Ensure x is strictly increasing (requirement for spline)
            interp_func = InterpolatedUnivariateSpline(step_nodes_np, nodes_np[:, i], k=2)
            us_np[:, i] = interp_func(step_us_np)
        
        # Convert back to torch tensor
        us = torch.tensor(us_np, device=self.device, dtype=torch.float32)
        return us

    def u2node(self, us):
        """Convert dense control sequence to control nodes using interpolation."""
        # Use scipy for interpolation
        us_np = us.cpu().numpy()
        step_us_np = self.step_us.cpu().numpy()
        step_nodes_np = self.step_nodes.cpu().numpy()
        
        # Create interpolation function for each output dimension
        nodes_np = np.zeros((step_nodes_np.shape[0], us_np.shape[1]))
        for i in range(us_np.shape[1]):
            # Ensure x is strictly increasing (requirement for spline)
            interp_func = InterpolatedUnivariateSpline(step_us_np, us_np[:, i], k=2)
            nodes_np[:, i] = interp_func(step_nodes_np)
        
        # Convert back to torch tensor
        nodes = torch.tensor(nodes_np, device=self.device, dtype=torch.float32)
        return nodes

    def reverse_once(self, state, Ybar_i, noise_scale):
        """Perform one reverse diffusion step to improve the control sequence."""
        # Sample from distribution
        eps_Y = torch.randn(
            (self.args.Nsample, self.args.Hnode + 1, self.nu), 
            device=self.device
        )
        Y0s = eps_Y * noise_scale[None, :, None] + Ybar_i
        
        # Fix the first control completely (matching JAX implementation)
        Y0s[:, 0] = Ybar_i[0]
        
        # Append Y0s with Ybar_i to also evaluate Ybar_i
        Y0s = torch.cat([Y0s, Ybar_i.unsqueeze(0)], dim=0)
        Y0s = torch.clamp(Y0s, -1.0, 1.0)
        
        # Convert Y0s to dense control sequences
        us_list = []
        for sample_idx in range(Y0s.shape[0]):
            us_list.append(self.node2u(Y0s[sample_idx]))
        
        # Now do all rollouts
        all_rewards = []
        all_states = []
        for sample_idx in range(len(us_list)):
            # Reset environment to current state
            env_copy = self.env.clone() if hasattr(self.env, 'clone') else self.env
            env_copy.reset()
            env_copy.dof_pos = state['dof_pos'].clone()
            env_copy.dof_vel = state['dof_vel'].clone() 
            env_copy.base_pos = state['base_pos'].clone()
            env_copy.base_quat = state['base_quat'].clone()
            env_copy.base_lin_vel = state['base_lin_vel'].clone()
            env_copy.base_ang_vel = state['base_ang_vel'].clone()
            
            # Roll out trajectory
            rewards = []
            states = {"q": [], "qd": [], "x": []}
            
            # Apply all controls in sequence
            for u_step in range(us_list[sample_idx].shape[0]):
                action = us_list[sample_idx][u_step]
                obs, reward, done, info = env_copy.step(action.unsqueeze(0))
                
                rewards.append(reward.item())
                states["q"].append(env_copy.dof_pos.clone())
                states["qd"].append(env_copy.dof_vel.clone())
                states["x"].append(env_copy.base_pos.clone())
            
            all_rewards.append(rewards)
            all_states.append(states)
        
        # Convert rewards to tensor
        all_rewards = torch.tensor(all_rewards, device=self.device)
        
        # Calculate reward for nominal trajectory (Ybar_i)
        rew_Ybar_i = all_rewards[-1].mean()
        
        # Extract state trajectories
        qss = torch.stack([torch.stack(states["q"]) for states in all_states])
        qdss = torch.stack([torch.stack(states["qd"]) for states in all_states])
        xss = torch.stack([torch.stack(states["x"]) for states in all_states])
        
        # Calculate mean rewards for each trajectory (matching JAX implementation)
        rews = all_rewards.mean(dim=1)  # Mean over timesteps
        
        # Calculate softmax weights based on rewards - match JAX implementation exactly
        reward_std = rews.std() if rews.std() > 0 else 1.0
        logp0 = (rews - rew_Ybar_i) / reward_std / self.args.temp_sample
        weights = torch.softmax(logp0, dim=0)
        
        # Update control nodes using weighted average - match JAX implementation
        Ybar = torch.einsum("n,nij->ij", weights, Y0s)
        
        # Match JAX implementation for state trajectories averaging
        qbar = torch.einsum("n,nij->ij", weights, qss)
        qdbar = torch.einsum("n,nij->ij", weights, qdss)
        xbar = torch.einsum("n,nijk->ijk", weights, xss)
        
        # Update noise scale (match JAX behavior)
        new_noise_scale = noise_scale
        
        # Prepare info dict to match JAX implementation
        info = {
            "rews": rews,
            "qbar": qbar,
            "qdbar": qdbar,
            "xbar": xbar,
            "new_noise_scale": new_noise_scale,
        }
        
        return Ybar, info

    def reverse(self, state, YN):
        """Perform the full reverse diffusion process."""
        Yi = YN
        with tqdm(range(self.args.Ndiffuse - 1, 0, -1), desc="Diffusing") as pbar:
            for i in pbar:
                t0 = time.time()
                Yi, info = self.reverse_once(
                    state, Yi, self.sigmas[i] * torch.ones(self.args.Hnode + 1, device=self.device)
                )
                freq = 1 / (time.time() - t0)
                pbar.set_postfix({"rew": f"{info['rews'].mean().item():.2e}", "freq": f"{freq:.2f}"})
        return Yi

    def shift(self, Y):
        """Shift the control sequence forward by one step."""
        u = self.node2u(Y)
        u = torch.roll(u, -1, dims=0)
        u[-1] = torch.zeros(self.nu, device=self.device)
        Y = self.u2node(u)
        return Y

    def shift_Y_from_u(self, u, n_step):
        """Shift the control sequence forward by n steps."""
        u = torch.roll(u, -n_step, dims=0)
        u[-n_step:] = torch.zeros_like(u[-n_step:])
        Y = self.u2node(u)
        return Y


def main():
    """Main function to run DIAL-MPC with Genesis environment."""
    # Print banner if art library is available
    try:
        art.tprint("Genesis\nDIAL-MPC", font="big", chr_ignore=True)
    except:
        print("=" * 40)
        print("Genesis DIAL-MPC")
        print("=" * 40)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="tesi/dial_mpc/examples/unitree_go2_trot.yaml", 
                       help="Path to config file")
    args = parser.parse_args()
    
    # Initialize Genesis
    gs.init()
    
    # Load configuration
    config_dict = yaml.safe_load(open(args.config))
    
    # Create DialConfig object from dictionary
    dial_config = DialConfig()
    for key, value in config_dict.items():
        if hasattr(dial_config, key):
            setattr(dial_config, key, value)
    
    # Set random seed
    torch.manual_seed(dial_config.seed)
    np.random.seed(dial_config.seed)
    
    # Create environment config
    env_cfg = UnitreeGo2EnvConfig()
    
    # Update environment config from YAML
    for key, value in config_dict.items():
        if hasattr(env_cfg, key):
            setattr(env_cfg, key, value)
    
    print("📋 Creating environment with config:")
    print(f"  - Control: {env_cfg.leg_control}")
    print(f"  - Gait: {env_cfg.gait}")
    print(f"  - Default velocity: [{env_cfg.default_vx}, {env_cfg.default_vy}, {env_cfg.default_vyaw}]")
    
    # Create environment
    env = UnitreeGo2Env(
        num_envs=1,
        env_cfg=env_cfg,
        show_viewer=True,
    )
    
    # Set DPI scale for better quality
    env.scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 1
    
    # Run simulation in a separate thread to make viewer work on macOS
    import threading
    sim_thread = threading.Thread(target=run_simulation, args=(env, dial_config))
    sim_thread.daemon = True
    sim_thread.start()
    
    # Start the viewer
    print("Starting viewer - close the viewer window to exit")
    env.scene.viewer.start()


def run_simulation(env, dial_config):
    
    # Create MBDPI controller
    mbdpi = MBDPI(dial_config, env)
    
    # Reset environment
    print("[DEBUG] Resetting environment before starting simulation")
    obs, _ = env.reset()
    
    # Initialize control nodes with zeros (matching the JAX implementation)
    YN = torch.zeros([dial_config.Hnode + 1, mbdpi.nu], device=env.device)
    
    # Start with initial control sequence
    Y0 = YN
    
    Nstep = dial_config.n_steps
    rewards = []
    rewards_plan = []
    rollout = []
    us = []
    infos = []
    
    # Get current state
    state = {
        'obs': obs,
        'dof_pos': env.dof_pos.clone(),
        'dof_vel': env.dof_vel.clone(),
        'base_pos': env.base_pos.clone(), 
        'base_quat': env.base_quat.clone(),
        'base_lin_vel': env.base_lin_vel.clone(),
        'base_ang_vel': env.base_ang_vel.clone(),
    }
    
    print(f"[DEBUG] Initial state: base_pos={env.base_pos[0]}, base_height={env.base_pos[0, 2]:.4f}")
    print(f"[DEBUG] Initial velocity: {env.base_lin_vel[0]}")
        
    with tqdm(range(Nstep), desc="Rollout") as pbar:
        for t in pbar:
            # Apply first control from the sequence
            action = Y0[0].unsqueeze(0)  # Add batch dimension
            print(f"[DEBUG] Step {t}: Applying action, min={action.min().item():.3f}, max={action.max().item():.3f}, mean={action.mean().item():.3f}")
            
            next_obs, reward, done, info = env.step(action)
            
            print(f"[DEBUG] Step {t}: After action - reward={reward.item():.4f}, done={done.item()}")
            print(f"[DEBUG] Base height={env.base_pos[0, 2]:.4f}, velocity={env.base_lin_vel[0, 0]:.4f}")
                        
            # Record state and reward
            rollout.append({
                'dof_pos': env.dof_pos.clone(),
                'dof_vel': env.dof_vel.clone(),
                'base_pos': env.base_pos.clone(),
                'quat': env.base_quat.clone(),
                'ctrl': action.clone(),
            })
            rewards.append(reward.item())
            us.append(Y0[0].cpu().numpy())
            
            # Check if the step resulted in a reset
            if done.item():
                print(f"[DEBUG] Environment was reset during step {t}")
            
            # Update state
            state = {
                'obs': next_obs,
                'dof_pos': env.dof_pos.clone(),
                'dof_vel': env.dof_vel.clone(),
                'base_pos': env.base_pos.clone(), 
                'base_quat': env.base_quat.clone(),
                'base_lin_vel': env.base_lin_vel.clone(),
                'base_ang_vel': env.base_ang_vel.clone(),
            }
            
            # Shift control sequence forward
            Y0_before = Y0.clone()
            Y0 = mbdpi.shift(Y0)
            
            # Determine number of diffusion steps
            n_diffuse = dial_config.Ndiffuse
            if t == 0:
                n_diffuse = dial_config.Ndiffuse_init
            
            # Optimize control sequence
            t0 = time.time()
            
            print(f"[DEBUG] Starting optimization with {n_diffuse} diffusion steps")
            
            # Show optimization progress
            for i in range(n_diffuse):
                traj_diffuse_factor = dial_config.traj_diffuse_factor ** i
                noise_scale = mbdpi.sigma_control * traj_diffuse_factor
                Y0, info = mbdpi.reverse_once(state, Y0, noise_scale)
                print(f"[DEBUG] Diffusion step {i+1}/{n_diffuse}, avg reward: {info['rews'].mean().item():.4f}")

            rewards_plan.append(info["rews"].mean().item())
            infos.append(info)
            
            # Calculate and display planning frequency
            freq = 1 / (time.time() - t0)
            pbar.set_postfix({"rew": f"{reward.item():.2e}", "freq": f"{freq:.2f}"})
            
            # Handle episode termination
            if done.item():
                print(f"[DEBUG] Episode terminated at step {t}")
                # Reset environment
                state['obs'], _ = env.reset()
                state['dof_pos'] = env.dof_pos.clone()
                state['dof_vel'] = env.dof_vel.clone()
                state['base_pos'] = env.base_pos.clone()
                state['base_quat'] = env.base_quat.clone()
                state['base_lin_vel'] = env.base_lin_vel.clone()
                state['base_ang_vel'] = env.base_ang_vel.clone()
                
    # Calculate average reward
    mean_reward = np.mean(rewards)
    print(f"Mean reward: {mean_reward:.4e}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(dial_config.output_dir):
        os.makedirs(dial_config.output_dir)
    
    # Generate timestamp for output files
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save rollout data
    rollout_data = []
    for i, roll in enumerate(rollout):
        # Convert to numpy arrays
        dof_pos = roll['dof_pos'][0].cpu().numpy()
        dof_vel = roll['dof_vel'][0].cpu().numpy() 
        base_pos = roll['base_pos'][0].cpu().numpy()
        ctrl = roll['ctrl'][0].cpu().numpy()
        
        # Combine into a single array
        step_data = np.concatenate([
            np.array([i]),
            dof_pos,
            dof_vel,
            base_pos,
            ctrl
        ])
        rollout_data.append(step_data)
    
    # Save as numpy array
    rollout_data = np.array(rollout_data)
    np.save(os.path.join(dial_config.output_dir, f"{timestamp}_states"), rollout_data)
    
    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title(f"Rewards over {Nstep} steps")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(dial_config.output_dir, f"{timestamp}_rewards.pdf"))
    
    print(f"Test completed successfully!")
    print(f"Results saved to {dial_config.output_dir}")


if __name__ == "__main__":
    main() 