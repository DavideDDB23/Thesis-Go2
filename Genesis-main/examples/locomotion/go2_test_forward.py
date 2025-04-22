import argparse
import numpy as np
import torch
import time
from go2_env import Go2Env
import genesis as gs


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "dof_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 40.0,  # Increased from 20.0 for better control
        "kd": 1.0,   # Increased from 0.5 for better damping
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],  # Fixed forward velocity command
        "lin_vel_y_range": [0, 0],       # No lateral velocity
        "ang_vel_range": [0, 0],         # No turning
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def improved_trot_controller(t, default_positions, device):
    """
    Improved trot gait controller for forward locomotion
    """
    # Parameters for better gait
    freq = 2.5  # Increased frequency for faster steps
    amplitude = 0.4  # Increased amplitude for larger steps
    phase = 2 * np.pi * freq * t
    
    # Create sine waves for leg pairs (trot pattern: diagonal legs in phase)
    fr_rl_phase = np.sin(phase)
    fl_rr_phase = np.sin(phase + np.pi)  # opposite phase
    
    actions = torch.zeros(12, device=device, dtype=torch.float32)
    
    # Forward bias to encourage forward movement
    # Reversed direction and increased magnitude
    forward_bias = -0.25  # Negative for front legs = forward direction
    
    # FR leg (0, 1, 2)
    actions[0] = forward_bias - 0.2 * np.sin(phase + np.pi/4)  # Hip joint - reversed direction
    actions[1] = amplitude * fr_rl_phase  # Thigh joint
    actions[2] = -amplitude * 1.2 * fr_rl_phase - 0.1  # Calf joint with slight bend bias
    
    # FL leg (3, 4, 5)
    actions[3] = forward_bias - 0.2 * np.sin(phase + np.pi/4)  # Hip joint - reversed direction
    actions[4] = amplitude * fl_rr_phase  # Thigh joint
    actions[5] = -amplitude * 1.2 * fl_rr_phase - 0.1  # Calf joint with slight bend bias
    
    # RR leg (6, 7, 8)
    actions[6] = -forward_bias - 0.2 * np.sin(phase - np.pi/4)  # Hip joint - reversed push
    actions[7] = amplitude * fl_rr_phase  # Thigh joint
    actions[8] = -amplitude * 1.2 * fl_rr_phase - 0.1  # Calf joint with slight bend bias
    
    # RL leg (9, 10, 11)
    actions[9] = -forward_bias - 0.2 * np.sin(phase - np.pi/4)  # Hip joint - reversed push
    actions[10] = amplitude * fr_rl_phase  # Thigh joint
    actions[11] = -amplitude * 1.2 * fr_rl_phase - 0.1  # Calf joint with slight bend bias
    
    return actions


def run_sim(env, duration):
    """Function to run simulation in a separate thread for macOS"""
    # Get the device used by the environment
    device = env.device
    print(f"Environment is using device: {device}")
    
    # Reset environment
    obs, _ = env.reset()
    
    # Get default positions
    default_positions = torch.tensor([env.env_cfg["default_joint_angles"][name] for name in env.env_cfg["dof_names"]], 
                                    device=device, dtype=torch.float32)
    
    # Simulation loop
    t_start = time.time()
    t_prev = time.time()
    
    while time.time() - t_start < duration:
        # Calculate current time
        t = time.time() - t_start
        
        # Generate actions using the improved controller
        actions = improved_trot_controller(t, default_positions, device)
        
        # Since the environment expects a batch of actions for multiple environments
        actions = actions.unsqueeze(0)
        
        # Step the environment
        obs, rews, dones, infos = env.step(actions)
        
        # Print some information
        if int(t * 5) % 5 == 0:  # Print every ~1 second
            current_time = time.time()
            fps = 1.0 / (current_time - t_prev)
            lin_vel = env.base_lin_vel[0].cpu().numpy()
            print(f"t={t:.1f}s, FPS={fps:.1f}, Velocity: x={lin_vel[0]:.2f}, y={lin_vel[1]:.2f}, z={lin_vel[2]:.2f}")
            t_prev = current_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=30, help="Duration of simulation in seconds")
    args = parser.parse_args()
    
    # Initialize Genesis
    gs.init()
    
    # Get configuration
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    
    # Create environment
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )
    
    # Run simulation in a separate thread for macOS
    gs.tools.run_in_another_thread(fn=run_sim, args=(env, args.duration))
    
    # Set DPI scale for better quality
    env.scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 1
    
    # Start the viewer
    env.scene.viewer.start()


if __name__ == "__main__":
    main()

"""
# Run the forward test
python Genesis-main/examples/locomotion/go2_test_forward.py
""" 