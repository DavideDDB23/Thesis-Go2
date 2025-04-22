import os
import sys
import torch
import numpy as np
import time
import genesis as gs
import threading
import math

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tesi.dial_mpc.envs.go2_env import UnitreeGo2Env, UnitreeGo2EnvConfig
from tesi.dial_mpc.utils.utils import get_foot_step


def test_random_actions(env, num_episodes=3, max_steps=100):
    """Test the environment with random actions."""
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        # Reset the environment
        obs, _ = env.reset()
        
        total_reward = 0
        for step in range(max_steps):
            # Sample random actions
            actions = torch.rand(env.num_envs, env.num_actions, device=env.device) * 2 - 1  # [-1, 1]
            
            # Step the environment
            next_obs, rewards, dones, info = env.step(actions)
            
            # Update total reward
            total_reward += rewards.mean().item()
            
            # Print step information
            if step % 10 == 0:
                print(f"Step {step}, Reward: {rewards.mean().item():.4f}, " 
                      f"Done: {dones.sum().item()}/{env.num_envs}")
                
                # Print robot state
                print(f"  Base height: {env.base_pos[0, 2]:.4f}")
                print(f"  Base velocity: {env.base_lin_vel[0, :2]}")
                
                # Check for contacts
                contacts = env.last_contacts[0]
                print(f"  Foot contacts: {contacts}")
            
            # Exit episode if all environments are done
            if dones.all().item():
                print(f"All environments done at step {step}")
                break
        
        print(f"Episode {episode + 1} complete - Total reward: {total_reward:.4f}")


def test_forward_command(env, num_episodes=1, max_steps=300):
    """Test the environment with a constant forward command."""
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes} - Forward Command")
        
        # Reset the environment
        obs, _ = env.reset()
        
        # Set a constant forward command
        env.commands[:, 0] = 0.5  # Forward velocity x
        env.commands[:, 1] = 0.0  # Lateral velocity y
        env.commands[:, 2] = 0.0  # Angular velocity yaw
        
        # Track initial position for distance measurement
        initial_pos = env.base_pos.clone()
        total_reward = 0
        
        # Print leg ordering information for clarity
        print("Leg order information:")
        print("  0-2: Front-Right (FR) - Hip, Thigh, Calf")
        print("  3-5: Front-Left (FL) - Hip, Thigh, Calf")
        print("  6-8: Rear-Right (RR) - Hip, Thigh, Calf")
        print("  9-11: Rear-Left (RL) - Hip, Thigh, Calf")
        print("\nHip joint action signs:")
        print("  Positive: Rotates leg forward relative to the robot's direction")
        print("  Negative: Rotates leg backward relative to the robot's direction")
        
        # Track positions and rewards
        positions = []
        velocities = []
        rewards = []
        
        # Use a more effective gait pattern with stronger hip movements
        for step in range(max_steps):
            # Create a simple trotting gait
            t = step / 20.0  # Time for oscillation
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            
            # Trot gait (diagonal legs move together)
            # Front-right and rear-left are in phase
            # Front-left and rear-right are in opposite phase
            
            # Generate periodic signals (sine waves)
            phase1 = 0.5 * torch.sin(torch.tensor(t * 2 * math.pi, device=env.device))
            phase2 = 0.5 * torch.sin(torch.tensor(t * 2 * math.pi + math.pi, device=env.device))
            
            # IMPROVED: Stronger hip actions for better forward propulsion
            # Front-right leg (FR)
            actions[0, 0] = -0.8 * phase1  # Hip - STRONGER NEGATIVE for forward motion
            actions[0, 1] = 0.2 - 0.3 * phase1  # Thigh
            actions[0, 2] = -1.0 + 0.5 * phase1  # Calf
            
            # Front-left leg (FL)
            actions[0, 3] = -0.8 * phase2  # Hip - STRONGER NEGATIVE for forward motion
            actions[0, 4] = 0.2 - 0.3 * phase2  # Thigh
            actions[0, 5] = -1.0 + 0.5 * phase2  # Calf
            
            # Rear-right leg (RR)
            actions[0, 6] = 0.8 * phase2  # Hip - STRONGER POSITIVE for rear legs
            actions[0, 7] = 0.2 - 0.3 * phase2  # Thigh
            actions[0, 8] = -1.0 + 0.5 * phase2  # Calf
            
            # Rear-left leg (RL)
            actions[0, 9] = 0.8 * phase1  # Hip - STRONGER POSITIVE for rear legs
            actions[0, 10] = 0.2 - 0.3 * phase1  # Thigh
            actions[0, 11] = -1.0 + 0.5 * phase1  # Calf
            
            # Step the environment
            next_obs, reward, dones, info = env.step(actions)
            
            # Track metrics
            positions.append(env.base_pos[0, 0].item())
            velocities.append(env.base_lin_vel[0, 0].item())
            rewards.append(reward.mean().item())
            
            # Update total reward
            total_reward += reward.mean().item()
            
            # Print step information periodically
            if step % 10 == 0 or step == max_steps-1:
                distance_traveled = env.base_pos[0, 0] - initial_pos[0, 0]
                print(f"Step {step}, Reward: {reward.mean().item():.4f}")
                print(f"  Forward distance: {distance_traveled:.4f} m")
                print(f"  Base velocity: {env.base_lin_vel[0, 0]:.4f} m/s")
                print(f"  Foot contacts: {env.last_contacts[0]}")
                
                # Print leg phase information for debugging
                if step % 50 == 0:
                    print(f"  Phases - FR: {phase1.item():.2f}, FL: {phase2.item():.2f}")
                    print(f"  FR hip: {actions[0, 0]:.2f}, FL hip: {actions[0, 3]:.2f}")
                    print(f"  RR hip: {actions[0, 6]:.2f}, RL hip: {actions[0, 9]:.2f}")
                    
                    # Also show foot heights
                    z_feet = env.foot_positions[0, :, 2] - env._foot_radius
                    print(f"  Foot heights: {z_feet}")
        
        # Final results
        distance_traveled = env.base_pos[0, 0] - initial_pos[0, 0]
        print(f"Episode {episode + 1} complete - Total reward: {total_reward:.4f}")
        print(f"Total forward distance: {distance_traveled:.4f} m")
        print(f"Average velocity: {distance_traveled/(max_steps * env.dt):.4f} m/s")
        
        # Show velocity trend
        print("\nVelocity progression:")
        for i in range(0, len(velocities), 50):
            print(f"  Step {i}: {velocities[i]:.4f} m/s")
            
        # Return useful metrics for comparison
        return distance_traveled, total_reward


def test_foot_contacts(env, steps=30):
    """Test specifically for foot contact detection."""
    print("\n=== FOOT CONTACT TEST ===")
    
    # Reset the environment
    obs, _ = env.reset()
    
    # First, print the initial height of each foot
    for i in range(len(env.feet_link_indices)):
        foot_pos = env.foot_positions[0, i]
        print(f"Initial foot {i} height: {foot_pos[2]:.6f}")
    
    print("\nLetting the robot settle on the ground...")
    
    # Let the robot settle on the ground
    for step in range(10):
        # Keep the robot in default pose
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        next_obs, rewards, dones, info = env.step(actions)
        
    print("\nVerifying all feet on ground:")
    print(f"  Foot contacts: {env.last_contacts[0]}")
    
    print("\nLifting front right leg (index 1)...")
    
    # Now specifically create an action to lift one foot
    for step in range(10):
        # Lift front right leg
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        # Apply strong hip flexion and knee extension to lift front right leg
        actions[0, 0] = 0.7  # FR hip - positive rotates leg forward
        actions[0, 1] = -0.5  # FR thigh - negative rotates up
        actions[0, 2] = 0.5  # FR calf - positive extends
        
        next_obs, rewards, dones, info = env.step(actions)
        
        # Every few steps, print foot heights and contact status
        if step == 9:
            print(f"\nAfter lifting FR leg:")
            print(f"  Base height: {env.base_pos[0, 2]:.6f}")
            
            # Print the actual height of each foot and its contact status
            for i in range(len(env.feet_link_indices)):
                foot_height = env.foot_positions[0, i, 2]
                print(f"  Foot {i} height: {foot_height:.6f}, Contact: {env.last_contacts[0, i]}")
            
            print(f"  Foot contact tensor: {env.last_contacts[0]}")
            
            # Get contact forces for verification
            link_contact_forces = torch.tensor(
                env.robot.get_links_net_contact_force(),
                device=env.device,
                dtype=torch.float32,
            )
            print("\nContact forces on feet (z-component):")
            for i, link_idx in enumerate(env.feet_link_indices):
                print(f"  Foot {i} force (z): {link_contact_forces[0, link_idx, 2]:.4f}")
    
    print("\n=== END FOOT CONTACT TEST ===")


def test_foot_lift_and_hold(env, steps=50):
    """Test specifically for foot contact detection with one foot completely lifted and held."""
    print("\n=== FOOT LIFT AND HOLD TEST ===")
    
    # Reset the environment
    obs, _ = env.reset()
    
    # First let the robot settle on the ground
    print("Letting the robot settle...")
    for step in range(20):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        next_obs, rewards, dones, info = env.step(actions)
    
    print("\nInitial foot contacts:")
    print(f"  Foot contacts: {env.last_contacts[0]}")
    
    # Link names for clearer output
    feet_names = ["Front Left", "Front Right", "Rear Left", "Rear Right"]
    
    # Print initial foot positions
    print("\nInitial foot positions:")
    for i in range(len(env.feet_link_indices)):
        foot_pos = env.foot_positions[0, i]
        print(f"  {feet_names[i]} foot height: {foot_pos[2]:.6f}")
    
    # Now strongly lift front right foot (index 1)
    print("\nLifting Front Right foot completely and holding position...")
    for step in range(steps):
        # Create an action that lifts the front right foot while keeping balance
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        
        # Front right leg - strong lift
        actions[0, 0] = 0.9   # FR hip - positive rotates leg forward
        actions[0, 1] = -0.9  # FR thigh - strong upward rotation
        actions[0, 2] = 0.7   # FR calf - extend to ensure no ground contact
        
        # Slightly adjust other legs to maintain balance
        # Front left leg - compensate
        actions[0, 3] = -0.1  # FL hip - slight backward rotation
        actions[0, 4] = 0.1   # FL thigh - slight downward
        actions[0, 5] = -0.1  # FL calf - slight flexion
        
        # Rear legs - compensate
        actions[0, 6] = -0.1  # RR hip
        actions[0, 9] = 0.1   # RL hip
        
        # Step the environment
        next_obs, rewards, dones, info = env.step(actions)
        
        # Print status periodically
        if step % 10 == 0 or step == steps-1:
            print(f"Step {step}:")
            
            # Get contact forces for verification
            link_contact_forces = torch.tensor(
                env.robot.get_links_net_contact_force(),
                device=env.device,
                dtype=torch.float32,
            )
            
            for i in range(len(env.feet_link_indices)):
                foot_height = env.foot_positions[0, i, 2]
                contact_force = link_contact_forces[0, env.feet_link_indices[i], 2]
                print(f"  {feet_names[i]} - Height: {foot_height:.4f}, Force: {contact_force:.2f}, Contact: {env.last_contacts[0, i]}")
            
            # Print base height to ensure stability
            print(f"  Base height: {env.base_pos[0, 2]:.4f}")
    
    print("\n=== END FOOT LIFT AND HOLD TEST ===")


def test_complete_environment(env, steps=350):
    """Run a comprehensive test of the environment with different commands."""
    print("\n=== COMPLETE ENVIRONMENT TEST ===")
    print("\n1. Forward Motion Test")
    
    # Reset the environment
    obs, _ = env.reset()
    
    # Initial position for distance measurement
    initial_pos = env.base_pos.clone()
    
    # Track phases for logging
    phase_log = []
    reward_log = []
    
    # Commands to test in sequence
    command_sequence = [
        (0, 0.5, 0.0, 0.0, "Forward"),          # Forward motion
        (100, 0.0, 0.3, 0.0, "Lateral"),        # Lateral motion
        (175, 0.3, 0.3, 0.3, "Combined"),       # Combined motion
        (250, 0.5, 0.0, 0.0, "Forward again"),  # Return to forward motion
    ]
    
    current_command_idx = 0
    next_command_step = command_sequence[current_command_idx][0]
    
    print(f"Starting with command: {command_sequence[0][4]} - vx={command_sequence[0][1]}, vy={command_sequence[0][2]}, vyaw={command_sequence[0][3]}")
    
    env.commands[0, 0] = command_sequence[0][1]  # vx
    env.commands[0, 1] = command_sequence[0][2]  # vy
    env.commands[0, 2] = command_sequence[0][3]  # vyaw
    
    for step in range(steps):
        # Check if it's time to change the command
        if current_command_idx < len(command_sequence) - 1 and step >= next_command_step:
            current_command_idx += 1
            cmd = command_sequence[current_command_idx]
            next_command_step = cmd[0]
            
            print(f"\nChanging command at step {step} to: {cmd[4]} - vx={cmd[1]}, vy={cmd[2]}, vyaw={cmd[3]}")
            
            env.commands[0, 0] = cmd[1]  # vx
            env.commands[0, 1] = cmd[2]  # vy
            env.commands[0, 2] = cmd[3]  # vyaw
        
        # Create a trotting gait based on the current timestep
        t = step / 20.0  # Time for oscillation
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        
        # Generate periodic signals (sine waves)
        phase1 = 0.5 * torch.sin(torch.tensor(t * 2 * math.pi, device=env.device))
        phase2 = 0.5 * torch.sin(torch.tensor(t * 2 * math.pi + math.pi, device=env.device))
        
        phase_log.append((phase1.item(), phase2.item()))
        
        # Front-right leg (FR)
        actions[0, 0] = -0.6 * phase1
        actions[0, 1] = 0.2 - 0.3 * phase1
        actions[0, 2] = -1.0 + 0.5 * phase1
        
        # Front-left leg (FL)
        actions[0, 3] = -0.6 * phase2
        actions[0, 4] = 0.2 - 0.3 * phase2
        actions[0, 5] = -1.0 + 0.5 * phase2
        
        # Rear-right leg (RR)
        actions[0, 6] = 0.6 * phase2
        actions[0, 7] = 0.2 - 0.3 * phase2
        actions[0, 8] = -1.0 + 0.5 * phase2
        
        # Rear-left leg (RL)
        actions[0, 9] = 0.6 * phase1
        actions[0, 10] = 0.2 - 0.3 * phase1
        actions[0, 11] = -1.0 + 0.5 * phase1
        
        # Step the environment
        next_obs, rewards, dones, info = env.step(actions)
        reward_log.append(rewards.item())
        
        # Print step information periodically
        if step % 50 == 0 or step == steps-1:
            # Calculate distances from starting position
            delta_pos = env.base_pos[0] - initial_pos[0]
            distance_forward = delta_pos[0]
            distance_lateral = delta_pos[1]
            yaw = env.base_euler[0, 2]
            
            print(f"\nStep {step}:")
            print(f"  Command: vx={env.commands[0, 0]:.2f}, vy={env.commands[0, 1]:.2f}, vyaw={env.commands[0, 2]:.2f}")
            print(f"  Forward distance: {distance_forward:.3f}m, Lateral: {distance_lateral:.3f}m")
            print(f"  Current velocity: vx={env.base_lin_vel[0, 0]:.3f}, vy={env.base_lin_vel[0, 1]:.3f}, vyaw={env.base_ang_vel[0, 2]:.3f}")
            print(f"  Yaw orientation: {yaw:.3f} rad")
            print(f"  Base height: {env.base_pos[0, 2]:.3f}m")
            print(f"  Reward: {rewards.item():.4f}")
            
            # Print foot contacts
            print(f"  Foot contacts: {env.last_contacts[0]}")
            
    # Calculate final metrics
    delta_pos = env.base_pos[0] - initial_pos[0]
    total_distance = torch.norm(delta_pos[:2])
    forward_distance = delta_pos[0]
    
    # Calculate forward velocity
    forward_velocity = forward_distance / (steps * env.dt)
    
    print("\n=== TEST RESULTS ===")
    print(f"Total travel distance: {total_distance:.4f}m")
    print(f"Forward distance: {forward_distance:.4f}m")
    print(f"Lateral distance: {delta_pos[1]:.4f}m")
    print(f"Vertical height change: {delta_pos[2]:.4f}m")
    print(f"Average forward velocity: {forward_velocity:.4f}m/s")
    print(f"Final orientation (yaw): {env.base_euler[0, 2]:.4f}rad")
    print(f"Average reward: {sum(reward_log)/len(reward_log):.4f}")
    print("=== END COMPLETE ENVIRONMENT TEST ===")
    
    return forward_distance, forward_velocity


def test_observation_space(env, steps=20):
    """Test the observation space structure and values."""
    print("\n=== OBSERVATION SPACE TEST ===")
    
    # Reset the environment
    obs, _ = env.reset()
    
    print(f"Observation shape: {obs.shape}")
    expected_shape = (env.num_envs, env.num_obs)
    print(f"Expected shape: {expected_shape}")
    
    if obs.shape != expected_shape:
        print(f"⚠️ Warning: Observation shape {obs.shape} doesn't match expected {expected_shape}")
        return False
    
    # Run a few steps to get meaningful observations
    for step in range(steps):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        obs, rewards, dones, info = env.step(actions)
    
    # Extract major components from observation
    # The layout should match what's defined in step() function
    # Examining one environment for simplicity
    obs = obs[0]
    
    # Expected ranges or values for different observation components
    print("\nChecking observation components and ranges:")
    
    # Command velocities
    cmd_vel = obs[:3]
    print(f"Command velocities: {cmd_vel}")
    
    # Angular velocity commands
    ang_vel_cmd = obs[2:3]
    print(f"Angular velocity command: {ang_vel_cmd}")
    
    # Control signals (actions)
    ctrl = obs[3:15]
    print(f"Control signals (sample): {ctrl[:3]}...")
    
    # Body pose components
    all_remaining = obs[15:]
    print(f"Remaining observations (sample): {all_remaining[:5]}...")
    
    # Check if values are reasonable (not NaN or Inf)
    has_nan = torch.isnan(obs).any().item()
    has_inf = torch.isinf(obs).any().item()
    
    if has_nan:
        print("⚠️ Warning: Observation contains NaN values!")
        return False
    
    if has_inf:
        print("⚠️ Warning: Observation contains Inf values!")
        return False
    
    print("✓ Observation space validation passed")
    print("=== END OBSERVATION SPACE TEST ===")
    return True


def test_rewards(env, steps=30):
    """Test reward calculation and components."""
    print("\n=== REWARD CALCULATION TEST ===")
    
    # Reset the environment
    obs, _ = env.reset()
    
    # Track rewards over time
    rewards_over_time = []
    
    # Set a specific command for consistency
    env.commands[0, 0] = 0.5  # Forward velocity
    env.commands[0, 1] = 0.0  # Lateral velocity
    env.commands[0, 2] = 0.0  # Angular velocity
    
    print(f"Testing with command: vx={env.commands[0, 0]}, vy={env.commands[0, 1]}, vyaw={env.commands[0, 2]}")
    
    for step in range(steps):
        # Create a simple action pattern
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        
        # Simple trotting pattern
        t = step / 20.0
        phase1 = 0.5 * torch.sin(torch.tensor(t * 2 * math.pi, device=env.device))
        phase2 = 0.5 * torch.sin(torch.tensor(t * 2 * math.pi + math.pi, device=env.device))
        
        # Apply the pattern to legs
        # Front-right leg (FR)
        actions[0, 0] = -0.6 * phase1
        actions[0, 1] = 0.2 - 0.3 * phase1
        actions[0, 2] = -1.0 + 0.5 * phase1
        
        # Front-left leg (FL)
        actions[0, 3] = -0.6 * phase2
        actions[0, 4] = 0.2 - 0.3 * phase2
        actions[0, 5] = -1.0 + 0.5 * phase2
        
        # Rear-right leg (RR)
        actions[0, 6] = 0.6 * phase2
        actions[0, 7] = 0.2 - 0.3 * phase2
        actions[0, 8] = -1.0 + 0.5 * phase2
        
        # Rear-left leg (RL)
        actions[0, 9] = 0.6 * phase1
        actions[0, 10] = 0.2 - 0.3 * phase1
        actions[0, 11] = -1.0 + 0.5 * phase1
        
        # Step the environment
        next_obs, rewards, dones, info = env.step(actions)
        rewards_over_time.append(rewards.item())
        
        if step % 10 == 0:
            print(f"Step {step}, Reward: {rewards.item():.4f}")
            print(f"  Base velocity: {env.base_lin_vel[0, 0]:.4f} m/s (target: {env.commands[0, 0]:.4f})")
            print(f"  Base height: {env.base_pos[0, 2]:.4f} m")
            
            # Get upright vector to check orientation
            up_vec = torch.tensor([0.0, 0.0, 1.0], device=env.device)
            current_up = gs.utils.geom.transform_by_quat(up_vec.unsqueeze(0), env.base_quat)[0]
            print(f"  Upright vector: {current_up}")
    
    # Calculate reward statistics
    avg_reward = sum(rewards_over_time) / len(rewards_over_time)
    min_reward = min(rewards_over_time)
    max_reward = max(rewards_over_time)
    
    print("\nReward statistics:")
    print(f"  Average reward: {avg_reward:.4f}")
    print(f"  Minimum reward: {min_reward:.4f}")
    print(f"  Maximum reward: {max_reward:.4f}")
    print("=== END REWARD CALCULATION TEST ===")
    
    return avg_reward


def test_command_sampling(env, num_samples=100):
    """Test command sampling functionality."""
    print("\n=== COMMAND SAMPLING TEST ===")
    
    # Generate multiple commands
    sampled_commands = env.sample_command(num_samples)
    
    # Check shape
    expected_shape = (num_samples, 3)
    print(f"Command shape: {sampled_commands.shape}, Expected: {expected_shape}")
    
    # Print statistics for each command dimension
    for i, name in enumerate(["Forward velocity", "Lateral velocity", "Angular velocity"]):
        cmd_values = sampled_commands[:, i]
        
        mean_val = cmd_values.mean().item()
        min_val = cmd_values.min().item()
        max_val = cmd_values.max().item()
        std_val = cmd_values.std().item()
        
        print(f"\n{name} statistics:")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Min: {min_val:.4f}")
        print(f"  Max: {max_val:.4f}")
        print(f"  Std: {std_val:.4f}")
    
    print("\nSample commands:")
    for i in range(min(5, num_samples)):
        cmd = sampled_commands[i]
        print(f"  {i+1}: vx={cmd[0]:.4f}, vy={cmd[1]:.4f}, vyaw={cmd[2]:.4f}")
    
    print("=== END COMMAND SAMPLING TEST ===")


def test_continuous_foot_contacts(env, steps=200):
    """Test continuous foot contacts during locomotion."""
    print("\n=== CONTINUOUS FOOT CONTACT TEST ===")
    
    # Reset the environment
    obs, _ = env.reset()
    
    # Set a forward command
    env.commands[0, 0] = 0.5  # Forward velocity
    env.commands[0, 1] = 0.0  # Lateral velocity
    env.commands[0, 2] = 0.0  # Angular velocity
    
    # Track initial position
    initial_pos = env.base_pos.clone()
    
    # Link names for clearer output
    feet_names = ["Front Left", "Front Right", "Rear Left", "Rear Right"]
    
    # Create continuous trotting motion
    print("\nRunning continuous trotting motion test...")
    for step in range(steps):
        # Generate trotting gait
        t = step / 20.0
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        
        # Generate periodic signals
        phase1 = 0.5 * torch.sin(torch.tensor(t * 2 * math.pi, device=env.device))
        phase2 = 0.5 * torch.sin(torch.tensor(t * 2 * math.pi + math.pi, device=env.device))
        
        # Front-right leg (FR)
        actions[0, 0] = -0.6 * phase1
        actions[0, 1] = 0.2 - 0.3 * phase1
        actions[0, 2] = -1.0 + 0.5 * phase1
        
        # Front-left leg (FL)
        actions[0, 3] = -0.6 * phase2
        actions[0, 4] = 0.2 - 0.3 * phase2
        actions[0, 5] = -1.0 + 0.5 * phase2
        
        # Rear-right leg (RR)
        actions[0, 6] = 0.6 * phase2
        actions[0, 7] = 0.2 - 0.3 * phase2
        actions[0, 8] = -1.0 + 0.5 * phase2
        
        # Rear-left leg (RL)
        actions[0, 9] = 0.6 * phase1
        actions[0, 10] = 0.2 - 0.3 * phase1
        actions[0, 11] = -1.0 + 0.5 * phase1
        
        # Step the environment
        next_obs, rewards, dones, info = env.step(actions)
        
        # Print status periodically
        if step % 40 == 0:
            print(f"\nStep {step}:")
            print(f"  Phase values - phase1: {phase1.item():.2f}, phase2: {phase2.item():.2f}")
            print(f"  Base velocity: {env.base_lin_vel[0, 0]:.3f} m/s")
            
            # Get contact forces for verification
            link_contact_forces = torch.tensor(
                env.robot.get_links_net_contact_force(),
                device=env.device,
                dtype=torch.float32,
            )
            
            print(f"  Contact status:")
            for i in range(len(env.feet_link_indices)):
                foot_height = env.foot_positions[0, i, 2]
                contact_force = link_contact_forces[0, env.feet_link_indices[i], 2]
                in_contact = env.last_contacts[0, i]
                phase_value = phase1 if i in [0, 2] else phase2  # FL/RL use phase1, FR/RR use phase2
                
                # Print contact info with expected phase relationship
                print(f"    {feet_names[i]}: Contact={in_contact}, Height={foot_height:.3f}m, Force={contact_force:.1f}N, Phase={phase_value.item():.2f}")
    
    # Final report
    total_distance = env.base_pos[0, 0] - initial_pos[0, 0]
    print(f"\nTotal distance traveled: {total_distance:.4f}m")
    if total_distance > 0:
        print("✓ Robot is moving FORWARD!")
    else:
        print("⚠️ Robot is moving BACKWARD!")
    
    print("=== END CONTINUOUS FOOT CONTACT TEST ===")
    return total_distance


def test_foot_reward_and_walking(env, steps=300):
    """Test the foot reward functionality, foot contact detection, and forward walking."""
    print("\n=== FOOT REWARD AND WALKING TEST ===")
    
    # Reset the environment
    obs, _ = env.reset()
    
    # Track initial position and rewards
    initial_pos = env.base_pos.clone()
    foot_rewards = []
    contact_history = []
    
    # Set forward command
    env.commands[:, 0] = 0.5  # Forward velocity
    env.commands[:, 1] = 0.0  # Lateral velocity
    env.commands[:, 2] = 0.0  # Angular velocity
    
    print("Starting test with forward command vx=0.5")
    print("This test will track foot contacts, reward components, and forward motion")
    
    # Run test for specified steps
    for step in range(steps):
        # Create a trotting gait
        t = step / 20.0  # Time for oscillation
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        
        # Generate periodic signals for trotting (diagonal legs move together)
        phase1 = 0.5 * torch.sin(torch.tensor(t * 2 * math.pi, device=env.device))
        phase2 = 0.5 * torch.sin(torch.tensor(t * 2 * math.pi + math.pi, device=env.device))
        
        # Front-right leg (FR)
        actions[0, 0] = -0.6 * phase1  # Hip - Note: negative for forward motion
        actions[0, 1] = 0.2 - 0.3 * phase1  # Thigh
        actions[0, 2] = -1.0 + 0.5 * phase1  # Calf
        
        # Front-left leg (FL)
        actions[0, 3] = -0.6 * phase2  # Hip - Note: negative for forward motion
        actions[0, 4] = 0.2 - 0.3 * phase2  # Thigh
        actions[0, 5] = -1.0 + 0.5 * phase2  # Calf
        
        # Rear-right leg (RR)
        actions[0, 6] = 0.6 * phase2  # Hip - Rear legs use opposite sign convention
        actions[0, 7] = 0.2 - 0.3 * phase2  # Thigh
        actions[0, 8] = -1.0 + 0.5 * phase2  # Calf
        
        # Rear-left leg (RL)
        actions[0, 9] = 0.6 * phase1  # Hip - Rear legs use opposite sign convention
        actions[0, 10] = 0.2 - 0.3 * phase1  # Thigh
        actions[0, 11] = -1.0 + 0.5 * phase1  # Calf
        
        # Step the environment
        next_obs, rewards, dones, info = env.step(actions)
        
        # Track foot contacts
        contact_history.append(env.last_contacts[0].clone().cpu())
        
        # Calculate the foot reward component manually to verify it's working
        duty_ratio, cadence, amplitude = env._gait_params[env._gait]
        phases = env._gait_phase[env._gait]
        
        # Get target foot heights based on gait
        z_feet_tar = get_foot_step(duty_ratio, cadence, amplitude, phases, env.episode_length_buf[0] * env.dt)
        
        # Get actual foot heights
        z_feet = env.foot_positions[:, :, 2] - env._foot_radius
        
        # Compute foot rewards (same calculation as in the environment)
        foot_reward = -torch.sum(((z_feet_tar - z_feet[0]) / 0.05) ** 2).item()
        foot_rewards.append(foot_reward)
        
        # Print status periodically
        if step % 50 == 0 or step == steps-1:
            distance_traveled = env.base_pos[0, 0] - initial_pos[0, 0]
            print(f"\nStep {step}:")
            print(f"  Distance traveled: {distance_traveled:.4f}m")
            print(f"  Current velocity: {env.base_lin_vel[0, 0]:.4f}m/s")
            print(f"  Foot contacts: {env.last_contacts[0]}")
            print(f"  Foot reward component: {foot_reward:.4f}")
            
            # Get foot heights for debugging
            print("  Foot heights (actual):")
            for i in range(4):
                print(f"    Foot {i}: {z_feet[0, i].item():.4f}m")
            
            print("  Foot heights (target):")
            for i in range(4):
                print(f"    Foot {i}: {z_feet_tar[i].item():.4f}m")
    
    # Calculate contact statistics
    contacts_tensor = torch.stack(contact_history)
    contact_percentage = contacts_tensor.float().mean(dim=0) * 100  # percentage per foot
    
    # Calculate foot reward statistics
    avg_foot_reward = sum(foot_rewards) / len(foot_rewards)
    min_foot_reward = min(foot_rewards)
    max_foot_reward = max(foot_rewards)
    
    # Calculate forward performance
    total_distance = env.base_pos[0, 0] - initial_pos[0, 0]
    avg_speed = total_distance / (steps * env.dt)
    
    # Print final results
    print("\n=== FOOT REWARD TEST RESULTS ===")
    print("Contact percentage per foot:")
    feet_names = ["Front Left", "Front Right", "Rear Left", "Rear Right"]
    for i, name in enumerate(feet_names):
        print(f"  {name}: {contact_percentage[i]:.1f}%")
    
    print("\nFoot reward statistics:")
    print(f"  Average: {avg_foot_reward:.4f}")
    print(f"  Min: {min_foot_reward:.4f}")
    print(f"  Max: {max_foot_reward:.4f}")
    
    print("\nForward walking performance:")
    print(f"  Total distance: {total_distance:.4f}m")
    print(f"  Average speed: {avg_speed:.4f}m/s")
    
    # Check if test passes basic criteria
    test_passes = True
    
    # 1. Check if robot moved forward significantly
    if total_distance < 1.0:
        print("⚠️ Warning: Robot didn't move forward enough (< 1m)")
        test_passes = False
    
    # 2. Check if foot contacts are reasonable (should be around 45-55% for trotting)
    ideal_contact = 50.0  # for trotting gait
    for i, pct in enumerate(contact_percentage):
        if abs(pct - ideal_contact) > 20:  # allow ±20% deviation
            print(f"⚠️ Warning: Foot {i} contact percentage ({pct:.1f}%) deviates significantly from expected ({ideal_contact}%)")
            test_passes = False
    
    # 3. Check if foot reward is working (should be negative but not too extreme)
    if avg_foot_reward > 0 or avg_foot_reward < -100:
        print(f"⚠️ Warning: Average foot reward ({avg_foot_reward:.4f}) is out of expected range")
        test_passes = False
    
    if test_passes:
        print("\n✅ Foot reward and contact detection test PASSED!")
    else:
        print("\n⚠️ Some parts of the test DID NOT PASS. Check warnings above.")
    
    print("=== END FOOT REWARD AND WALKING TEST ===")
    
    return total_distance, avg_foot_reward


def run_simulation(env):
    """Run the simulation tests in a separate thread."""
    # First test the observation space
    obs_valid = test_observation_space(env)
    
    if not obs_valid:
        print("\n⚠️ Observation space issues detected. Fix before continuing.")
        return
    
    # Test foot reward and walking - NEW TEST
    distance, foot_reward = test_foot_reward_and_walking(env)
    print(f"\nFoot reward test results - Distance: {distance:.4f}m, Avg foot reward: {foot_reward:.4f}")
    
    # Test continuous foot contacts to verify direction of movement
    distance = test_continuous_foot_contacts(env)
    
    # Only continue if the robot is moving forward
    if distance <= 0:
        print("\n⚠️ Movement direction issue detected. Please fix before continuing.")
        return
    
    # Test command sampling
    test_command_sampling(env)
    
    # Test rewards    
    avg_reward = test_rewards(env)
    print(f"\nAverage reward: {avg_reward:.4f}")
        
    # Run the comprehensive validation test
    dist, vel = test_complete_environment(env, steps=350)
    
    # If we want to see more specific tests, we can uncomment these:
    test_foot_contacts(env, steps=30)
    test_forward_command(env, num_episodes=1, max_steps=300)
    # test_foot_lift_and_hold(env, steps=50)
    # test_random_actions(env, num_episodes=1, max_steps=100)
    
    print("\nTest completed successfully!")
    print(f"Forward distance: {dist:.4f}m, Velocity: {vel:.4f}m/s")


def main():
    # Initialize Genesis
    gs.init()
    
    # Create environment configuration
    env_cfg = UnitreeGo2EnvConfig(
        kp=70.0,
        kd=1.0,
        default_vx=0.5,
        default_vy=0.0,
        default_vyaw=0.0,
        ramp_up_time=1.0,
        gait="trot",
        leg_control="position",  # Use position control for testing
        randomize_tasks=False,
        dt=0.02,
        action_scale=0.3,
    )
    
    # Create environment
    env = UnitreeGo2Env(
        num_envs=1,
        env_cfg=env_cfg,
        show_viewer=True,  # Enable visualization
    )
    
    # Set DPI scale for better quality
    env.scene._visualizer._viewer._pyrender_viewer._renderer.dpscale = 1
    
    # Run simulation in a separate thread
    sim_thread = threading.Thread(target=run_simulation, args=(env,))
    sim_thread.daemon = True
    sim_thread.start()
    
    # Start the viewer
    print("Starting viewer - close the viewer window to exit")
    env.scene.viewer.start()


if __name__ == "__main__":
    main() 