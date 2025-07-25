from h1_env import H1StandEnv
import torch
import time
import sys
import mujoco.viewer

sys.path.insert(0, '/home/mehmed-damak/PROJECT09/LearningHumanoidWalking')

print("Testing custom PPO models...")

# Load the custom PPO actor model
try:
    actor = torch.load("h1_custom_ppo_logs/actor.pt", weights_only=False)
    print("✅ Loaded custom PPO actor model")
except Exception as e:
    print(f"❌ Could not load custom PPO model: {e}")
    exit()

# Create environment
env = H1StandEnv()

# Ask user for visualization preference
use_viewer = input("Use MuJoCo viewer? (y/n, default=n): ").lower().strip() == 'y'

if use_viewer:
    print("🎮 Running with MuJoCo viewer...")
    # Test with visualization
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            obs, _ = env.reset()
            total_reward = 0
            episode_count = 0
            step_count = 0
            
            print("✅ MuJoCo viewer launched! Press Ctrl+C to stop.")
            
            for step in range(5000):  # Longer test with viewer
                # Get action from custom policy
                obs_tensor = torch.tensor(obs, dtype=torch.float)
                with torch.no_grad():
                    action = actor(obs_tensor, deterministic=True)
                
                obs, reward, done, _, info = env.step(action.numpy())
                total_reward += reward
                step_count += 1
                
                # Update viewer
                viewer.sync()
                time.sleep(0.01)  # Control simulation speed
                
                # Print progress every 200 steps
                if step % 200 == 0:
                    height = env.data.body('torso_link').xpos[2]
                    print(f"Step {step}: Reward = {total_reward:.2f}, Height = {height:.3f}")
                
                if done:
                    episode_count += 1
                    print(f"Episode {episode_count} ended. Total reward: {total_reward:.2f}, Steps: {step_count}")
                    obs, _ = env.reset()
                    total_reward = 0
                    step_count = 0
                    time.sleep(1.0)  # Pause between episodes
            
            print("Viewer test completed!")
            
    except Exception as e:
        print(f"❌ Viewer failed: {e}")
        print("Falling back to headless mode...")
        use_viewer = False

if not use_viewer:
    print("🤖 Running without visualization...")
    # Test without visualization
    obs, _ = env.reset()
    total_reward = 0
    episode_count = 0

    print("Running test without visualization...")

    for step in range(500):  # Test run
        # Get action from custom policy
        obs_tensor = torch.tensor(obs, dtype=torch.float)
        with torch.no_grad():
            action = actor(obs_tensor, deterministic=True)
        
        obs, reward, done, _, info = env.step(action.numpy())
        total_reward += reward
        
        # Print progress every 100 steps
        if step % 100 == 0:
            print(f"Step {step}: Reward = {total_reward:.2f}, Height = {env.data.body('torso_link').xpos[2]:.3f}")
        
        if done:
            episode_count += 1
            print(f"Episode {episode_count} ended. Total reward: {total_reward:.2f}")
            obs, _ = env.reset()
            total_reward = 0

    print("Custom PPO test completed!")
