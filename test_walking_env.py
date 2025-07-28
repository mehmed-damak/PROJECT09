"""
Test script for H1 Walking Environment
"""
from h1_env import H1WalkEnv
import numpy as np
import mujoco.viewer
import time

def test_walking_env():
    """Test the walking environment"""
    print("Creating H1 Walking Environment...")
    env = H1WalkEnv()
    
    print("Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test reset
    print("\nTesting reset...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs[:10]}...")  # Show first 10 elements
    
    # Test a few steps
    print("\nTesting steps...")
    for i in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}, done={done}")
    
    print("\nTest completed successfully!")

def test_walking_with_visualization():
    """Test walking with MuJoCo viewer"""
    print("Creating H1 Walking Environment with visualization...")
    env = H1WalkEnv()
    
    # Run with visualizer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        obs, _ = env.reset()
        total_reward = 0
        
        for step in range(2000):  # Run for more steps to see walking
            # Use random actions for now - later we'll use trained policy
            action = env.action_space.sample() * 0.1  # Small random actions
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            
            viewer.sync()
            time.sleep(0.025)  # 40Hz
            
            if step % 100 == 0:
                print(f"Step {step}: reward={reward:.4f}, total_reward={total_reward:.2f}")
                print(f"Phase: {env._phase}, Target idx: {env._current_step_idx}")
            
            if done:
                print(f"Episode ended at step {step}. Total reward: {total_reward:.2f}")
                obs, _ = env.reset()
                total_reward = 0

if __name__ == "__main__":
    print("=== Testing H1 Walking Environment ===")
    
    try:
        # Basic functionality test
        test_walking_env()
        
        # Visualization test (comment out if you don't want the viewer)
        print("\n=== Starting visualization test ===")
        print("Close the viewer window to end the test")
        test_walking_with_visualization()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
