"""
Quick training starter for H1 Walking
"""
from h1_env import H1WalkEnv
import numpy as np

def test_quick_training():
    """Quick test to verify environment is ready for training"""
    print("Testing H1 Walking Environment for training readiness...")
    
    env = H1WalkEnv()
    print(f"✓ Environment created")
    print(f"✓ Observation space: {env.observation_space.shape}")
    print(f"✓ Action space: {env.action_space.shape}")
    
    # Test multiple episodes
    total_rewards = []
    for episode in range(3):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(100):  # Short episodes for testing
            action = env.action_space.sample() * 0.1  # Small random actions
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: reward = {episode_reward:.2f}")
    
    avg_reward = np.mean(total_rewards)
    print(f"✓ Average reward over 3 episodes: {avg_reward:.2f}")
    print(f"✓ Environment ready for training!")
    
    return avg_reward

if __name__ == "__main__":
    test_quick_training()
