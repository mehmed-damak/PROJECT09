from h1_env import H1StandEnv
from stable_baselines3 import PPO
import mujoco.viewer
import time

# Load best model
model = PPO.load("h1_stand_0")  # Try different checkpoints

# Create environment
env = H1StandEnv()

# Run with visualizer
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    obs, _ = env.reset()
    total_reward = 0
    
    for _ in range(10000):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        
        viewer.sync()
        time.sleep(0.01)
        
        if done:
            print(f"Episode ended. Total reward: {total_reward:.2f}")
            obs, _ = env.reset()
            total_reward = 0