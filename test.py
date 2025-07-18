from h1_env import H1StandEnv
from stable_baselines3 import PPO
import mujoco.viewer
import time

# Load best model
model = PPO.load("h1_stand_multiproc_final")  # Try different checkpoints

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
        time.sleep(0.001)#0.01
        #print(env.data.body('torso_link').xpos[2])
        obs=env._get_obs()
        #print("PITCH:", obs[1])
        #print ("ROLL:", obs[0])
        if done:
            print(f"Episode ended. Total reward: {total_reward:.2f}")
            obs, _ = env.reset()
            total_reward = 0