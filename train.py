from h1_env import H1StandEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import mujoco.viewer
import os

# Create environment
env = H1StandEnv()

# Create PPO model with updated settings
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs={
        "net_arch": [128, 128]  # Larger network for more complex state
    },
    learning_rate=0.8e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    device="cpu",
    tensorboard_log="./h1_tensorboard/"
)

# TensorBoard logging for reward components
os.makedirs("./h1_tensorboard/", exist_ok=True)

# Train in increments with evaluation
for i in range(500):
    model.learn(total_timesteps=10000, reset_num_timesteps=False)
    model.save(f"h1_stand_{i*10000}")
    
    # Test current policy
    obs, _ = env.reset()
    total_reward = 0
    episode_length = 0
    reward_info_sum = None
    
    for _ in range(500):
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        episode_length += 1
        if reward_info_sum is None:
            reward_info_sum = {k: 0.0 for k in info}
        for k in info:
            reward_info_sum[k] += info[k]
        if done:
            break
    
    # Print average reward components for this test episode
    if reward_info_sum is not None:
        avg_info = {k: v/episode_length for k, v in reward_info_sum.items()}
        print(f"After {(i+1)*10000} steps: Test reward = {total_reward:.2f}, Steps = {episode_length}")
        print("  Avg reward components:")
        for k, v in avg_info.items():
            print(f"    {k}: {v:.4f}")
    else:
        print(f"After {(i+1)*10000} steps: Test reward = {total_reward:.2f}, Steps = {episode_length}")

# Save final model
model.save("h1_stand_final")