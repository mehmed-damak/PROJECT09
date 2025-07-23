"""
Enhanced training script using components from LearningHumanoidWalking
This demonstrates how to incorporate their advanced features into your existing setup
"""
from h1_env import H1StandEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import numpy as np
import torch
import torch.nn as nn
import os
import argparse

# Custom Policy with better initialization (from LearningHumanoidWalking)
class CustomMlpPolicy(nn.Module):
    def __init__(self, observation_space, action_space, lr_schedule, 
                 net_arch=[256, 256], activation_fn=nn.ReLU, **kwargs):
        super().__init__()
        
        # Apply normc initialization like in LearningHumanoidWalking
        def normc_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Build network
        layers = []
        input_dim = observation_space.shape[0]
        
        for hidden_dim in net_arch:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_fn())
            input_dim = hidden_dim
            
        self.shared_net = nn.Sequential(*layers)
        
        # Policy and value heads
        self.policy_net = nn.Linear(input_dim, action_space.shape[0])
        self.value_net = nn.Linear(input_dim, 1)
        
        # Apply normc initialization
        self.apply(normc_fn)
        # Scale down policy output layer
        self.policy_net.weight.data.mul_(0.01)

# Enhanced training with observation normalization
def create_normalized_env():
    """Create environment with normalization like LearningHumanoidWalking"""
    def make_env():
        return H1StandEnv()
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.995)
    return env

def train_with_advanced_features():
    parser = argparse.ArgumentParser(description="Enhanced H1 training with LearningHumanoidWalking features")
    parser.add_argument('--from-checkpoint', type=str, default=None)
    parser.add_argument('--use-normalization', action='store_true', help='Use observation normalization')
    parser.add_argument('--std-dev', type=float, default=0.223, help='Action noise standard deviation')
    args = parser.parse_args()
    
    # Create environment with optional normalization
    if args.use_normalization:
        env = create_normalized_env()
        print("Using normalized environment")
    else:
        env = H1StandEnv()
    
    if args.from_checkpoint is not None:
        print(f"Loading model from {args.from_checkpoint}")
        model = PPO.load(args.from_checkpoint, env=env, tensorboard_log="./h1_tensorboard/")
    else:
        # Enhanced PPO configuration inspired by LearningHumanoidWalking
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs={
                "net_arch": [256, 256],  # Larger network like LearningHumanoidWalking
                "activation_fn": torch.nn.ReLU,
                # Could add custom policy class here if needed
            },
            learning_rate=1e-4,  # Same as LearningHumanoidWalking default
            n_steps=2048,
            batch_size=64,
            gamma=0.99,  # LearningHumanoidWalking default
            gae_lambda=0.95,  # LearningHumanoidWalking default
            clip_range=0.2,
            ent_coef=0.0,  # LearningHumanoidWalking uses 0.0 by default
            max_grad_norm=0.05,  # LearningHumanoidWalking uses 0.05
            verbose=1,
            device="cpu",
            tensorboard_log="./h1_tensorboard/"
        )
    
    # Training loop with enhanced logging
    os.makedirs("./h1_tensorboard/", exist_ok=True)
    
    for i in range(500):
        model.learn(total_timesteps=10000, reset_num_timesteps=False)
        model.save(f"h1_stand_enhanced_{i*10000}")
        
        # Enhanced evaluation
        obs = env.reset() if hasattr(env, 'reset') else env.reset()[0]
        total_reward = 0
        episode_length = 0
        
        for _ in range(5000):
            action, _ = model.predict(obs, deterministic=True)  # Use deterministic for evaluation
            if hasattr(env, 'step'):
                obs, reward, done, info = env.step(action)
            else:
                obs, reward, done, _, info = env.step(action)
            total_reward += reward
            episode_length += 1
            if done:
                break
        
        print(f"Enhanced training - Step {(i+1)*10000}: Reward = {total_reward:.2f}, Length = {episode_length}")
        
        # Save normalization stats if using VecNormalize
        if args.use_normalization and hasattr(env, 'save'):
            env.save(f"h1_stand_enhanced_vecnormalize_{i*10000}.pkl")
    
    model.save("h1_stand_enhanced_final")
    if args.use_normalization and hasattr(env, 'save'):
        env.save("h1_stand_enhanced_vecnormalize_final.pkl")

if __name__ == "__main__":
    train_with_advanced_features()
