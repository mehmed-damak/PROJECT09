#!/usr/bin/env python3
"""
Single comprehensive training script for H1 Standing Robot
Uses multiple parallel environments to train a single PPO model with GPU acceleration.
Compatible with the existing test.py script without modifications.
"""

import os
import time
import numpy as np
import torch
import multiprocessing as mp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from h1_env import H1StandEnv


class TrainingProgressCallback(BaseCallback):
    """Custom callback to track training progress and performance metrics."""
    
    def __init__(self, check_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.start_time = None
        self.best_mean_reward = -np.inf
        
    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print(f"\n🚀 Training started with {self.training_env.num_envs} environments")
        print(f"🔧 Using device: {self.model.device}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print("=" * 60)
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            elapsed_time = time.time() - self.start_time
            fps = self.num_timesteps / elapsed_time
            
            # Get recent episode statistics
            if len(self.model.ep_info_buffer) > 0:
                ep_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
                ep_lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer]
                mean_reward = np.mean(ep_rewards)
                
                # Check if this is a new best
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    improvement = "🔥 NEW BEST!"
                else:
                    improvement = ""
                
                print(f"📊 Step: {self.num_timesteps:,} | FPS: {fps:.0f} | "
                      f"Reward: {mean_reward:.2f}±{np.std(ep_rewards):.2f} | "
                      f"Length: {np.mean(ep_lengths):.0f} | Episodes: {len(ep_rewards)} {improvement}")
                
        return True


def make_env(rank: int, seed: int = 0):
    """Utility function for multiprocessing environment creation."""
    def _init():
        env = H1StandEnv()
        env = Monitor(env)  # Monitor wrapper for episode statistics
        env.reset(seed=seed + rank)
        return env
    
    set_random_seed(seed)
    return _init


def create_training_environment(num_envs: int = 16, seed: int = 42):
    """Create vectorized environment for training."""
    print(f"🏭 Creating {num_envs} parallel environments...")
    
    # Create environment functions for each process
    env_fns = [make_env(i, seed) for i in range(num_envs)]
    
    # Create vectorized environment with multiprocessing
    vec_env = SubprocVecEnv(env_fns, start_method='spawn')
    
    # Normalize observations and rewards for better training stability
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )
    
    print("✅ Vectorized environment created successfully!")
    return vec_env


def create_model(vec_env, learning_rate: float = 3e-4):
    """Create PPO model optimized for the H1 standing task."""
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🤖 Setting up PPO model on device: {device}")
    
    # Create model with optimized hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=2048,  # Steps per environment per update
        batch_size=64,  # Minibatch size
        n_epochs=10,  # Number of epochs per update
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE lambda
        clip_range=0.2,  # PPO clipping parameter
        clip_range_vf=None,  # Value function clipping
        ent_coef=0.0,  # Entropy coefficient
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        use_sde=False,  # State dependent exploration
        policy_kwargs=dict(
            net_arch=[256, 256, 256],  # Neural network architecture
            activation_fn=torch.nn.ReLU,
        ),
        device=device,
        verbose=0,  # Reduce verbosity since we have custom callback
        tensorboard_log="./tensorboard_logs/",
    )
    
    print(f"✅ Model created successfully!")
    print(f"📐 Input features: {model.policy.mlp_extractor.policy_net[0].in_features}")
    print(f"🎯 Action space: {vec_env.action_space.shape}")
    print(f"👁️ Observation space: {vec_env.observation_space.shape}")
    
    return model


def setup_callbacks():
    """Setup training callbacks for monitoring and saving."""
    os.makedirs("./models/", exist_ok=True)
    
    callbacks = []
    
    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # Save every 50k timesteps
        save_path="./models/",
        name_prefix="h1_stand_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Progress monitoring callback
    progress_callback = TrainingProgressCallback(check_freq=5000)
    callbacks.append(progress_callback)
    
    return callbacks


def test_trained_model(model_path: str, episodes: int = 3):
    """Quick test of the trained model."""
    print(f"\n🧪 Testing trained model: {model_path}")
    
    try:
        # Load model (just the .zip file, no normalization for testing)
        model = PPO.load(model_path)
        
        # Create single environment (same as your test.py)
        env = H1StandEnv()
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(5000):  # Max 5000 steps per test episode
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        print(f"📈 Test Results:")
        print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        
        success_rate = sum(1 for length in episode_lengths if length > 1000) / len(episode_lengths)
        print(f"  Success rate (>1000 steps): {success_rate:.1%}")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")


def main():
    """Main training function."""
    print("=" * 60)
    print("🤖 H1 Standing Robot - Multiprocessing Training")
    print("=" * 60)
    
    # ==================== CONFIGURATION ====================
    NUM_ENVS = 16              # Number of parallel environments
    TOTAL_TIMESTEPS = 5_000_000  # Total training timesteps
    LEARNING_RATE = 1e-4       # Learning rate
    SEED = 42                  # Random seed
    MODEL_NAME = "h1_stand_multiproc_final"  # Final model name (matches your test.py)
    # =======================================================
    
    # Print system information
    print(f"💻 CPU cores available: {mp.cpu_count()}")
    print(f"🔥 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"\n⚙️ Training Configuration:")
    print(f"  🏭 Environments: {NUM_ENVS}")
    print(f"  📊 Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  🎯 Learning rate: {LEARNING_RATE}")
    print(f"  🎲 Random seed: {SEED}")
    print(f"  💾 Model name: {MODEL_NAME}")
    print("=" * 60)
    
    # Set random seeds
    set_random_seed(SEED)
    
    try:
        # Create training environment
        print("🏗️ Setting up training environment...")
        train_env = create_training_environment(num_envs=NUM_ENVS, seed=SEED)
        
        # Create model
        print("\n🤖 Creating model...")
        model = create_model(train_env, learning_rate=LEARNING_RATE)
        
        # Setup callbacks
        print("\n📋 Setting up callbacks...")
        callbacks = setup_callbacks()
        
        # Start training
        print("\n🚀 Starting training...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            tb_log_name="h1_stand_multiproc",
            reset_num_timesteps=False,
            progress_bar=True
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n🎉 Training completed!")
        print(f"⏱️ Total training time: {training_time / 3600:.2f} hours")
        print(f"🏃 Average FPS: {TOTAL_TIMESTEPS / training_time:.1f}")
        
        # Save final model (compatible with your test.py)
        model_path = f"./models/{MODEL_NAME}"
        model.save(model_path)
        
        # Also save the VecNormalize stats separately (not needed for your test.py)
        train_env.save(f"{model_path}_vecnormalize.pkl")
        
        print(f"💾 Final model saved: {model_path}.zip")
        print(f"📊 VecNormalize stats saved: {model_path}_vecnormalize.pkl")
        
        # Test the model
        test_trained_model(model_path, episodes=3)
        
        print(f"\n✅ Training complete! Your test.py should work with: {MODEL_NAME}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user!")
        # Save current model
        interrupt_path = f"./models/{MODEL_NAME}_interrupted"
        model.save(interrupt_path)
        train_env.save(f"{interrupt_path}_vecnormalize.pkl")
        print(f"💾 Model saved before exit: {interrupt_path}.zip")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise
        
    finally:
        # Clean up
        try:
            train_env.close()
        except:
            pass


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
