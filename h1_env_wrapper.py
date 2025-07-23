"""
Wrapper to make H1StandEnv compatible with LearningHumanoidWalking PPO
"""
import sys
import os
import numpy as np
import torch

# Add path for Ray workers
sys.path.insert(0, '/home/mehmed-damak/ProjectH1/LearningHumanoidWalking')
os.environ['PYTHONPATH'] = '/home/mehmed-damak/ProjectH1/LearningHumanoidWalking:' + os.environ.get('PYTHONPATH', '')

from h1_env import H1StandEnv

class H1EnvWrapper:
    def __init__(self, path_to_yaml=None):
        self.env = H1StandEnv()
        
        # Required for LearningHumanoidWalking PPO compatibility
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # For mirror symmetry (optional - can implement if needed)
        self.robot = SimpleRobot()
    
    def reset(self):
        obs, info = self.env.reset()
        return obs
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # LearningHumanoidWalking expects 4-tuple, not 5-tuple
        # Also convert numpy types to Python types for torch compatibility
        return obs, float(reward), bool(done), info
    
    def mirror_observation(self, obs):
        """Mirror observation for symmetry learning (optional)"""
        # Implement mirroring logic for your H1 robot if desired
        # For now, return as-is
        return obs
    
    def mirror_clock_observation(self, obs):
        """Mirror observation with clock for symmetry learning"""
        # For now, just return the regular mirror observation
        return self.mirror_observation(obs)
    
    def mirror_action(self, action):
        """Mirror action for symmetry learning (optional)"""
        # Implement action mirroring logic if desired
        return action

class SimpleRobot:
    """Minimal robot class for compatibility"""
    def __init__(self):
        # These would be used for mirror symmetry if implemented
        self.mirrored_obs = []  # indices of observations to mirror
        self.mirrored_acts = []  # indices of actions to mirror  
        self.clock_inds = []    # clock indices for phase information
