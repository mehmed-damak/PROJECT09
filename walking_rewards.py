"""
Walking reward functions adapted from LearningHumanoidWalking
"""
import numpy as np
import sys
import os

# Add LearningHumanoidWalking to path
sys.path.insert(0, '/home/mehmed-damak/PROJECT09/LearningHumanoidWalking')

def create_clock_functions(swing_duration=0.4, stance_duration=0.6, period=80):
    """
    Create simplified clock functions for H1 walking
    """
    def right_force_clock(phase):
        # Right foot should have force during left swing (second half of cycle)
        return 1.0 if phase >= period // 2 else -1.0
    
    def right_vel_clock(phase):
        # Right foot should move during right swing (first half of cycle)
        return 1.0 if phase < period // 2 else -1.0
    
    def left_force_clock(phase):
        # Left foot should have force during right swing (first half of cycle)  
        return 1.0 if phase < period // 2 else -1.0
    
    def left_vel_clock(phase):
        # Left foot should move during left swing (second half of cycle)
        return 1.0 if phase >= period // 2 else -1.0
    
    return ([right_force_clock, right_vel_clock], 
            [left_force_clock, left_vel_clock])

def calc_foot_frc_clock_reward(left_frc, right_frc, left_frc_fn, right_frc_fn, phase, robot_mass=80):
    """Calculate foot force clock reward"""
    desired_max_foot_frc = robot_mass * 9.8 * 0.5
    
    normed_left_frc = min(left_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_right_frc = min(right_frc, desired_max_foot_frc) / desired_max_foot_frc
    normed_left_frc = normed_left_frc * 2 - 1
    normed_right_frc = normed_right_frc * 2 - 1
    
    left_frc_clock = left_frc_fn(phase)
    right_frc_clock = right_frc_fn(phase)
    
    left_frc_score = np.tanh(np.pi/4 * left_frc_clock * normed_left_frc)
    right_frc_score = np.tanh(np.pi/4 * right_frc_clock * normed_right_frc)
    
    return (left_frc_score + right_frc_score) / 2

def calc_foot_vel_clock_reward(left_vel, right_vel, left_vel_fn, right_vel_fn, phase):
    """Calculate foot velocity clock reward"""
    desired_max_foot_vel = 0.2
    
    normed_left_vel = min(np.linalg.norm(left_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_right_vel = min(np.linalg.norm(right_vel), desired_max_foot_vel) / desired_max_foot_vel
    normed_left_vel = normed_left_vel * 2 - 1
    normed_right_vel = normed_right_vel * 2 - 1
    
    left_vel_clock = left_vel_fn(phase)
    right_vel_clock = right_vel_fn(phase)
    
    left_vel_score = np.tanh(np.pi/4 * left_vel_clock * normed_left_vel)
    right_vel_score = np.tanh(np.pi/4 * right_vel_clock * normed_right_vel)
    
    return (left_vel_score + right_vel_score) / 2

def calc_height_reward(current_height, goal_height=0.98, goal_speed=0.5):
    """Calculate height reward"""
    error = np.abs(current_height - goal_height)
    deadzone_size = 0.01 + 0.05 * goal_speed
    if error < deadzone_size:
        error = 0
    return np.exp(-40 * np.square(error))

def calc_orientation_reward(roll, pitch):
    """Calculate orientation reward for staying upright"""
    return np.exp(-10 * (roll**2 + pitch**2))

def calc_step_reward(foot_positions, target_pos, target_reached, root_pos, next_target_pos):
    """Calculate stepping reward"""
    foot_dist_to_target = min([np.linalg.norm(fp - target_pos) for fp in foot_positions])
    
    hit_reward = 0
    if target_reached:
        hit_reward = np.exp(-foot_dist_to_target / 0.25)
    
    target_mp = (target_pos[:2] + next_target_pos[:2]) / 2
    root_dist_to_target = np.linalg.norm(root_pos[:2] - target_mp)
    progress_reward = np.exp(-root_dist_to_target / 2)
    
    return 0.8 * hit_reward + 0.2 * progress_reward

def calc_action_smoothness_reward(action, prev_action):
    """Calculate action smoothness reward"""
    if prev_action is None:
        return 0.0
    penalty = 5 * np.mean(np.abs(prev_action - action))
    return np.exp(-penalty)
