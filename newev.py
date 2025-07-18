import mujoco
import numpy as np
from gymnasium import Env, spaces
 
class H1StandEnv(Env):
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("assets/h1/scene.xml")
        self.data = mujoco.MjData(self.model)
        
        # Get torso body
        self.torso_body = self.model.body("torso_link")
        
        # Get joint IDs (legs only)
        self.joint_ids = {
            "left_hip_yaw": self.model.joint("left_hip_yaw").id,
            "left_hip_roll": self.model.joint("left_hip_roll").id,
            "left_hip_pitch": self.model.joint("left_hip_pitch").id,
            "left_knee": self.model.joint("left_knee").id,
            "left_ankle": self.model.joint("left_ankle").id,
            "right_hip_yaw": self.model.joint("right_hip_yaw").id,
            "right_hip_roll": self.model.joint("right_hip_roll").id,
            "right_hip_pitch": self.model.joint("right_hip_pitch").id,
            "right_knee": self.model.joint("right_knee").id,
            "right_ankle": self.model.joint("right_ankle").id
        }
        
        # Get actuator IDs (legs only)
        self.actuator_ids = {
            "left_hip_yaw": self.model.actuator("left_hip_yaw").id,
            "left_hip_roll": self.model.actuator("left_hip_roll").id,
            "left_hip_pitch": self.model.actuator("left_hip_pitch").id,
            "left_knee": self.model.actuator("left_knee").id,
            "left_ankle": self.model.actuator("left_ankle").id,
            "right_hip_yaw": self.model.actuator("right_hip_yaw").id,
            "right_hip_roll": self.model.actuator("right_hip_roll").id,
            "right_hip_pitch": self.model.actuator("right_hip_pitch").id,
            "right_knee": self.model.actuator("right_knee").id,
            "right_ankle": self.model.actuator("right_ankle").id
        }
        
        # Expanded action space (legs only - 10 actuators)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.actuator_ids),), dtype=np.float32
        )
        
        # Corrected observation space - legs only
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 + 2 + len(self.joint_ids)*2,), dtype=np.float32
        )
        
        # Joint scaling (legs only)
        self.joint_scaling = {
            'left_hip_yaw': 200,
            'left_hip_roll': 200,
            'left_hip_pitch': 200,
            'left_knee': 300,
            'left_ankle': 40,
            'right_hip_yaw': 200,
            'right_hip_roll': 200,
            'right_hip_pitch': 200,
            'right_knee': 300,
            'right_ankle': 40
        }
        
        # Joint-specific PD gains (legs only)
        self.Kp = {
            "left_hip_yaw": 50.0,
            "left_hip_roll": 50.0,
            "left_hip_pitch": 60.0,
            "left_knee": 80.0,
            "left_ankle": 40.0,
            "right_hip_yaw": 50.0,
            "right_hip_roll": 50.0,
            "right_hip_pitch": 60.0,
            "right_knee": 80.0,
            "right_ankle": 40.0
        }
        self.Kd = {
            "left_hip_yaw": 2.0,
            "left_hip_roll": 2.0,
            "left_hip_pitch": 2.5,
            "left_knee": 3.0,
            "left_ankle": 1.5,
            "right_hip_yaw": 2.0,
            "right_hip_roll": 2.0,
            "right_hip_pitch": 2.5,
            "right_knee": 3.0,
            "right_ankle": 1.5
        }
        
        # Reset to standing position
        self.reset()
        
        # Store reference (initial) joint positions for penalty calculation
        self.reference_positions = {}
        for joint_name, joint_id in self.joint_ids.items():
            self.reference_positions[joint_name] = self.data.qpos[joint_id]
        
        self.prev_feet_contact = [True, True]
        self.feet_air_time = [0.0, 0.0]
        self.last_time = 0.0
        
    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0, 0, 1.015]  # x, y, z position - start on the ground
        self.data.qpos[3:7] = [1, 0, 0, 0]  # quaternion orientation (w, x, y, z) - identity quaternion

        self.data.qpos[9] = -0.2   # Left hip pitch
        self.data.qpos[10] = 0.5 # Left knee 0.6 orginally
        self.data.qpos[11] = -0.3  # Left ankle
        
        # Right leg
        self.data.qpos[14] = -0.2  # Right hip pitch
        self.data.qpos[15] = 0.5  # Right knee
        self.data.qpos[16] = -0.3  # Right ankle

        
        # Forward simulation to update physics
        mujoco.mj_forward(self.model, self.data)
        self.prev_feet_contact = [True, True]
        self.feet_air_time = [0.0, 0.0]
        self.last_time = self.data.time
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Torso orientation
        torso_mat = self.data.xmat[self.torso_body.id].reshape(3, 3)
        z_axis = torso_mat[:, 2]
        pitch = np.arcsin(z_axis[0])
        roll = np.arcsin(z_axis[1])
        
        # Angular velocity
        ang_vel = self.data.qvel[3:5]
 
        # Explicit joint positions (legs only)
        left_hip_yaw_pos = self.data.qpos[self.joint_ids["left_hip_yaw"]]
        left_hip_roll_pos = self.data.qpos[self.joint_ids["left_hip_roll"]]
        left_hip_pitch_pos = self.data.qpos[self.joint_ids["left_hip_pitch"]]
        left_knee_pos = self.data.qpos[self.joint_ids["left_knee"]]
        left_ankle_pos = self.data.qpos[self.joint_ids["left_ankle"]]
        right_hip_yaw_pos = self.data.qpos[self.joint_ids["right_hip_yaw"]]
        right_hip_roll_pos = self.data.qpos[self.joint_ids["right_hip_roll"]]
        right_hip_pitch_pos = self.data.qpos[self.joint_ids["right_hip_pitch"]]
        right_knee_pos = self.data.qpos[self.joint_ids["right_knee"]]
        right_ankle_pos = self.data.qpos[self.joint_ids["right_ankle"]]
 
        # Explicit joint velocities (legs only)
        left_hip_yaw_vel = self.data.qvel[self.joint_ids["left_hip_yaw"]]
        left_hip_roll_vel = self.data.qvel[self.joint_ids["left_hip_roll"]]
        left_hip_pitch_vel = self.data.qvel[self.joint_ids["left_hip_pitch"]]
        left_knee_vel = self.data.qvel[self.joint_ids["left_knee"]]
        left_ankle_vel = self.data.qvel[self.joint_ids["left_ankle"]]
        right_hip_yaw_vel = self.data.qvel[self.joint_ids["right_hip_yaw"]]
        right_hip_roll_vel = self.data.qvel[self.joint_ids["right_hip_roll"]]
        right_hip_pitch_vel = self.data.qvel[self.joint_ids["right_hip_pitch"]]
        right_knee_vel = self.data.qvel[self.joint_ids["right_knee"]]
        right_ankle_vel = self.data.qvel[self.joint_ids["right_ankle"]]
 
        return np.array([
            roll, pitch, ang_vel[0], ang_vel[1],
            left_hip_yaw_pos, left_hip_roll_pos, left_hip_pitch_pos, left_knee_pos, left_ankle_pos,
            right_hip_yaw_pos, right_hip_roll_pos, right_hip_pitch_pos, right_knee_pos, right_ankle_pos,
            left_hip_yaw_vel, left_hip_roll_vel, left_hip_pitch_vel, left_knee_vel, left_ankle_vel,
            right_hip_yaw_vel, right_hip_roll_vel, right_hip_pitch_vel, right_knee_vel, right_ankle_vel
        ], dtype=np.float32)



 
    def _get_reward(self, obs, return_info=False):
        """Reward for standing upright and stable with improved learning signal."""
        
        # Extract observations
        roll = obs[0]
        pitch = obs[1]
        ang_vel = obs[2:4]
        
        # Get torso position and orientation
        torso_height = self.data.xpos[self.torso_body.id][2]
        torso_pos = self.data.xpos[self.torso_body.id]
        
        # 1. Upright orientation reward (most important for standing)
        # Use exponential reward that peaks at zero orientation
        orientation_reward = 3.0 * np.exp(-5.0 * (roll**2 + pitch**2))
        
        # 2. Height reward with smooth transitions
        target_height = 0.98
        height_error = abs(torso_height - target_height)
        if height_error < 0.03:  # Very close to target
            height_reward = 2.0 * (1.0 - height_error / 0.03)
        elif height_error < 0.1:  # Reasonably close
            height_reward = 1.0 * (1.0 - (height_error - 0.03) / 0.07)
        else:  # Too far from target
            height_reward = -0.5 * height_error
        
        # 3. Stability reward (low angular velocities)
        angular_velocity_penalty = -0.5 * np.sum(np.square(ang_vel))
        
        # 4. Joint velocity smoothness (encourage smooth movements)
        joint_velocities = np.array([self.data.qvel[joint_id] for joint_id in self.joint_ids.values()])
        joint_velocity_penalty = -0.1 * np.sum(np.square(joint_velocities))
        
        
        # 6. Center of mass stability (penalize lateral movement)
        com_x_penalty = -2.0 * (torso_pos[0]**2)  # Penalize x displacement
        com_y_penalty = -2.0 * (torso_pos[1]**2)  # Penalize y displacement
        
        # 7. Joint position regularization with EXTRA strict focus on hip rolls
        joint_position_penalty = 0
        hip_roll_penalty = 0
        
        for joint_name, joint_id in self.joint_ids.items():
            current_pos = self.data.qpos[joint_id]
            reference_pos = self.reference_positions[joint_name]
            deviation = abs(current_pos - reference_pos)
            
            # EXTREMELY strict handling for hip roll joints (prevent ANY leg lifting)
            if "hip_roll" in joint_name:
                if deviation > 0.05:  # Even stricter - only 0.05 rad allowed
                    hip_roll_penalty -= 10.0 * (deviation - 0.05)**2  # DOUBLE the penalty
                # Additional penalty for any movement away from reference
                hip_roll_penalty -= 2.0 * deviation**2  # Constant penalty for any deviation
            else:
                # Allow some deviation for other joints
                if deviation > 0.2:
                    joint_position_penalty -= 0.3 * (deviation - 0.2)**2
        
        # 8. Enhanced foot height penalty (prevent ANY lifting of feet)
        left_ankle_height = self.data.xpos[self.model.body("left_ankle_link").id][2]
        right_ankle_height = self.data.xpos[self.model.body("right_ankle_link").id][2]
        ground_level = 0.05  # Expected foot height when on ground
        
        foot_height_penalty = 0
        # Much stricter foot height monitoring
        if left_ankle_height > ground_level + 0.02:  # Only 2cm tolerance instead of 5cm
            foot_height_penalty -= 5.0 * (left_ankle_height - ground_level - 0.02)**2  # Stronger penalty
        if right_ankle_height > ground_level + 0.02:  # Extra strict on right foot
            foot_height_penalty -= 5.0 * (right_ankle_height - ground_level - 0.02)**2

        
        # 9. Enhanced foot contact reward (require BOTH feet on ground for standing)
        left_foot_contact = False
        right_foot_contact = False
        
        # Check for foot contacts
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1_name = self.model.geom(contact.geom1).name
            geom2_name = self.model.geom(contact.geom2).name
            
            # Check if contact involves feet (adjust names as needed for your model)
            if "left_ankle" in geom1_name or "left_ankle" in geom2_name or \
               "left_foot" in geom1_name or "left_foot" in geom2_name:
                left_foot_contact = True
            if "right_ankle" in geom1_name or "right_ankle" in geom2_name or \
               "right_foot" in geom1_name or "right_foot" in geom2_name:
                right_foot_contact = True
        
        # Require both feet on ground for standing - EXTREMELY strong penalty if either foot is lifted
        if left_foot_contact and right_foot_contact:
            foot_contact_reward = 3.0  # Even stronger reward for both feet down
        elif left_foot_contact or right_foot_contact:
            foot_contact_reward = -2.0  # Stronger penalty for lifting one foot
        else:
            foot_contact_reward = -5.0  # VERY strong penalty for lifting both feet
            
        
        # 10. Enhanced leg symmetry reward (encourage symmetric stance)
        left_hip_roll = self.data.qpos[self.joint_ids["left_hip_roll"]]
        right_hip_roll = self.data.qpos[self.joint_ids["right_hip_roll"]]
        left_hip_pitch = self.data.qpos[self.joint_ids["left_hip_pitch"]]
        right_hip_pitch = self.data.qpos[self.joint_ids["right_hip_pitch"]]
        left_hip_yaw = self.data.qpos[self.joint_ids["left_hip_yaw"]]
        right_hip_yaw = self.data.qpos[self.joint_ids["right_hip_yaw"]]
        left_knee = self.data.qpos[self.joint_ids["left_knee"]]
        right_knee = self.data.qpos[self.joint_ids["right_knee"]]
        left_ankle = self.data.qpos[self.joint_ids["left_ankle"]]
        right_ankle = self.data.qpos[self.joint_ids["right_ankle"]]
        
        # Hip roll symmetry: should be mirrored (left positive, right negative or vice versa)
        # But for standing, they should both be close to zero
        ideal_hip_roll_diff = 0.0  # Both hips should be at reference position
        hip_roll_symmetry_error = abs((left_hip_roll + right_hip_roll) / 2.0)  # Average should be zero
        hip_roll_symmetry = 2.0 * np.exp(-10.0 * hip_roll_symmetry_error**2)  # Strong reward for symmetric
        
        # Hip yaw symmetry: for standing, both should be close to zero (feet pointing forward)
        hip_yaw_symmetry_error = abs((left_hip_yaw + right_hip_yaw) / 2.0)  # Average should be zero
        hip_yaw_symmetry = 1.5 * np.exp(-12.0 * hip_yaw_symmetry_error**2)  # Reward for feet pointing forward
        
        # Hip yaw alignment: also reward when both hips have similar yaw values (parallel feet)
        hip_yaw_diff = abs(left_hip_yaw - right_hip_yaw)
        hip_yaw_alignment = 1.0 * np.exp(-8.0 * hip_yaw_diff**2)  # Reward for parallel feet
        
        # Hip pitch symmetry: should be nearly identical for standing
        hip_pitch_diff = abs(left_hip_pitch - right_hip_pitch)
        hip_pitch_symmetry = 1.5 * np.exp(-15.0 * hip_pitch_diff**2)  # Reward similar angles
        
        # Knee symmetry: should be nearly identical
        knee_diff = abs(left_knee - right_knee)
        knee_symmetry = 1.5 * np.exp(-15.0 * knee_diff**2)  # Reward similar angles
        
        # Ankle symmetry: should be nearly identical
        ankle_diff = abs(left_ankle - right_ankle)
        ankle_symmetry = 1.0 * np.exp(-15.0 * ankle_diff**2)  # Reward similar angles
        
        # Leg posture reward: reward for keeping legs in good standing position
        # Penalize extreme deviations from reference pose
        left_leg_deviation = (
            abs(left_hip_roll - self.reference_positions["left_hip_roll"]) +
            abs(left_hip_pitch - self.reference_positions["left_hip_pitch"]) +
            abs(left_knee - self.reference_positions["left_knee"])
        )
        right_leg_deviation = (
            abs(right_hip_roll - self.reference_positions["right_hip_roll"]) +
            abs(right_hip_pitch - self.reference_positions["right_hip_pitch"]) +
            abs(right_knee - self.reference_positions["right_knee"])
        )
        
        leg_posture_reward = 1.0 * np.exp(-2.0 * (left_leg_deviation + right_leg_deviation))
        
        # Combined symmetry reward
        leg_symmetry_reward = (
            hip_roll_symmetry + 
            hip_yaw_symmetry +
            hip_yaw_alignment +
            hip_pitch_symmetry + 
            knee_symmetry + 
            ankle_symmetry +
            leg_posture_reward
        )
        
        # 11. Specific right leg stability reward (extra encouragement for right leg to stay down)
        right_leg_stability = 0
        right_hip_roll_current = self.data.qpos[self.joint_ids["right_hip_roll"]]
        right_hip_roll_ref = self.reference_positions["right_hip_roll"]
        right_hip_roll_dev = abs(right_hip_roll_current - right_hip_roll_ref)
        
        # Give extra reward for keeping right hip roll very close to reference
        if right_hip_roll_dev < 0.02:  # Very tight tolerance
            right_leg_stability = 2.0 * (1.0 - right_hip_roll_dev / 0.02)
        else:
            right_leg_stability = -3.0 * right_hip_roll_dev  # Strong penalty for deviation
        
        # 12. Base bonus for staying alive and not terminating
        alive_bonus = 0.5
        
        # 13. Progressive height bonus (reward improvement over time)
        min_viable_height = 0.7
        height_progress_bonus = max(0, (torso_height - min_viable_height) / (target_height - min_viable_height))
        
        # Combine all rewards with appropriate weights
        total_reward = (
            orientation_reward +           # 3.0 max - most important
            height_reward +               # 2.0 max - second most important  
            angular_velocity_penalty +    # stability
            joint_velocity_penalty +     # smoothness
            com_x_penalty +              # lateral stability
            com_y_penalty +              # lateral stability
            joint_position_penalty +     # general pose regularization
            hip_roll_penalty +           # SUPER STRONG penalty for hip roll deviation
            foot_height_penalty +        # penalty for lifting feet
            foot_contact_reward +        # strong requirement for both feet down
            leg_symmetry_reward +        # encourage symmetric stance
            right_leg_stability +        # EXTRA right leg stability reward
            alive_bonus +                # survival
            height_progress_bonus        # progressive improvement
        )

        if return_info:
            return total_reward, {
                'orientation_reward': orientation_reward,
                'height_reward': height_reward,
                'angular_velocity_penalty': angular_velocity_penalty,
                'joint_velocity_penalty': joint_velocity_penalty,
                'com_x_penalty': com_x_penalty,
                'com_y_penalty': com_y_penalty,
                'joint_position_penalty': joint_position_penalty,
                'hip_roll_penalty': hip_roll_penalty,
                'foot_height_penalty': foot_height_penalty,
                'foot_contact_reward': foot_contact_reward,
                'leg_symmetry_reward': leg_symmetry_reward,
                'right_leg_stability': right_leg_stability,
                'hip_roll_symmetry': hip_roll_symmetry,
                'hip_yaw_symmetry': hip_yaw_symmetry,
                'hip_yaw_alignment': hip_yaw_alignment,
                'hip_pitch_symmetry': hip_pitch_symmetry,
                'knee_symmetry': knee_symmetry,
                'ankle_symmetry': ankle_symmetry,
                'leg_posture_reward': leg_posture_reward,
                'alive_bonus': alive_bonus,
                'height_progress_bonus': height_progress_bonus,
                'total_reward': total_reward
            }
        return total_reward
 
    def _get_terminated(self, obs):
        """Check termination conditions"""
        roll, pitch = obs[0], obs[1]
        torso_height = self.data.xpos[self.torso_body.id][2]
        
        # Check hip roll deviation from initial positions
        current_left_hip_roll = self.data.qpos[self.joint_ids["left_hip_roll"]]
        current_right_hip_roll = self.data.qpos[self.joint_ids["right_hip_roll"]]
        initial_left_hip_roll = self.reference_positions["left_hip_roll"]
        initial_right_hip_roll = self.reference_positions["right_hip_roll"]
        
        left_hip_deviation = abs(current_left_hip_roll - initial_left_hip_roll)
        right_hip_deviation = abs(current_right_hip_roll - initial_right_hip_roll)
        max_hip_deviation = 0.5  # Maximum allowed deviation in radians
        
        fallen = abs(roll) > 0.5 or abs(pitch) > 0.5 or torso_height < 0.5
        hip_deviation_exceeded = left_hip_deviation > max_hip_deviation or right_hip_deviation > max_hip_deviation
        timeout = self.data.time > 50
        
        return fallen or hip_deviation_exceeded or timeout
    def step(self, action):
        # PD control: interpret action as desired position in [-1, 1] scaled to joint range
        # Explicit, line-by-line for all 10 leg actuators/joints
        # Compute desired positions
        q_des_left_hip_yaw = 0.5 * (action[0] + 1) * (self.model.jnt_range[self.joint_ids["left_hip_yaw"]][1] - self.model.jnt_range[self.joint_ids["left_hip_yaw"]][0]) + self.model.jnt_range[self.joint_ids["left_hip_yaw"]][0]
        q_des_left_hip_roll = 0.5 * (action[1] + 1) * (self.model.jnt_range[self.joint_ids["left_hip_roll"]][1] - self.model.jnt_range[self.joint_ids["left_hip_roll"]][0]) + self.model.jnt_range[self.joint_ids["left_hip_roll"]][0]
        q_des_left_hip_pitch = 0.5 * (action[2] + 1) * (self.model.jnt_range[self.joint_ids["left_hip_pitch"]][1] - self.model.jnt_range[self.joint_ids["left_hip_pitch"]][0]) + self.model.jnt_range[self.joint_ids["left_hip_pitch"]][0]
        q_des_left_knee = 0.5 * (action[3] + 1) * (self.model.jnt_range[self.joint_ids["left_knee"]][1] - self.model.jnt_range[self.joint_ids["left_knee"]][0]) + self.model.jnt_range[self.joint_ids["left_knee"]][0]
        q_des_left_ankle = 0.5 * (action[4] + 1) * (self.model.jnt_range[self.joint_ids["left_ankle"]][1] - self.model.jnt_range[self.joint_ids["left_ankle"]][0]) + self.model.jnt_range[self.joint_ids["left_ankle"]][0]
        q_des_right_hip_yaw = 0.5 * (action[5] + 1) * (self.model.jnt_range[self.joint_ids["right_hip_yaw"]][1] - self.model.jnt_range[self.joint_ids["right_hip_yaw"]][0]) + self.model.jnt_range[self.joint_ids["right_hip_yaw"]][0]
        q_des_right_hip_roll = 0.5 * (action[6] + 1) * (self.model.jnt_range[self.joint_ids["right_hip_roll"]][1] - self.model.jnt_range[self.joint_ids["right_hip_roll"]][0]) + self.model.jnt_range[self.joint_ids["right_hip_roll"]][0]
        q_des_right_hip_pitch = 0.5 * (action[7] + 1) * (self.model.jnt_range[self.joint_ids["right_hip_pitch"]][1] - self.model.jnt_range[self.joint_ids["right_hip_pitch"]][0]) + self.model.jnt_range[self.joint_ids["right_hip_pitch"]][0]
        q_des_right_knee = 0.5 * (action[8] + 1) * (self.model.jnt_range[self.joint_ids["right_knee"]][1] - self.model.jnt_range[self.joint_ids["right_knee"]][0]) + self.model.jnt_range[self.joint_ids["right_knee"]][0]
        q_des_right_ankle = 0.5 * (action[9] + 1) * (self.model.jnt_range[self.joint_ids["right_ankle"]][1] - self.model.jnt_range[self.joint_ids["right_ankle"]][0]) + self.model.jnt_range[self.joint_ids["right_ankle"]][0]
 
        # PD control for each joint (legs only)
        self.data.ctrl[self.actuator_ids["left_hip_yaw"]] = self.Kp["left_hip_yaw"] * (q_des_left_hip_yaw - self.data.qpos[self.joint_ids["left_hip_yaw"]]) - self.Kd["left_hip_yaw"] * self.data.qvel[self.joint_ids["left_hip_yaw"]]
        self.data.ctrl[self.actuator_ids["left_hip_roll"]] = self.Kp["left_hip_roll"] * (q_des_left_hip_roll - self.data.qpos[self.joint_ids["left_hip_roll"]]) - self.Kd["left_hip_roll"] * self.data.qvel[self.joint_ids["left_hip_roll"]]
        self.data.ctrl[self.actuator_ids["left_hip_pitch"]] = self.Kp["left_hip_pitch"] * (q_des_left_hip_pitch - self.data.qpos[self.joint_ids["left_hip_pitch"]]) - self.Kd["left_hip_pitch"] * self.data.qvel[self.joint_ids["left_hip_pitch"]]
        self.data.ctrl[self.actuator_ids["left_knee"]] = self.Kp["left_knee"] * (q_des_left_knee - self.data.qpos[self.joint_ids["left_knee"]]) - self.Kd["left_knee"] * self.data.qvel[self.joint_ids["left_knee"]]
        self.data.ctrl[self.actuator_ids["left_ankle"]] = self.Kp["left_ankle"] * (q_des_left_ankle - self.data.qpos[self.joint_ids["left_ankle"]]) - self.Kd["left_ankle"] * self.data.qvel[self.joint_ids["left_ankle"]]
        self.data.ctrl[self.actuator_ids["right_hip_yaw"]] = self.Kp["right_hip_yaw"] * (q_des_right_hip_yaw - self.data.qpos[self.joint_ids["right_hip_yaw"]]) - self.Kd["right_hip_yaw"] * self.data.qvel[self.joint_ids["right_hip_yaw"]]
        self.data.ctrl[self.actuator_ids["right_hip_roll"]] = self.Kp["right_hip_roll"] * (q_des_right_hip_roll - self.data.qpos[self.joint_ids["right_hip_roll"]]) - self.Kd["right_hip_roll"] * self.data.qvel[self.joint_ids["right_hip_roll"]]
        self.data.ctrl[self.actuator_ids["right_hip_pitch"]] = self.Kp["right_hip_pitch"] * (q_des_right_hip_pitch - self.data.qpos[self.joint_ids["right_hip_pitch"]]) - self.Kd["right_hip_pitch"] * self.data.qvel[self.joint_ids["right_hip_pitch"]]
        self.data.ctrl[self.actuator_ids["right_knee"]] = self.Kp["right_knee"] * (q_des_right_knee - self.data.qpos[self.joint_ids["right_knee"]]) - self.Kd["right_knee"] * self.data.qvel[self.joint_ids["right_knee"]]
        self.data.ctrl[self.actuator_ids["right_ankle"]] = self.Kp["right_ankle"] * (q_des_right_ankle - self.data.qpos[self.joint_ids["right_ankle"]]) - self.Kd["right_ankle"] * self.data.qvel[self.joint_ids["right_ankle"]]
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward, reward_info = self._get_reward(obs, return_info=True)
        terminated = self._get_terminated(obs)
        info = reward_info
        return obs, reward, terminated, False, info
