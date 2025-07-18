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
        orientation_reward = -3.0 * np.exp((abs(roll) + abs(pitch)))
        
        # 2. Height reward with exponential decay
        target_height = 0.98
        height_error = abs(torso_height - target_height)
        # Exponential reward that peaks at target height and decays smoothly
        height_reward = 2.0 * np.exp(-20.0 * height_error**2)
        # Additional penalty for being too far below target (falling)
        if torso_height < 0.8:
            height_reward -= 1.0 * (0.8 - torso_height)
        
        # 3. Foot contact reward (encourage both feet on ground)
        left_foot_contact = False
        right_foot_contact = False
        
        # Check for foot contacts with ground
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            
            # Get body names for both geometries in contact
            geom1_body_id = self.model.geom_bodyid[contact.geom1]
            geom2_body_id = self.model.geom_bodyid[contact.geom2]
            geom1_body_name = self.model.body(geom1_body_id).name
            geom2_body_name = self.model.body(geom2_body_id).name
            
            # Check if contact involves floor and foot
            ground_contact = False
            foot_body = None
            
            # Check if one of the bodies is the ground/floor
            if geom1_body_name == "world" or "floor" in self.model.geom(contact.geom1).name:
                ground_contact = True
                foot_body = geom2_body_name
            elif geom2_body_name == "world" or "floor" in self.model.geom(contact.geom2).name:
                ground_contact = True
                foot_body = geom1_body_name
            
            # If it's a ground contact, check which foot
            if ground_contact and foot_body:
                if "left_ankle_link" in foot_body:
                    left_foot_contact = True
                elif "right_ankle_link" in foot_body:
                    right_foot_contact = True
        
        # Reward for foot contact
        if left_foot_contact and right_foot_contact:
            foot_contact_reward = 1.0  # Both feet on ground - good for standing
        elif left_foot_contact or right_foot_contact:
            foot_contact_reward = 0.2  # One foot on ground - partial reward
        else:
            foot_contact_reward = -1.0  # No feet on ground - bad for standing
        
        # 4. Stability reward (low angular velocities)
        angular_velocity_penalty = -0.5 * np.sum(np.square(ang_vel))
    
        # 6. Base bonus for staying alive and not terminating
        alive_bonus = 0.5
        
        
        # Combine all rewards with appropriate weights
        total_reward = (
            orientation_reward +           # 3.0 max - most important
            height_reward +               # 2.0 max - second most important  
            foot_contact_reward +         # 1.0 max - encourage foot contact
            angular_velocity_penalty +    # stability
            alive_bonus                # survival
        )

        if return_info:
            return total_reward, {
                'orientation_reward': orientation_reward,
                'height_reward': height_reward,
                'foot_contact_reward': foot_contact_reward,
                'angular_velocity_penalty': angular_velocity_penalty,
                'alive_bonus': alive_bonus,
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
        #hip_deviation_exceeded = left_hip_deviation > max_hip_deviation or right_hip_deviation > max_hip_deviation
        timeout = self.data.time > 75
        
        return fallen or timeout
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
