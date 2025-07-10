import mujoco
import numpy as np
from gymnasium import Env, spaces
 
class H1StandEnv(Env):
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("assets/h1/scene.xml")
        self.data = mujoco.MjData(self.model)
        
        # Get torso body
        self.torso_body = self.model.body("torso_link")
        
        # Get joint IDs
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
            "right_ankle": self.model.joint("right_ankle").id,
            "torso": self.model.joint("torso").id,
            "left_shoulder_pitch": self.model.joint("left_shoulder_pitch").id,
            "left_shoulder_roll": self.model.joint("left_shoulder_roll").id,
            "left_shoulder_yaw": self.model.joint("left_shoulder_yaw").id,
            "left_elbow": self.model.joint("left_elbow").id,
            "right_shoulder_pitch": self.model.joint("right_shoulder_pitch").id,
            "right_shoulder_roll": self.model.joint("right_shoulder_roll").id,
            "right_shoulder_yaw": self.model.joint("right_shoulder_yaw").id,
            "right_elbow": self.model.joint("right_elbow").id
        }
        
        # Get actuator IDs
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
            "right_ankle": self.model.actuator("right_ankle").id,
            "torso": self.model.actuator("torso").id,
            "left_shoulder_pitch": self.model.actuator("left_shoulder_pitch").id,
            "left_shoulder_roll": self.model.actuator("left_shoulder_roll").id,
            "left_shoulder_yaw": self.model.actuator("left_shoulder_yaw").id,
            "left_elbow": self.model.actuator("left_elbow").id,
            "right_shoulder_pitch": self.model.actuator("right_shoulder_pitch").id,
            "right_shoulder_roll": self.model.actuator("right_shoulder_roll").id,
            "right_shoulder_yaw": self.model.actuator("right_shoulder_yaw").id,
            "right_elbow": self.model.actuator("right_elbow").id
        }
        
        # Expanded action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.actuator_ids),), dtype=np.float32
        )
        
        # Corrected observation space - 18 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 + 2 + len(self.joint_ids)*2,), dtype=np.float32
        )
        
        # Joint scaling
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
            'right_ankle': 40,
            'torso': 200,
            'left_shoulder_pitch': 40,
            'left_shoulder_roll': 40,
            'left_shoulder_yaw': 18,
            'left_elbow': 18,
            'right_shoulder_pitch': 40,
            'right_shoulder_roll': 40,
            'right_shoulder_yaw': 18,
            'right_elbow': 18
        }
        
        # Joint-specific PD gains - updated values
        self.Kp = {
            "left_hip_yaw": 100.0,
            "left_hip_roll": 100.0,
            "left_hip_pitch": 100.0,
            "left_knee": 100.0,
            "left_ankle": 20.0,
            "right_hip_yaw": 100.0,
            "right_hip_roll": 100.0,
            "right_hip_pitch": 100.0,
            "right_knee": 100.0,
            "right_ankle": 20.0,
            "torso": 40.0,
            "left_shoulder_pitch": 20.0,
            "left_shoulder_roll": 20.0,
            "left_shoulder_yaw": 20.0,
            "left_elbow": 20.0,
            "right_shoulder_pitch": 20.0,
            "right_shoulder_roll": 20.0,
            "right_shoulder_yaw": 20.0,
            "right_elbow": 20.0
        }
        self.Kd = {
            "left_hip_yaw": 10.0,
            "left_hip_roll": 10.0,
            "left_hip_pitch": 10.0,
            "left_knee": 10.0,
            "left_ankle": 4.0,
            "right_hip_yaw": 10.0,
            "right_hip_roll": 10.0,
            "right_hip_pitch": 10.0,
            "right_knee": 10.0,
            "right_ankle": 4.0,
            "torso": 4.0,
            "left_shoulder_pitch": 2.0,
            "left_shoulder_roll": 2.0,
            "left_shoulder_yaw": 2.0,
            "left_elbow": 2.0,
            "right_shoulder_pitch": 2.0,
            "right_shoulder_roll": 2.0,
            "right_shoulder_yaw": 2.0,
            "right_elbow": 2.0
        }
        
        # Default standing pose (targets for actions near 0)
        self.default_pose = {
            "left_hip_yaw": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_pitch": -0.2,
            "left_knee": 0.5,
            "left_ankle": -0.3,
            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.0,
            "right_hip_pitch": -0.2,
            "right_knee": 0.5,
            "right_ankle": -0.3,
            "torso": 0.0,
            "left_shoulder_pitch": 0.0,
            "left_shoulder_roll": 0.0,
            "left_shoulder_yaw": 0.0,
            "left_elbow": 0.0,
            "right_shoulder_pitch": 0.0,
            "right_shoulder_roll": 0.0,
            "right_shoulder_yaw": 0.0,
            "right_elbow": 0.0
        }
        
        # Action scaling (smaller range around default pose)
        self.action_scale = {
            "left_hip_yaw": 0.2,
            "left_hip_roll": 0.2,
            "left_hip_pitch": 0.3,
            "left_knee": 0.4,
            "left_ankle": 0.3,
            "right_hip_yaw": 0.2,
            "right_hip_roll": 0.2,
            "right_hip_pitch": 0.3,
            "right_knee": 0.4,
            "right_ankle": 0.3,
            "torso": 0.3,
            "left_shoulder_pitch": 0.5,
            "left_shoulder_roll": 0.5,
            "left_shoulder_yaw": 0.8,
            "left_elbow": 0.8,
            "right_shoulder_pitch": 0.5,
            "right_shoulder_roll": 0.5,
            "right_shoulder_yaw": 0.8,
            "right_elbow": 0.8
        }
        
        # Reset to standing position
        self.reset()
        
        self.prev_feet_contact = [True, True]
        self.feet_air_time = [0.0, 0.0]
        self.last_time = 0.0
        
    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        
        # Revert to original working initialization
        # Left leg
        self.data.qpos[9] = -0.2   # Left hip pitch
        self.data.qpos[10] = 0.5 # Left knee 0.6 orginally
        self.data.qpos[11] = -0.3  # Left ankle
        
        # Right leg
        self.data.qpos[14] = -0.2  # Right hip pitch
        self.data.qpos[15] = 0.5  # Right knee
        self.data.qpos[16] = -0.3  # Right ankle
        '''
        self.data.qpos[9] = 0.3    # Left hip pitch
        self.data.qpos[10] = -0.6  # Left knee 0.6 orginally
        self.data.qpos[11] = -0.3  # Left ankle
        
        # Right leg
        self.data.qpos[14] = 0.3   # Right hip pitch
        self.data.qpos[15] = -0.6  # Right knee
        self.data.qpos[16] = -0.3  # Right ankle
        
        data.qpos[model.joint('left_hip_pitch').qposadr] = -.2
        data.qpos[model.joint('left_knee').qposadr] = 0.5
        data.qpos[model.joint('left_ankle').qposadr] = -.3
        data.qpos[model.joint('right_hip_pitch').qposadr] = -.2
        data.qpos[model.joint('right_knee').qposadr] = .5
        data.qpos[model.joint('right_ankle').qposadr] = -.3
        '''
        # Torso
        self.data.qpos[self.joint_ids["torso"]] = 0.0
        
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
 
        # Explicit joint positions
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
        torso_pos = self.data.qpos[self.joint_ids["torso"]]
        left_shoulder_pitch_pos = self.data.qpos[self.joint_ids["left_shoulder_pitch"]]
        left_shoulder_roll_pos = self.data.qpos[self.joint_ids["left_shoulder_roll"]]
        left_shoulder_yaw_pos = self.data.qpos[self.joint_ids["left_shoulder_yaw"]]
        left_elbow_pos = self.data.qpos[self.joint_ids["left_elbow"]]
        right_shoulder_pitch_pos = self.data.qpos[self.joint_ids["right_shoulder_pitch"]]
        right_shoulder_roll_pos = self.data.qpos[self.joint_ids["right_shoulder_roll"]]
        right_shoulder_yaw_pos = self.data.qpos[self.joint_ids["right_shoulder_yaw"]]
        right_elbow_pos = self.data.qpos[self.joint_ids["right_elbow"]]
 
        # Explicit joint velocities
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
        torso_vel = self.data.qvel[self.joint_ids["torso"]]
        left_shoulder_pitch_vel = self.data.qvel[self.joint_ids["left_shoulder_pitch"]]
        left_shoulder_roll_vel = self.data.qvel[self.joint_ids["left_shoulder_roll"]]
        left_shoulder_yaw_vel = self.data.qvel[self.joint_ids["left_shoulder_yaw"]]
        left_elbow_vel = self.data.qvel[self.joint_ids["left_elbow"]]
        right_shoulder_pitch_vel = self.data.qvel[self.joint_ids["right_shoulder_pitch"]]
        right_shoulder_roll_vel = self.data.qvel[self.joint_ids["right_shoulder_roll"]]
        right_shoulder_yaw_vel = self.data.qvel[self.joint_ids["right_shoulder_yaw"]]
        right_elbow_vel = self.data.qvel[self.joint_ids["right_elbow"]]
 
        return np.array([
            roll, pitch, ang_vel[0], ang_vel[1],
            left_hip_yaw_pos, left_hip_roll_pos, left_hip_pitch_pos, left_knee_pos, left_ankle_pos,
            right_hip_yaw_pos, right_hip_roll_pos, right_hip_pitch_pos, right_knee_pos, right_ankle_pos,
            torso_pos,
            left_shoulder_pitch_pos, left_shoulder_roll_pos, left_shoulder_yaw_pos, left_elbow_pos,
            right_shoulder_pitch_pos, right_shoulder_roll_pos, right_shoulder_yaw_pos, right_elbow_pos,
            left_hip_yaw_vel, left_hip_roll_vel, left_hip_pitch_vel, left_knee_vel, left_ankle_vel,
            right_hip_yaw_vel, right_hip_roll_vel, right_hip_pitch_vel, right_knee_vel, right_ankle_vel,
            torso_vel,
            left_shoulder_pitch_vel, left_shoulder_roll_vel, left_shoulder_yaw_vel, left_elbow_vel,
            right_shoulder_pitch_vel, right_shoulder_roll_vel, right_shoulder_yaw_vel, right_elbow_vel
        ], dtype=np.float32)
    
    def _foot_in_contact(self, foot_geom_names):
        geom_ids = [self.model.geom(name).id for name in foot_geom_names]
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 in geom_ids or contact.geom2 in geom_ids:
                return True
        return False
 
    def _get_reward(self, obs, return_info=False):
        """Improved reward function for better standing stability"""
        torso_height = self.data.body('torso_link').xpos[2]
        
        # --- Orientation reward (encourage upright posture) ---
        k_orientation = 10.0
        torso_mat = self.data.xmat[self.torso_body.id].reshape(3, 3)
        z_axis = torso_mat[:, 2]
        # Reward for being upright (z_axis[2] close to 1)
        upright_reward = k_orientation * z_axis[2]  # z_axis[2] = 1 when perfectly upright
        reward = upright_reward

        # --- Angular velocity damping (encourage stability) ---
        k_ang_vel = 5.0
        ang_vel = self.data.qvel[3:6]  # Angular velocity of torso
        ang_vel_penalty = -k_ang_vel * np.sum(np.square(ang_vel))
        reward += ang_vel_penalty

        # --- Base height reward (encourage proper standing height) ---
        k_base_height = 8.0
        target_height = 0.98
        height_error = abs(torso_height - target_height)
        height_reward = -k_base_height * height_error
        reward += height_reward

        # --- Foot contact reward (critical for standing) ---
        k_foot_contact = 5.0
        left_foot_contact = self._foot_in_contact(["left_foot1", "left_foot2", "left_foot3"])
        right_foot_contact = self._foot_in_contact(["right_foot1", "right_foot2", "right_foot3"])
        foot_contact_reward = k_foot_contact * (float(left_foot_contact) + float(right_foot_contact))
        reward += foot_contact_reward

        # --- Posture reward (encourage standing pose) ---
        k_posture = 2.0
        posture_penalty = 0.0
        for joint_name in ["left_hip_pitch", "left_knee", "left_ankle", 
                          "right_hip_pitch", "right_knee", "right_ankle"]:
            joint_id = self.joint_ids[joint_name]
            current_pos = self.data.qpos[joint_id]
            target_pos = self.default_pose[joint_name]
            posture_penalty += (current_pos - target_pos) ** 2
        posture_reward = -k_posture * posture_penalty
        reward += posture_reward

        # --- Joint velocity penalty (encourage smooth motion) ---
        k_joint_vel = 0.1
        joint_vel_penalty = -k_joint_vel * np.sum(np.square(self.data.qvel[6:]))  # All joint velocities
        reward += joint_vel_penalty

        # --- Torque penalty (energy efficiency) ---
        k_torque = 0.01
        torque_penalty = -k_torque * np.sum(np.square(self.data.ctrl))
        reward += torque_penalty

        # --- Joint limit penalty ---
        k_joint_limit = 1.0
        joint_limit_penalty = 0.0
        for joint_name in self.joint_ids:
            joint_id = self.joint_ids[joint_name]
            q = self.data.qpos[joint_id]
            qmin = self.model.jnt_range[joint_id][0]
            qmax = self.model.jnt_range[joint_id][1]
            # Penalize if close to limits (within 20% of range)
            margin = 0.2 * (qmax - qmin)
            if q < qmin + margin:
                joint_limit_penalty -= k_joint_limit * ((qmin + margin - q) ** 2)
            elif q > qmax - margin:
                joint_limit_penalty -= k_joint_limit * ((q - qmax + margin) ** 2)
        reward += joint_limit_penalty

        # --- Alive bonus ---
        alive_bonus = 1.0
        reward += alive_bonus

        if return_info:
            return reward, {
                'upright_reward': upright_reward,
                'ang_vel_penalty': ang_vel_penalty,
                'height_reward': height_reward,
                'foot_contact_reward': foot_contact_reward,
                'posture_reward': posture_reward,
                'joint_vel_penalty': joint_vel_penalty,
                'torque_penalty': torque_penalty,
                'joint_limit_penalty': joint_limit_penalty,
                'alive_bonus': alive_bonus,
                'total_reward': reward
            }
        return reward
 
    def _get_terminated(self, obs):
        """Check termination conditions"""
        roll, pitch = obs[0], obs[1]
        torso_height = self.data.xpos[self.torso_body.id][2]
        
        fallen = abs(roll) > 0.5 or abs(pitch) > 0.5 or torso_height < 0.5
        timeout = self.data.time > 13.0
        
        return fallen or timeout
    def step(self, action):
        # Improved PD control: actions are small deviations around default standing pose
        joint_names = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
                      "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
                      "torso", "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
                      "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"]
        
        # Compute desired positions as: default_pose + action * scale
        for i, joint_name in enumerate(joint_names):
            joint_id = self.joint_ids[joint_name]
            actuator_id = self.actuator_ids[joint_name]
            
            # Desired position: default pose + scaled action
            q_des = self.default_pose[joint_name] + action[i] * self.action_scale[joint_name]
            
            # Clamp to joint limits
            qmin = self.model.jnt_range[joint_id][0]
            qmax = self.model.jnt_range[joint_id][1]
            q_des = np.clip(q_des, qmin, qmax)
            
            # PD control
            q_current = self.data.qpos[joint_id]
            qvel_current = self.data.qvel[joint_id]
            
            self.data.ctrl[actuator_id] = (self.Kp[joint_name] * (q_des - q_current) - 
                                         self.Kd[joint_name] * qvel_current)
            
            # Clamp control to actuator limits
            ctrl_min = self.model.actuator_ctrlrange[actuator_id][0]
            ctrl_max = self.model.actuator_ctrlrange[actuator_id][1]
            self.data.ctrl[actuator_id] = np.clip(self.data.ctrl[actuator_id], ctrl_min, ctrl_max)

        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward, reward_info = self._get_reward(obs, return_info=True)
        terminated = self._get_terminated(obs)
        info = reward_info
        return obs, reward, terminated, False, info
 