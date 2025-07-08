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
        
        # Joint-specific PD gains
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
            "right_ankle": 40.0,
            "torso": 30.0,
            "left_shoulder_pitch": 20.0,
            "left_shoulder_roll": 20.0,
            "left_shoulder_yaw": 10.0,
            "left_elbow": 10.0,
            "right_shoulder_pitch": 20.0,
            "right_shoulder_roll": 20.0,
            "right_shoulder_yaw": 10.0,
            "right_elbow": 10.0
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
            "right_ankle": 1.5,
            "torso": 1.0,
            "left_shoulder_pitch": 0.8,
            "left_shoulder_roll": 0.8,
            "left_shoulder_yaw": 0.5,
            "left_elbow": 0.5,
            "right_shoulder_pitch": 0.8,
            "right_shoulder_roll": 0.8,
            "right_shoulder_yaw": 0.5,
            "right_elbow": 0.5
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
    
    def _reward_hip_pos(self):
        """Calculate hip position penalty - penalizes deviation from neutral hip positions"""
        # Get hip yaw and hip roll positions for both legs
        left_hip_yaw = self.data.qpos[self.joint_ids["left_hip_yaw"]]
        left_hip_roll = self.data.qpos[self.joint_ids["left_hip_roll"]]
        right_hip_yaw = self.data.qpos[self.joint_ids["right_hip_yaw"]]
        right_hip_roll = self.data.qpos[self.joint_ids["right_hip_roll"]]
        
        # Calculate squared positions (penalty for deviation from 0)
        hip_positions = np.array([left_hip_yaw, left_hip_roll, right_hip_yaw, right_hip_roll])
        return np.sum(np.square(hip_positions))
 
    def _foot_in_contact(self, foot_geom_names):
        geom_ids = [self.model.geom(name).id for name in foot_geom_names]
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 in geom_ids or contact.geom2 in geom_ids:
                return True
        return False
 
    def _get_reward(self, obs, return_info=False):
        """Richer reward: orientation, base height, torque penalty, foot contact, joint limit penalty, alive bonus, forward velocity, with diagnostics"""
        torso_height = self.data.body('torso_link').xpos[2]        # --- Orientation penalty (most important for standing) ---
        k_orientation = 2.0  # Increased for better stability
        torso_mat = self.data.xmat[self.torso_body.id].reshape(3, 3)
        z_axis = torso_mat[:, 2]
        gp = abs(z_axis[0]) + abs(z_axis[1]) + abs(z_axis[2])
        Rorientation = -k_orientation * gp
        reward = Rorientation

        # --- Base height penalty ---
        k_base_height = 1.5  # Increased for height importance
        hbase = torso_height
        height_penalty = 0
        if hbase < 0.96:  # Lower threshold for shorter stance
            height_penalty = -k_base_height * (0.96 - hbase)
        elif hbase > 1.0:
            height_penalty = -k_base_height * (hbase - 1.0)  # Encourage not too high
        reward += height_penalty

        # --- Torque penalty (energy efficiency) ---
        k_torque = 0.001  # Increased to discourage excessive torques
        torque_penalty = -k_torque * np.sum(np.square(self.data.ctrl))
        reward += torque_penalty        # --- Foot contact encouragement (critical for standing) ---
        k_foot_contact = 2.0  # Increased - very important for stability
        left_foot_contact = self._foot_in_contact(["left_foot1", "left_foot2", "left_foot3"])
        right_foot_contact = self._foot_in_contact(["right_foot1", "right_foot2", "right_foot3"])
        foot_contact_reward = k_foot_contact * (float(left_foot_contact) + float(right_foot_contact))
        reward += foot_contact_reward

        # --- Joint limit penalty ---
        k_joint_limit = 0.1  # Increased to better avoid joint limits
        joint_limit_penalty = 0.0
        for joint_name in self.joint_ids:
            joint_id = self.joint_ids[joint_name]
            q = self.data.qpos[joint_id]
            qmin = self.model.jnt_range[joint_id][0]
            qmax = self.model.jnt_range[joint_id][1]
            # Penalize if close to limits (within 10% of range)
            margin = 0.1 * (qmax - qmin)
            if q < qmin + margin or q > qmax - margin:
                joint_limit_penalty -= k_joint_limit
        reward += joint_limit_penalty

        # --- Alive bonus (scaled with time) ---
        alive_bonus = 0.05 * self.data.time  # Reduced to not dominate other rewards
        reward += alive_bonus        
        
        # --- Hip position penalty (encourage neutral hip positions) ---
        k_hip_pos = 0.5  # Reduced from 1.0 to balance with other penalties
        hip_pos_penalty = self._reward_hip_pos()
        reward -= k_hip_pos * hip_pos_penalty

        # --- Torso position penalty (encourage stable torso) ---
        k_torso_pos = 1.0  # Increased for better torso stability
        torso_pos = self.data.qpos[self.joint_ids["torso"]]
        torso_pos_penalty = k_torso_pos * (torso_pos ** 2)
        reward -= torso_pos_penalty

        # --- Forward velocity reward (encourage walking) ---
        '''
        k_forward_vel = 0.5
        base_forward_vel = self.data.qvel[0]  # x velocity of base
        forward_vel_reward = k_forward_vel * base_forward_vel
        reward += forward_vel_reward
        '''
        if return_info:
            return reward, {
                'Rorientation': Rorientation,
                'height_penalty': height_penalty,
                'torque_penalty': torque_penalty,
                'foot_contact_reward': foot_contact_reward,
                'joint_limit_penalty': joint_limit_penalty,
                'alive_bonus': alive_bonus,
                'hip_pos_penalty': -k_hip_pos * hip_pos_penalty,
                'torso_pos_penalty': -torso_pos_penalty,
                #'forward_vel_reward': forward_vel_reward,
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
    #MODIFY THIS SO IT MOVES ALL JOINT
    def step(self, action):
        # PD control: interpret action as desired position in [-1, 1] scaled to joint range
        # Explicit, line-by-line for all 19 actuators/joints
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
        q_des_torso = 0.5 * (action[10] + 1) * (self.model.jnt_range[self.joint_ids["torso"]][1] - self.model.jnt_range[self.joint_ids["torso"]][0]) + self.model.jnt_range[self.joint_ids["torso"]][0]
        q_des_left_shoulder_pitch = 0.5 * (action[11] + 1) * (self.model.jnt_range[self.joint_ids["left_shoulder_pitch"]][1] - self.model.jnt_range[self.joint_ids["left_shoulder_pitch"]][0]) + self.model.jnt_range[self.joint_ids["left_shoulder_pitch"]][0]
        q_des_left_shoulder_roll = 0.5 * (action[12] + 1) * (self.model.jnt_range[self.joint_ids["left_shoulder_roll"]][1] - self.model.jnt_range[self.joint_ids["left_shoulder_roll"]][0]) + self.model.jnt_range[self.joint_ids["left_shoulder_roll"]][0]
        q_des_left_shoulder_yaw = 0.5 * (action[13] + 1) * (self.model.jnt_range[self.joint_ids["left_shoulder_yaw"]][1] - self.model.jnt_range[self.joint_ids["left_shoulder_yaw"]][0]) + self.model.jnt_range[self.joint_ids["left_shoulder_yaw"]][0]
        q_des_left_elbow = 0.5 * (action[14] + 1) * (self.model.jnt_range[self.joint_ids["left_elbow"]][1] - self.model.jnt_range[self.joint_ids["left_elbow"]][0]) + self.model.jnt_range[self.joint_ids["left_elbow"]][0]
        q_des_right_shoulder_pitch = 0.5 * (action[15] + 1) * (self.model.jnt_range[self.joint_ids["right_shoulder_pitch"]][1] - self.model.jnt_range[self.joint_ids["right_shoulder_pitch"]][0]) + self.model.jnt_range[self.joint_ids["right_shoulder_pitch"]][0]
        q_des_right_shoulder_roll = 0.5 * (action[16] + 1) * (self.model.jnt_range[self.joint_ids["right_shoulder_roll"]][1] - self.model.jnt_range[self.joint_ids["right_shoulder_roll"]][0]) + self.model.jnt_range[self.joint_ids["right_shoulder_roll"]][0]
        q_des_right_shoulder_yaw = 0.5 * (action[17] + 1) * (self.model.jnt_range[self.joint_ids["right_shoulder_yaw"]][1] - self.model.jnt_range[self.joint_ids["right_shoulder_yaw"]][0]) + self.model.jnt_range[self.joint_ids["right_shoulder_yaw"]][0]
        q_des_right_elbow = 0.5 * (action[18] + 1) * (self.model.jnt_range[self.joint_ids["right_elbow"]][1] - self.model.jnt_range[self.joint_ids["right_elbow"]][0]) + self.model.jnt_range[self.joint_ids["right_elbow"]][0]
 
        # PD control for each joint
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
        self.data.ctrl[self.actuator_ids["torso"]] = self.Kp["torso"] * (q_des_torso - self.data.qpos[self.joint_ids["torso"]]) - self.Kd["torso"] * self.data.qvel[self.joint_ids["torso"]]
        self.data.ctrl[self.actuator_ids["left_shoulder_pitch"]] = self.Kp["left_shoulder_pitch"] * (q_des_left_shoulder_pitch - self.data.qpos[self.joint_ids["left_shoulder_pitch"]]) - self.Kd["left_shoulder_pitch"] * self.data.qvel[self.joint_ids["left_shoulder_pitch"]]
        self.data.ctrl[self.actuator_ids["left_shoulder_roll"]] = self.Kp["left_shoulder_roll"] * (q_des_left_shoulder_roll - self.data.qpos[self.joint_ids["left_shoulder_roll"]]) - self.Kd["left_shoulder_roll"] * self.data.qvel[self.joint_ids["left_shoulder_roll"]]
        self.data.ctrl[self.actuator_ids["left_shoulder_yaw"]] = self.Kp["left_shoulder_yaw"] * (q_des_left_shoulder_yaw - self.data.qpos[self.joint_ids["left_shoulder_yaw"]]) - self.Kd["left_shoulder_yaw"] * self.data.qvel[self.joint_ids["left_shoulder_yaw"]]
        self.data.ctrl[self.actuator_ids["left_elbow"]] = self.Kp["left_elbow"] * (q_des_left_elbow - self.data.qpos[self.joint_ids["left_elbow"]]) - self.Kd["left_elbow"] * self.data.qvel[self.joint_ids["left_elbow"]]
        self.data.ctrl[self.actuator_ids["right_shoulder_pitch"]] = self.Kp["right_shoulder_pitch"] * (q_des_right_shoulder_pitch - self.data.qpos[self.joint_ids["right_shoulder_pitch"]]) - self.Kd["right_shoulder_pitch"] * self.data.qvel[self.joint_ids["right_shoulder_pitch"]]
        self.data.ctrl[self.actuator_ids["right_shoulder_roll"]] = self.Kp["right_shoulder_roll"] * (q_des_right_shoulder_roll - self.data.qpos[self.joint_ids["right_shoulder_roll"]]) - self.Kd["right_shoulder_roll"] * self.data.qvel[self.joint_ids["right_shoulder_roll"]]
        self.data.ctrl[self.actuator_ids["right_shoulder_yaw"]] = self.Kp["right_shoulder_yaw"] * (q_des_right_shoulder_yaw - self.data.qpos[self.joint_ids["right_shoulder_yaw"]]) - self.Kd["right_shoulder_yaw"] * self.data.qvel[self.joint_ids["right_shoulder_yaw"]]
        self.data.ctrl[self.actuator_ids["right_elbow"]] = self.Kp["right_elbow"] * (q_des_right_elbow - self.data.qpos[self.joint_ids["right_elbow"]]) - self.Kd["right_elbow"] * self.data.qvel[self.joint_ids["right_elbow"]]
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward, reward_info = self._get_reward(obs, return_info=True)
        terminated = self._get_terminated(obs)
        info = reward_info
        return obs, reward, terminated, False, info
 