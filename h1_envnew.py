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
        
        # Reset to standing position
        self.reset()
        
        self.prev_feet_contact = [True, True]
        self.feet_air_time = [0.0, 0.0]
        self.last_time = 0.0
        
    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        
        # Revert to original working initialization
        # Left leg
        self.data.qpos[9] = 0.3    # Left hip pitch
        self.data.qpos[10] = -0.6  # Left knee 0.6 orginally
        self.data.qpos[11] = -0.3  # Left ankle
        
        # Right leg
        self.data.qpos[14] = 0.3   # Right hip pitch
        self.data.qpos[15] = -0.6  # Right knee
        self.data.qpos[16] = -0.3  # Right ankle
        
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
        """Richer reward: orientation, base height, torque penalty, foot contact, joint limit penalty, alive bonus, forward velocity, with diagnostics"""
        torso_height = self.data.body('torso_link').xpos[2]
        # --- Orientation penalty ---
        k_orientation = 0.1
        torso_mat = self.data.xmat[self.torso_body.id].reshape(3, 3)
        z_axis = torso_mat[:, 2]
        gp = abs(z_axis[0]) + abs(z_axis[1]) + abs(z_axis[2])
        Rorientation = -k_orientation * gp
        reward = Rorientation

        # --- Base height penalty ---
        k_base_height = 1.0
        hbase = torso_height
        Rbase_height = -k_base_height * ((hbase - 0.96) ** 2)
        reward += Rbase_height

        # --- Torque penalty ---
        k_torque = .0001 #0.001
        torque_penalty = -k_torque * np.sum(np.square(self.data.ctrl))
        reward += torque_penalty

        # --- Foot contact encouragement (reward for both feet in contact) ---
        k_foot_contact = 1 #0.2
        left_foot_contact = self._foot_in_contact(["left_foot1", "left_foot2", "left_foot3"])
        right_foot_contact = self._foot_in_contact(["right_foot1", "right_foot2", "right_foot3"])
        foot_contact_reward = k_foot_contact * (float(left_foot_contact) + float(right_foot_contact))
        reward += foot_contact_reward

        # --- Joint limit penalty ---
        k_joint_limit = 0.01
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
        alive_bonus = 0.1 * self.data.time
        reward += alive_bonus

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
                'Rbase_height': Rbase_height,
                'torque_penalty': torque_penalty,
                'foot_contact_reward': foot_contact_reward,
                'joint_limit_penalty': joint_limit_penalty,
                'alive_bonus': alive_bonus,
                #'forward_vel_reward': forward_vel_reward,
                'total_reward': reward
            }
        return reward

    def _get_terminated(self, obs):
        """Check termination conditions"""
        roll, pitch = obs[0], obs[1]
        torso_height = self.data.xpos[self.torso_body.id][2]
        
        fallen = abs(roll) > 0.5 or abs(pitch) > 0.5 or torso_height < 0.5
        timeout = self.data.time > 10.0
        
        return fallen or timeout
    #MODIFY THIS SO IT MOVES ALL JOINT
    def step(self, action):
        # Apply scaled control inputs
        # Explicit, line-by-line for all 19 actuators/joints
        self.data.ctrl[self.actuator_ids["left_hip_yaw"]] = action[0] * self.joint_scaling["left_hip_yaw"]
        self.data.ctrl[self.actuator_ids["left_hip_roll"]] = action[1] * self.joint_scaling["left_hip_roll"]
        self.data.ctrl[self.actuator_ids["left_hip_pitch"]] = action[2] * self.joint_scaling["left_hip_pitch"]
        self.data.ctrl[self.actuator_ids["left_knee"]] = action[3] * self.joint_scaling["left_knee"]
        self.data.ctrl[self.actuator_ids["left_ankle"]] = action[4] * self.joint_scaling["left_ankle"]
        self.data.ctrl[self.actuator_ids["right_hip_yaw"]] = action[5] * self.joint_scaling["right_hip_yaw"]
        self.data.ctrl[self.actuator_ids["right_hip_roll"]] = action[6] * self.joint_scaling["right_hip_roll"]
        self.data.ctrl[self.actuator_ids["right_hip_pitch"]] = action[7] * self.joint_scaling["right_hip_pitch"]
        self.data.ctrl[self.actuator_ids["right_knee"]] = action[8] * self.joint_scaling["right_knee"]
        self.data.ctrl[self.actuator_ids["right_ankle"]] = action[9] * self.joint_scaling["right_ankle"]
        self.data.ctrl[self.actuator_ids["torso"]] = action[10] * self.joint_scaling["torso"]
        self.data.ctrl[self.actuator_ids["left_shoulder_pitch"]] = action[11] * self.joint_scaling["left_shoulder_pitch"]
        self.data.ctrl[self.actuator_ids["left_shoulder_roll"]] = action[12] * self.joint_scaling["left_shoulder_roll"]
        self.data.ctrl[self.actuator_ids["left_shoulder_yaw"]] = action[13] * self.joint_scaling["left_shoulder_yaw"]
        self.data.ctrl[self.actuator_ids["left_elbow"]] = action[14] * self.joint_scaling["left_elbow"]
        self.data.ctrl[self.actuator_ids["right_shoulder_pitch"]] = action[15] * self.joint_scaling["right_shoulder_pitch"]
        self.data.ctrl[self.actuator_ids["right_shoulder_roll"]] = action[16] * self.joint_scaling["right_shoulder_roll"]
        self.data.ctrl[self.actuator_ids["right_shoulder_yaw"]] = action[17] * self.joint_scaling["right_shoulder_yaw"]
        self.data.ctrl[self.actuator_ids["right_elbow"]] = action[18] * self.joint_scaling["right_elbow"]
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward, reward_info = self._get_reward(obs, return_info=True)
        terminated = self._get_terminated(obs)
        info = reward_info
        return obs, reward, terminated, False, info
