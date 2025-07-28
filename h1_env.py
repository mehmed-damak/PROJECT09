import mujoco
import numpy as np
import sys
from gymnasium import Env, spaces
from walking_rewards import (create_clock_functions, calc_foot_frc_clock_reward, 
                            calc_foot_vel_clock_reward, calc_height_reward,
                            calc_orientation_reward, calc_step_reward)
 
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
        """Reward for walking with balance and stability."""
        
        # Extract observations
        roll = obs[0]
        pitch = obs[1]
        ang_vel = obs[2:4]
        
        # Get torso position and orientation
        torso_height = self.data.xpos[self.torso_body.id][2]
        torso_pos = self.data.xpos[self.torso_body.id]
        
        # Get current velocity
        torso_vel = self.data.qvel[0:3]  # linear velocity in x, y, z
        forward_vel = torso_vel[0]  # velocity in x direction (forward)
        lateral_vel = abs(torso_vel[1])  # absolute velocity in y direction (sideways)
        
        # 1. Upright orientation reward (most important for standing/walking)
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
        
        # 3. Forward velocity reward (new for walking)
        target_forward_speed = 1.0  # m/s
        # Reward forward movement, with optimal speed around 1.0 m/s
        if forward_vel > 0:
            # Gaussian reward centered at target speed
            forward_reward = 3.0 * np.exp(-0.5 * ((forward_vel - target_forward_speed) ** 2) / (0.5 ** 2))
        else:
            # Penalty for moving backward
            forward_reward = -1.0 * abs(forward_vel)
        
        # 4. Lateral stability reward (discourage sideways movement)
        lateral_penalty = -2.0 * lateral_vel
        
        # 4. Lateral stability reward (discourage sideways movement)
        lateral_penalty = -2.0 * lateral_vel
        
        # 5. Foot contact reward with walking adaptation
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
        
        # Update foot air time tracking
        dt = self.data.time - self.last_time
        self.last_time = self.data.time
        
        # Track air time for gait analysis
        if not left_foot_contact:
            self.feet_air_time[0] += dt
        else:
            self.feet_air_time[0] = 0.0
            
        if not right_foot_contact:
            self.feet_air_time[1] += dt
        else:
            self.feet_air_time[1] = 0.0
        
        # Foot contact reward adapted for walking
        if left_foot_contact and right_foot_contact:
            # Both feet on ground - good for standing, but not ideal for walking
            foot_contact_reward = 0.5 if forward_vel < 0.1 else 0.1
        elif left_foot_contact or right_foot_contact:
            # One foot on ground - good for walking
            foot_contact_reward = 1.0
        else:
            # No feet on ground - bad unless it's a brief flight phase
            max_air_time = max(self.feet_air_time)
            if max_air_time < 0.1:  # Brief flight phase is okay
                foot_contact_reward = 0.0
            else:
                foot_contact_reward = -2.0  # Penalty for long air time
        
        # 6. Gait reward (encourage alternating foot contacts)
        gait_reward = 0.0
        if len(self.prev_feet_contact) == 2:
            # Reward transitions from one foot to the other
            current_contacts = [left_foot_contact, right_foot_contact]
            if (self.prev_feet_contact[0] and not self.prev_feet_contact[1] and 
                not current_contacts[0] and current_contacts[1]):
                gait_reward = 0.5  # Left to right transition
            elif (not self.prev_feet_contact[0] and self.prev_feet_contact[1] and 
                  current_contacts[0] and not current_contacts[1]):
                gait_reward = 0.5  # Right to left transition
        
        self.prev_feet_contact = [left_foot_contact, right_foot_contact]
        
        # 7. Stability reward (low angular velocities but allow some for walking)
        angular_velocity_penalty = -0.3 * np.sum(np.square(ang_vel))
        
        # 8. Energy efficiency reward (discourage excessive joint movements)
        joint_vel_penalty = -0.1 * np.sum(np.square(obs[14:24]))  # joint velocities
        
        # 9. Base bonus for staying alive and not terminating
        alive_bonus = 0.5
        
        # 9. Base bonus for staying alive and not terminating
        alive_bonus = 0.5
        
        # Combine all rewards with appropriate weights
        total_reward = (
            orientation_reward +           # 3.0 max - most important
            height_reward +               # 2.0 max - second most important  
            forward_reward +              # 3.0 max - encourage forward movement
            lateral_penalty +             # discourage sideways drift
            foot_contact_reward +         # 1.0 max - encourage proper foot contact
            gait_reward +                 # 0.5 max - encourage gait patterns
            angular_velocity_penalty +    # stability
            joint_vel_penalty +           # energy efficiency
            alive_bonus                   # survival
        )

        if return_info:
            return total_reward, {
                'orientation_reward': orientation_reward,
                'height_reward': height_reward,
                'forward_reward': forward_reward,
                'lateral_penalty': lateral_penalty,
                'foot_contact_reward': foot_contact_reward,
                'gait_reward': gait_reward,
                'angular_velocity_penalty': angular_velocity_penalty,
                'joint_vel_penalty': joint_vel_penalty,
                'alive_bonus': alive_bonus,
                'total_reward': total_reward,
                'forward_vel': forward_vel,
                'left_foot_contact': left_foot_contact,
                'right_foot_contact': right_foot_contact
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


class H1WalkEnv(H1StandEnv):
    """
    H1 Walking Environment that integrates with LearningHumanoidWalking framework
    """
    def __init__(self):
        # Initialize walking task components BEFORE calling super().__init__
        self._init_walking_task()
        
        # Now initialize the parent class
        super().__init__()
        
        # Extended observation space for walking (includes clock phase and target info)
        obs_size = 2 + 2 + len(self.joint_ids)*2 + 3  # phase + target distance + original obs
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
    def _init_walking_task(self):
        """Initialize walking task parameters"""
        # Clock phase parameters
        self._phase = 0
        self._period = 80  # 2 seconds at 40Hz
        
        # Walking parameters
        self._swing_duration = 0.4
        self._stance_duration = 0.6
        self._goal_speed_ref = 0.5  # m/s
        self._goal_height_ref = 0.98  # meters (H1 standing height)
        
        # Load footstep plans
        self._load_footstep_plans()
        
        # Initialize current target
        self._current_plan_idx = 0
        self._current_step_idx = 0
        self._target_reached = False
        
        # Initialize clock functions for gait
        self._init_clock_functions()
        
        # Foot force/velocity tracking
        self.prev_left_foot_pos = np.array([0.0, 0.15, 0.0])  # H1 foot spacing
        self.prev_right_foot_pos = np.array([0.0, -0.15, 0.0])
        
    def _load_footstep_plans(self):
        """Load pre-generated footstep plans"""
        try:
            with open('/home/mehmed-damak/PROJECT09/LearningHumanoidWalking/utils/footstep_plans.txt', 'r') as fn:
                lines = [l.strip() for l in fn.readlines()]
            
            self.footstep_plans = []
            sequence = []
            for line in lines:
                if line == '---':
                    if len(sequence):
                        self.footstep_plans.append(sequence)
                    sequence = []
                    continue
                else:
                    sequence.append(np.array([float(l) for l in line.split(',')]))
            
            if len(sequence):  # Add last sequence
                self.footstep_plans.append(sequence)
                
        except FileNotFoundError:
            # Create simple forward walking plan as fallback
            self.footstep_plans = [self._create_simple_walking_plan()]
    
    def _create_simple_walking_plan(self):
        """Create a simple forward walking footstep plan"""
        plan = []
        for i in range(10):  # 10 steps
            # Alternate between left (y=0.15) and right (y=-0.15) feet
            y_pos = 0.15 if i % 2 == 0 else -0.15
            x_pos = i * 0.3  # 30cm step length
            plan.append(np.array([x_pos, y_pos, 0.0, 0.0]))  # [x, y, z, heading]
        return plan
    
    def _init_clock_functions(self):
        """Initialize clock functions for gait timing"""
        self.right_clock, self.left_clock = create_clock_functions(
            self._swing_duration, self._stance_duration, self._period
        )
    
    def _get_current_footstep_targets(self):
        """Get current footstep targets"""
        if not self.footstep_plans:
            return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
        
        current_plan = self.footstep_plans[self._current_plan_idx]
        
        # Get next two targets (for calculating midpoint)
        t1_idx = min(self._current_step_idx, len(current_plan) - 1)
        t2_idx = min(self._current_step_idx + 1, len(current_plan) - 1)
        
        t1 = current_plan[t1_idx][:3]  # [x, y, z]
        t2 = current_plan[t2_idx][:3]
        
        return t1, t2
    
    def _get_foot_positions(self):
        """Get current foot positions"""
        # For H1, we need to get the foot body positions
        # This is a simplified version - you may need to adjust based on your H1 model
        left_foot_pos = np.array([
            self.data.body('left_ankle_link').xpos[0],
            self.data.body('left_ankle_link').xpos[1], 
            self.data.body('left_ankle_link').xpos[2]
        ])
        
        right_foot_pos = np.array([
            self.data.body('right_ankle_link').xpos[0],
            self.data.body('right_ankle_link').xpos[1],
            self.data.body('right_ankle_link').xpos[2]
        ])
        
        return left_foot_pos, right_foot_pos
    
    def _get_foot_velocities(self):
        """Calculate foot velocities"""
        left_foot_pos, right_foot_pos = self._get_foot_positions()
        
        left_foot_vel = left_foot_pos - self.prev_left_foot_pos
        right_foot_vel = right_foot_pos - self.prev_right_foot_pos
        
        self.prev_left_foot_pos = left_foot_pos.copy()
        self.prev_right_foot_pos = right_foot_pos.copy()
        
        return left_foot_vel, right_foot_vel
    
    def _get_foot_forces(self):
        """Get foot contact forces (simplified)"""
        # This is a simplified version - you may need to implement proper force sensing
        left_force = 0.0
        right_force = 0.0
        
        # Estimate robot mass (H1 is approximately 80kg)
        robot_mass = 80.0
        
        # Check contact forces if your model has force sensors
        # For now, estimate based on contact detection
        if self._is_foot_in_contact('left'):
            left_force = robot_mass * 9.81 / 2  # Half body weight
        if self._is_foot_in_contact('right'):
            right_force = robot_mass * 9.81 / 2
            
        return left_force, right_force
    
    def _get_roll_pitch(self):
        """Get current roll and pitch angles"""
        torso_mat = self.data.xmat[self.torso_body.id].reshape(3, 3)
        z_axis = torso_mat[:, 2]
        pitch = np.arcsin(z_axis[0])
        roll = np.arcsin(z_axis[1])
        return roll, pitch
    
    def _is_foot_in_contact(self, foot):
        """Check if foot is in contact with ground"""
        # Simplified contact detection based on height
        if foot == 'left':
            foot_height = self.data.body('left_ankle_link').xpos[2]
        else:
            foot_height = self.data.body('right_ankle_link').xpos[2]
        
        return foot_height < 0.05  # 5cm threshold
    
    def _get_walking_obs(self):
        """Get walking-specific observations"""
        # Get base observations
        base_obs = self._get_obs()
        
        # Add phase information
        phase_obs = np.array([
            np.sin(2 * np.pi * self._phase / self._period),
            np.cos(2 * np.pi * self._phase / self._period)
        ])
        
        # Add target information
        t1, t2 = self._get_current_footstep_targets()
        root_pos = self.data.body('torso_link').xpos[:2]  # x, y position
        target_distance = np.linalg.norm(root_pos - (t1[:2] + t2[:2]) / 2)
        
        target_obs = np.array([target_distance])
        
        # Combine all observations
        full_obs = np.concatenate([base_obs, phase_obs, target_obs])
        return full_obs.astype(np.float32)
    
    def _get_walking_reward(self):
        """Calculate walking reward using LearningHumanoidWalking framework"""
        # Get current foot data
        left_foot_pos, right_foot_pos = self._get_foot_positions()
        left_foot_vel, right_foot_vel = self._get_foot_velocities()
        left_foot_force, right_foot_force = self._get_foot_forces()
        
        # Calculate step reward
        t1, t2 = self._get_current_footstep_targets()
        foot_positions = [left_foot_pos, right_foot_pos]
        root_pos = self.data.body('torso_link').xpos
        
        step_reward = calc_step_reward(
            foot_positions, t1, self._target_reached, root_pos, t2
        )
        
        # Calculate foot force/velocity rewards
        foot_frc_reward = calc_foot_frc_clock_reward(
            left_foot_force, right_foot_force,
            self.left_clock[0], self.right_clock[0], 
            self._phase, robot_mass=80
        )
        
        foot_vel_reward = calc_foot_vel_clock_reward(
            left_foot_vel, right_foot_vel,
            self.left_clock[1], self.right_clock[1],
            self._phase
        )
        
        # Height reward
        current_height = self.data.body('torso_link').xpos[2]
        height_reward = calc_height_reward(current_height, self._goal_height_ref, self._goal_speed_ref)
        
        # Orientation reward (keep upright)
        roll, pitch = self._get_roll_pitch()
        orient_reward = calc_orientation_reward(roll, pitch)
        
        # Combine rewards
        total_reward = (
            0.15 * foot_frc_reward +
            0.15 * foot_vel_reward +
            0.45 * step_reward +
            0.15 * height_reward +
            0.10 * orient_reward
        )
        
        return total_reward
    
    def reset(self, seed=None, options=None):
        """Reset environment for walking"""
        obs, info = super().reset(seed, options)
        
        # Reset walking task
        self._phase = np.random.choice([0, self._period // 2])  # Random phase start
        self._current_step_idx = 0
        self._current_plan_idx = np.random.randint(0, len(self.footstep_plans))
        self._target_reached = False
        
        # Reset foot tracking
        self.prev_left_foot_pos = np.array([0.0, 0.15, 0.0])
        self.prev_right_foot_pos = np.array([0.0, -0.15, 0.0])
        
        return self._get_walking_obs(), info
    
    def step(self, action):
        """Step environment with walking task"""
        # Execute the action
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Update phase
        self._phase = (self._phase + 1) % self._period
        
        # Check if target reached and update
        self._update_target_progress()
        
        # Get walking observations and reward
        walking_obs = self._get_walking_obs()
        walking_reward = self._get_walking_reward()
        
        return walking_obs, walking_reward, terminated, truncated, info
    
    def _update_target_progress(self):
        """Update target progress and switch targets"""
        t1, _ = self._get_current_footstep_targets()
        left_foot_pos, right_foot_pos = self._get_foot_positions()
        
        # Check if either foot reached the target
        min_dist = min([
            np.linalg.norm(left_foot_pos - t1),
            np.linalg.norm(right_foot_pos - t1)
        ])
        
        if min_dist < 0.1:  # 10cm threshold
            self._target_reached = True
            # Advance to next target
            if self._phase == (self._period // 2):  # Mid-cycle
                self._current_step_idx += 1
                current_plan = self.footstep_plans[self._current_plan_idx]
                if self._current_step_idx >= len(current_plan):
                    # Switch to next plan or restart
                    self._current_plan_idx = (self._current_plan_idx + 1) % len(self.footstep_plans)
                    self._current_step_idx = 0
        else:
            self._target_reached = False
