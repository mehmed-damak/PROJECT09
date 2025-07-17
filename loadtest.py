import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the model from your scene.xml (which includes h1.xml)
model = mujoco.MjModel.from_xml_path("assets/h1/scene.xml")
data = mujoco.MjData(model)

# Set initial joint positions based on the "home" keyframe from scene.xml
# First 7 elements are for the free joint (base position and orientation)
data.qpos[0:3] = [0, 0, 1]  # x, y, z position - elevate robot above ground
data.qpos[3:7] = [1, 0, 0, 0]  # quaternion orientation (w, x, y, z) - identity quaternion

# Set joint positions to match the environment's initial pose
# Hip yaw and roll joints (set to match XML keyframe values)
data.qpos[model.joint('left_hip_yaw').qposadr] = 0.0
data.qpos[model.joint('left_hip_roll').qposadr] = 0.0
data.qpos[model.joint('right_hip_yaw').qposadr] = 0.0
data.qpos[model.joint('right_hip_roll').qposadr] = 0.0

# Hip pitch, knee, and ankle positions for standing pose (from XML keyframe)
data.qpos[model.joint('left_hip_pitch').qposadr] = -0.4
data.qpos[model.joint('left_knee').qposadr] = 0.8
data.qpos[model.joint('left_ankle').qposadr] = -0.4
data.qpos[model.joint('right_hip_pitch').qposadr] = -0.4
data.qpos[model.joint('right_knee').qposadr] = 0.8
data.qpos[model.joint('right_ankle').qposadr] = -0.4

# Torso position
data.qpos[model.joint('torso').qposadr] = 0.0

# Forward the simulation to update the state
mujoco.mj_forward(model, data)

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer running. Close the window to exit.")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.0002)