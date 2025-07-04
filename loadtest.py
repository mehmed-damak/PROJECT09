import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the model from your scene.xml (which includes h1.xml)
model = mujoco.MjModel.from_xml_path("assets/h1/scene.xml")
data = mujoco.MjData(model)

# Set initial joint positions (edit these values as needed)
# You can get the order and meaning of qpos from model.joint_names or model.jnt_qposadr
# Example: set left hip pitch, left knee, left ankle, right hip pitch, right knee, right ankle
# (You can print model.joint_names to see the order)
data.qpos[:] = 0  # Start with all zeros

# Example: set left hip pitch to 0.3, left knee to -0.6, left ankle to -0.3
# (You must use the correct indices for your robot)
# You can print(model.joint_names) and print(model.jnt_qposadr) to find the mapping
data.qpos[model.joint('left_hip_pitch').qposadr] = -.2
data.qpos[model.joint('left_knee').qposadr] = 0.5
data.qpos[model.joint('left_ankle').qposadr] = -.3
data.qpos[model.joint('right_hip_pitch').qposadr] = -.2
data.qpos[model.joint('right_knee').qposadr] = .5
data.qpos[model.joint('right_ankle').qposadr] = -.3
data.qpos[model.joint('torso').qposadr] = 0.0

# Forward the simulation to update the state
mujoco.mj_forward(model, data)

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer running. Close the window to exit.")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.005)