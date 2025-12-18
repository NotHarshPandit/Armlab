import numpy as np
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
import os

# Path to your recorded bag file
name_of_bag= "joint_states"
bag_path = os.path.join(os.getcwd(), name_of_bag + ".bag")

# Initialize storage
time_stamps = []
joint_angles = []

# Read data from bag
with AnyReader([bag_path]) as reader:
    for topic, msg, t in reader.messages():
        if topic == "/rx200/joint_states":
            time_stamps.append(t.time)  # Extract timestamp
            joint_angles.append(msg.position)  # Extract joint positions

# Convert to numpy arrays
time_stamps = np.array(time_stamps)
joint_angles = np.array(joint_angles)

# Normalize time to start at zero
time_stamps = (time_stamps - time_stamps[0]) / 1e9  # Convert nanoseconds to seconds

# Plot joint angles over time
plt.figure(figsize=(10, 5))
for i in range(joint_angles.shape[1]):  # Loop over joints
    plt.plot(time_stamps, joint_angles[:, i], label=f'Joint {i+1}')
    
plt.xlabel("Time (s)")
plt.ylabel("Joint Angle (rad)")
plt.title("Joint Angles Over One Cycle")
plt.legend()
plt.grid()
plt.show()
