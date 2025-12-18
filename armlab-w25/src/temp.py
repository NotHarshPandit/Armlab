import kinematics
import numpy as np

joint_angles = [0,0,np.pi/2,0,0]

dh_params = np.array([[0 , -np.pi/2, 103.91, np.pi/2],
                          [205.73, 0, 0, np.arctan(50/200) -np.pi/2], #beta
                          [200.0 ,0 , 0,  (np.pi/2) -np.arctan(50/200) ],  # gamma
                          [0, -np.pi/2, 0, -np.pi/2],
                          [0, 0, 174.15, 0.0]])
links = None

transform = kinematics.FK_dh(dh_params, joint_angles, links)
print("x, y, z: ")
print(transform[:3,3])

joint_angles = [0,0,np.pi/2,0]
joint_angles_sum = np.array(joint_angles)
joint_angles_sum = joint_angles_sum.sum() - joint_angles[0]


pose = np.array([transform[0][3],transform[1][3],transform[2][3],joint_angles_sum])
inv_present,ik_vals = kinematics.IK_geometric(dh_params,pose)
if inv_present:
    print("Joint angles expected")
    print(joint_angles)
    print("Calculated Joint Angles")
    print(ik_vals)