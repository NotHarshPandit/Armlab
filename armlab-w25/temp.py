# import torch 
import numpy as np

def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """
    transformation = np.array([[np.cos(theta) , -np.sin(theta)*np.cos(alpha) , np.sin(theta)*np.sin(alpha) , a*np.cos(theta)],
                               [np.sin(theta) , np.cos(theta)*np.cos(alpha) , -np.cos(theta)*np.sin(alpha) , a*np.sin(theta)],
                               [0 , np.sin(alpha) , np.cos(alpha) , d],
                               [0, 0, 0, 1]])

    return transformation

dh_params = np.array([[0 , -np.pi/2, 103.91, np.pi/2],
                          [205.73, 0, 0, np.arctan(50/200) -np.pi/2], #beta
                          [200.0 ,0 , 0,  (np.pi/2) -np.arctan(50/200) ],  # gamma
                          [0, -np.pi/2, 0, -np.pi/2],
                          [0, 0, 174.15, 0.0]])

joint_angles = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2])
dh_params[:,3] += joint_angles  # Direct addition, no need for reshaping
transformation = np.eye(4)
for dh_param in dh_params:
    transformation = np.matmul(transformation,get_transform_from_dh(dh_param[0],dh_param[1],dh_param[2],dh_param[3]))
print("DH tranform Original")
print(transformation)

# Harsh DH params
dh_params = np.array([[0 , -np.pi/2, 103.91, np.pi/2],
                          [205.73, 0, 0, np.arctan(50/200) -np.pi/2 ], #beta
                          [200.0 ,0 , 0,  (np.pi/2) - np.arctan(50/200) ],  # gamma
                          [0, np.pi/2, 0, np.pi/2], 
                          [0, 0, 174.15, 0.0]])

                

joint_angles = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2])
dh_params[:,3] += joint_angles  # Direct addition, no need for reshaping
transformation = np.eye(4)
for dh_param in dh_params:
    transformation = np.matmul(transformation,get_transform_from_dh(dh_param[0],dh_param[1],dh_param[2],dh_param[3]))
print("DH tranform Harsh")
print(transformation)

