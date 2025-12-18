"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np

# expm is a matrix exponential function
from scipy.linalg import expm


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, links):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    # #np.arctan(200/50) (a, alpha, d, theta)
    dh_params = np.array([[0 , -np.pi/2, 103.91, np.pi/2],
                          [205.73, 0, 0, np.arctan(50/200) -np.pi/2], #beta
                          [200.0 ,0 , 0,  (np.pi/2) -np.arctan(50/200) ],  # gamma
                          [0, -np.pi/2, 0, -np.pi/2],
                          [0, 0, 174.15, 0.0]])
    # print(joint_angles)
    dh_params[:,3]+=joint_angles

    transformation = np.eye(4)
    for dh_param in dh_params:
        transformation = np.matmul(transformation,get_transform_from_dh(dh_param[0],dh_param[1],dh_param[2],dh_param[3]))
        # print(transformation[:3,3])
    return transformation
    



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
    # pass
    

def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T.
    """

    theta4 = np.arctan2(T[1,2],T[0,2])
    theta5 = np.arccos(T[2,2])
    theta6 = np.arctan2(T[2,1],-T[2,0])
    return np.array([theta4,theta5,theta6])

def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """
    pose_vector = []
    
    temp = T[:3,3].reshape(1,3)
    temp = temp.squeeze()
    # temp = list(temp)
    # pose_vector.extend(temp)
    angles = get_euler_angles_from_T(T)
    # pose_vector.extend(angles)
    pose_vector = [T[0,3],T[1,3],T[2,3],angles[0],angles[1],angles[2]]
    return pose_vector



def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """

    transform = m_mat

    for angle_index in range(len(joint_angles)-1, -1, -1):
        S_matrix = to_s_matrix(s_lst[:, angle_index])
        S_exp_matrix = expm(S_matrix * joint_angles[angle_index])
        transform = np.matmul(S_exp_matrix, transform)

    return transform
    pass

    
def to_w_matrix(w):
    ''' wmat = [0 -w3 w2
                w3 0 -w1
                -w2 w1 0]'''
    wmat = np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])
    return wmat

def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
   

    pass


def IK_geometric(dh_params, pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """

    
    l1 = 103.91                # from t1 to t2, aka base offset
    l2 = 205.73 # from t2 to t3, shoulder to elbow shortest distance
    l3 = 200                    # from t3 to t4, elbow to wrist
    l4 = 174.15     # from t4 to ee, center of gripper (?)
    t_offset = np.arctan2(50, 200) # offset angle bewteen t3 and t2

    theta1 = np.arctan2(-pose[0], pose[1])

    phi = pose[3]
    # l4_unit: orientation of l4 w.r.t. origion
    l4_unit = np.array([-np.sin(theta1)*np.cos(phi), np.cos(theta1)*np.cos(phi), -np.sin(phi)], dtype=np.float64)
    xc, yc, zc = pose[0:3] - l4*l4_unit # xyz of the wrist (t4)
    # print("xc, yc, zc; ", (xc,yc,zc))
    if np.sqrt(xc*xc + yc*yc + (zc - l1)*(zc - l1)) > (l2 + l3):
        # print("[KINEMATICS] Pose is unreachable! Cannot form a triangle.")
        return False, [0, 0, 0, 0, 0]

    r = np.sqrt(xc*xc + yc*yc)   # (r, s) are planar xy of the wrist 
    s = zc - l1
    
    # two cases for t3: t3 = t3; t3 = -t3
    theta3 = - np.arccos((r*r + s*s - l2*l2 - l3*l3)/(2*l2*l3))
    theta2 = np.arctan2(s, r) - np.arctan2(l3*np.sin(theta3), l2 + l3*np.cos(theta3)) # TODO something to do with offset
    
    # t3 = t3 + t_offset - np.pi/2 # offset
    theta3 += np.pi/2 -t_offset
    theta3 = -theta3
    theta2 = np.pi/2 - t_offset - theta2 # offset

    theta4 = phi - (theta2 + theta3) # by geometry

    # enforcing the joint limits
    if theta1 >= np.pi or theta1 <= -np.pi:
        return False, [0, 0, 0, 0, 0]

    if theta2 >= np.deg2rad(113) or theta2 <= -np.deg2rad(108):
        return False, [0, 0, 0, 0, 0]

    if theta3 >= np.deg2rad(93) or theta3 <= -np.deg2rad(108):
        return False, [0, 0, 0, 0, 0]

    if theta4 >= np.deg2rad(123) or theta4 <= -np.deg2rad(100):
        return False, [0, 0, 0, 0, 0]

    return True,np.array([theta1,theta2,theta3,theta4])


