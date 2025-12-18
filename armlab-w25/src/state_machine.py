"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
import cv2
import kinematics

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,          0.0,        0.0],
            [0.75*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.5*-np.pi/2,   -0.5,      -0.3,      np.pi/2,        0.0],
            [0.25*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.0,             0.0,       0.0,          0.0,        0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [0.5*np.pi/2,     0.5,       0.3,     -np.pi/3,        0.0],
            [0.75*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [np.pi/2,         0.5,       0.3,     -np.pi/3,        0.0],
            [0.0,             0.0,       0.0,          0.0,        0.0]]
        
        # 
        self.teach = []
        # 

        self.end_turn = 0
        self.end_orentation_grab = np.pi/2
        self.end_orentation_drop = np.pi/2
        self.block_size = 'Large'
        

    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and functions as needed.
        """

        # IMPORTANT: This function runs in a loop. If you make a new state, it will be run every iteration.
        #            The function (and the state functions within) will continuously be called until the state changes.

        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()
        
        # new state 
        if self.next_state == "record":
            self.record()
        if self.next_state == "repeat":
            self.repeat()

        if self.next_state == "Grab":
            self.click_to_grab()

        if self.next_state == "Drop":
            self.click_to_drop()

        if self.next_state == "Compete 1":
            self.compete1()

        if self.next_state == "Compete 2":
            self.compete2()

        if self.next_state == "Compete 3":
            self.compete3()

        if self.next_state == "Compete 4":
            self.compete4()

        if self.next_state == "Compete 5":
            self.compete5()
        # 



    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        
        self.status_message = "State: Execute - Executing motion plan"

        # 
        for positions in self.waypoints:
            self.rxarm.set_positions(positions)
            time.sleep(3)

        # 
        self.next_state = "idle"
    
    # 
    def record(self):
        self.status_message = "State: Record - This position is being recoded"
        current_position = self.rxarm.get_positions()
        self.teach.append(current_position)
        self.status_message = "State: Record - Position Recorded"
        time.sleep(1)
        self.next_state = "idle"
 
    def repeat(self):
        self.status_message = "State: Repeat - Executing motion plan"
    
        #taking the robot arm to home position
        self.rxarm.initialize()
        time.sleep(3)

        #opening the gripper
        self.rxarm.gripper.release()
        gripper_open = True #gripper open

        counter = 1 #to count number of moves

        for positions in self.teach:
            # got to the position
            self.rxarm.set_positions(positions)
            counter+=1 #increase the number of moves
            time.sleep(3)
            if (counter%3 == 0): #after every 3 moves, open or close gripper
                if (gripper_open):
                    self.rxarm.gripper.grasp()
                    time.sleep(3)
                    gripper_open = False
                else:
                    self.rxarm.gripper.release()
                    time.sleep(3)
                    gripper_open = True
            

        # Arm go to sleep position after launch
        self.rxarm.sleep()
        time.sleep(3)

        self.next_state = "idle"

        # 

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.camera.camera_calibrated = True
        

        """TODO Perform camera calibration routine here"""
        self.status_message = "Calibration-started"
        # K matrix
        
        K = np.array([[896.86, 0.000000, 660.52],
                      [0.000000, 897.203, 381.419],
                      [0,0,1]])
        self.camera.intrinsic = K
        
        d = np.array([
        0.1452116072177887,
        -0.4892308712005615,
        -0.0012472006492316723,
        -0.0003476899000816047,
         0.45278146862983704])
        
        # initializing an empty array to store the center points of April Tag
        center_points_image_frame = np.empty((4,2),dtype = np.float64)
        # msg = self.camera.tag_detections
        center_points_counter = 0
        # for center_points in self.camera.tag_detections.detections:
        for center_points in self.camera.tag_detections.detections:
            center_points_image_frame[center_points.id-1] = np.array([center_points.centre.x,center_points.centre.y])

    
        # center points of April Tags in pixel frame
        center_points_April_tag_pixel_frame = np.array(center_points_image_frame,dtype = np.float64)
        
        # center points of April Tags in world coordinate frame
        center_points_April_tag_world_frame = np.array([[-250.0,-25.0,0.0],
                                [250.0,-25.0,0.0],
                                [250.0,275.0,0.0],
                                [-250.0,275.0,0.0]],dtype = np.float64) 
           
        # using solve PnP method to find the R and t matrix
        [_, R_exp, t] = cv2.solvePnP(center_points_April_tag_world_frame,
                                 center_points_April_tag_pixel_frame,
                                 K,
                                 d,
                                flags=cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(R_exp)
        # storing the extrinsic matrix
        self.camera.extrinsic_matrix =  np.row_stack((np.column_stack((R, t)), (0, 0, 0, 1)))

        # //////////////////////////////////////////////////////////////////////////////////////////
        # Implementing the recover homogenous affine transformation part here
        p = center_points_April_tag_pixel_frame[0:3,:]
        p_prime = center_points_April_tag_world_frame[0:3,:]
        # construct intermediate matrix
        Q = p[1:] - p[0]
        Q_prime = p_prime[1:] - p_prime[0]

        # calculate rotation matrix
        R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
                np.row_stack((Q_prime, np.cross(*Q_prime))))

        # calculate translation vector
        t = p_prime[0] - np.dot(p[0], R)

        # calculate affine transformation matrix
        extrinsinc_matrix_homogenous_affine_tranformation =  np.transpose(np.column_stack((np.row_stack((R, t)), (0, 0, 0, 1))))
        print("Homogenous Affine Transformation Matrix:")
        print(extrinsinc_matrix_homogenous_affine_tranformation)
        print("Extrinsic Matrix by solvePnP:")
        print(self.camera.extrinsic_matrix)
        # code for recover homogenous affine transformation ends here
        # //////////////////////////////////////////////////////////////////////////////////////////


        self.status_message = "Calibration - Completed Calibration"
        # time.sleep(1)
        self.status_message = "Starting Homography"

        April_Tag_world_to_pixel_points = np.array([[370,565],    # Top left corner
                              [915,565],   # Top right
                              [915,265],    # Bottom right
                              [370,265]])    # Botom left


        H = cv2.findHomography(center_points_image_frame, April_Tag_world_to_pixel_points)[0]
        #self.camera.homography = np.linalg.inv(H)
        self.camera.homography = H

        # extrinsic matrix using recover homogenous transform
        p = center_points_April_tag_world_frame[0:3,:]
        p_prime = np.column_stack((April_Tag_world_to_pixel_points,np.zeros((4,1))))
        p_prime = p_prime[0:3,:]
        Q = p[1:] - p[0]
        Q_prime = p_prime[1:] - p_prime[0]

        # calculate rotation matrix
        R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
                np.row_stack((Q_prime, np.cross(*Q_prime))))

        # calculate translation vector
        t = p_prime[0] - np.dot(p[0], R)

        # calculate affine transformation matrix
        # H_temp = np.transpose(np.column_stack((np.row_stack((R, t)), (0, 0, 0, 1))))
        # self.camera.homography = H_temp
        # print("Homogenous tranform matrix using homogenours affine transform: ")
        # print(H_temp)
        # print("H original")
        # print(H)


        ####################NEXT STEPS : FIND THE CALCULATION AS IN THE BOARD
    
        print("Ending Homography")

        self.next_state = "idle"

    # Grab
    def click_to_grab(self):
        self.status_message = "State: Click to grab"
        # self.current_state = "Grab"

        # self.camera.last_click[0] = pt.x()
        # self.camera.last_click[1] = pt.y()
        
        #  u and v cordinates in mouse co ordinates
        u = self.camera.last_click[0]
        v = self.camera.last_click[1]
        real_world_cood = np.array([[u],[v],[1]])
        
        if (self.camera.camera_calibrated):
            # if we are in homography mode 
            # [u*lambda v*lambda lambda]
            real_world_cood = np.matmul(np.linalg.inv(self.camera.homography),real_world_cood)
            # [u/lambda v/lambda 1]
            real_world_cood = real_world_cood/real_world_cood[2]

        # getting the depth value using u and v values 
        z = self.camera.DepthFrameRaw[int(real_world_cood[1])][int(real_world_cood[0])]

        # from image to camera frame
        camera_cood = np.matmul(np.linalg.inv(self.camera.intrinsic_matrix),real_world_cood)
        camera_cood = z*camera_cood

        # adding a pdding of 1
        camera_cood = np.row_stack((camera_cood,np.array([1])))

        # from camera to world frame
        real_world_cood = np.matmul(np.linalg.inv(self.camera.extrinsic_matrix),camera_cood)
                
        real_world_cood[2] -=21*(real_world_cood[1]-406)/513
        # now real world cood has x,y,z values

        end_effector_orientation = self.end_orentation_grab
        real_world_cood[3] = end_effector_orientation

        # buffer for z co ordinate
        print("XYZ of the point: ",real_world_cood[0]," , ",real_world_cood[1]," , ",real_world_cood[2])
        real_world_cood[2]+=50
        print("XYZ 50 above point: ",real_world_cood[0]," , ",real_world_cood[1]," , ",real_world_cood[2])
        inverse_present,joint_angles = kinematics.IK_geometric(None,real_world_cood)
        self.rxarm.gripper.release()
        gripper_open = True
        if inverse_present:
            joint_angles = np.column_stack((joint_angles.reshape(1,4),self.end_turn))
            # joint_angles = np.rad2deg(joint_angles)
            joint_angles = joint_angles.flatten().tolist()
            # print(joint_angles.shape)
            self.rxarm.set_positions(joint_angles)
            time.sleep(self.rxarm.moving_time+0.2)
        else:
            self.end_orentation_grab = self.end_orentation_grab/2
            self.end_turn = 0
            self.click_to_grab()
            return 
        real_world_cood[2]-= 60
        print("XYZ before picking: ",real_world_cood[0]," , ",real_world_cood[1]," , ",real_world_cood[2])
        # if self.block_size == 'Large':
        #     real_world_cood[2] = 15+(((real_world_cood[2]-50)//41))*40
        # if self.block_size == 'Small':
        #     real_world_cood[2] = 15+(((real_world_cood[2]-50)//26))*25
        inverse_present,joint_angles = kinematics.IK_geometric(None,real_world_cood)
        if inverse_present:
            joint_angles = np.column_stack((joint_angles.reshape(1,4),float(self.end_turn)))
            # joint_angles = np.rad2deg(joint_angles)
            joint_angles = joint_angles.flatten().tolist()
            # print(joint_angles.shape)
            self.rxarm.set_positions(joint_angles)
            time.sleep(self.rxarm.moving_time+0.2)
        else:
            print("Grab Pose Unreachable2")
        if gripper_open:
            self.rxarm.gripper.grasp()
            gripper_open = False
            # 5s to 2s
            time.sleep(self.rxarm.moving_time+0.2)
        print("XYZ after picking: ",real_world_cood[0]," , ",real_world_cood[1]," , ",real_world_cood[2]) 
        real_world_cood[2]+=50
        inverse_present,joint_angles = kinematics.IK_geometric(None,real_world_cood)
        if inverse_present:
            joint_angles = np.column_stack((joint_angles.reshape(1,4),float(self.end_turn)))
            # joint_angles = np.rad2deg(joint_angles)
            joint_angles = joint_angles.flatten().tolist()
            # print(joint_angles.shape)
            self.rxarm.set_positions(joint_angles)
            # 5s to 2s
            time.sleep(self.rxarm.moving_time+0.2)
        else:
            print("Grab Pose Unreachable3")
        print("XYZ after going up: ",real_world_cood[0]," , ",real_world_cood[1]," , ",real_world_cood[2])
        self.next_state = "idle"

    def click_to_drop(self):
        self.status_message = "State: Click to drop"
        u = self.camera.last_click[0]
        v = self.camera.last_click[1]
        real_world_cood = np.array([[u],[v],[1]])
        
        if (self.camera.camera_calibrated):
            # if we are in homography mode 
            # [u*lambda v*lambda lambda]
            real_world_cood = np.matmul(np.linalg.inv(self.camera.homography),real_world_cood)
            # [u/lambda v/lambda 1]
            real_world_cood = real_world_cood/real_world_cood[2]

        # getting the depth value using u and v values 
        z = self.camera.DepthFrameRaw[int(real_world_cood[1])][int(real_world_cood[0])]+20

        # from image to camera frame
        camera_cood = np.matmul(np.linalg.inv(self.camera.intrinsic_matrix),real_world_cood)
        camera_cood = z*camera_cood

        # adding a pdding of 1
        camera_cood = np.row_stack((camera_cood,np.array([1])))

        # from camera to world frame
        real_world_cood = np.matmul(np.linalg.inv(self.camera.extrinsic_matrix),camera_cood)
                
        real_world_cood[2] -=21*(real_world_cood[1]-406)/513
        # now real world cood has x,y,z values

        # buffer for z co ordinate
        real_world_cood[2]+= 100
        print("XYZ above the drop point: ",real_world_cood[0]," , ",real_world_cood[1]," , ",real_world_cood[2])
        end_effector_orientation = self.end_orentation_drop
        real_world_cood[3] = end_effector_orientation
        inverse_present,joint_angles = kinematics.IK_geometric(None,real_world_cood)

        self.rxarm.gripper.grasp()
        gripper_close = True
        if inverse_present:
            joint_angles = np.column_stack((joint_angles.reshape(1,4),0))
            # joint_angles = np.rad2deg(joint_angles)
            joint_angles = joint_angles.flatten().tolist()
            # print(joint_angles.shape)
            self.rxarm.set_positions(joint_angles)
            time.sleep(self.rxarm.moving_time+0.2)
        else:
            self.end_orentation_drop = self.end_orentation_drop/2
            self.click_to_drop()
            return
        real_world_cood[2]-= 60
        print("XYZ at the drop point: ",real_world_cood[0]," , ",real_world_cood[1]," , ",real_world_cood[2])
        # if self.block_size == 'Large':
        #     real_world_cood[2] = 15+abs((((real_world_cood[2]-50)//41)))*40
        # if self.block_size == 'Small':
        #     real_world_cood[2] = 15+abs((((real_world_cood[2]-50)//26)))*25
        inverse_present,joint_angles = kinematics.IK_geometric(None,real_world_cood)
        if inverse_present:
            joint_angles = np.column_stack((joint_angles.reshape(1,4),0))
            # joint_angles = np.rad2deg(joint_angles)
            joint_angles = joint_angles.flatten().tolist()
            # print(joint_angles.shape)
            self.rxarm.set_positions(joint_angles)
            time.sleep(self.rxarm.moving_time+0.2)
        else:
            print("Drop Pose Unreachable")
        if gripper_close:
            self.rxarm.gripper.release()
            gripper_close = False
            time.sleep(self.rxarm.moving_time+0.2)
        real_world_cood[2]+= 80
        print("XYZ above the drop point: ",real_world_cood[0]," , ",real_world_cood[1]," , ",real_world_cood[2])
        real_world_cood[3] = end_effector_orientation
        inverse_present,joint_angles = kinematics.IK_geometric(None,real_world_cood)
        if inverse_present:
            joint_angles = np.column_stack((joint_angles.reshape(1,4),0))
            # joint_angles = np.rad2deg(joint_angles)
            joint_angles = joint_angles.flatten().tolist()
            # print(joint_angles.shape)
            self.rxarm.set_positions(joint_angles)
            time.sleep(self.rxarm.moving_time+0.2)
        else:
            print("Drop Pose Unreachable")
        self.status_message = "State: successfully drop"
        self.next_state = "idle"

    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        time.sleep(1)

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            time.sleep(5)
        self.next_state = "idle"

    def sweeper(self):
        # blah = self.rxarm.get_positions()
        # print(blah)
        
        self.rxarm.initialize()
        self.rxarm.moving_time = 1.5
        self.rxarm.accel_time = 0.5
        cord_sweep = [[380,-100,55,-0],[330,350,60,0],[-330,350,65,0],[-380,-100,70,0]]
        
        for position in cord_sweep:
            inverse_present,joint_angles = kinematics.IK_geometric(None,position)
            joint_angles = np.column_stack((joint_angles.reshape(1,4),0))
            # joint_angles = np.rad2deg(joint_angles)
            joint_angles = joint_angles.flatten().tolist()
            self.rxarm.set_positions(joint_angles)
            time.sleep(1.5)

    def compete1(self):
        self.status_message = "State: Compete - Entering Sort and Stack Mode"
        colors = ['red','orange','yellow','green','blue','violet']
        if(not self.camera.camera_calibrated):
            self.calibrate()
        # Detecting the ball based on shape and color (circle and orange)
        small_drop_off = [[350,600],[350,600],[350,600]]
        large_drop_off = [[850,600],[850,600],[850,600]]
        self.rxarm.initialize()

        self.rxarm.moving_time = 1
        self.rxarm.accel_time = 0.3

        prev_color = colors[0]
        current_color = colors[0]

        # blocks = self.camera.block_detections:
        for color in colors:
            large_block_count = 0
            small_block_count = 0
            # count for each block of a color

            current_color = color
            for block in self.camera.block_detections:
                    if prev_color != current_color:
                    if color == block[0]:
                        self.camera.last_click[0] = block[2]
                        self.camera.last_click[1] = block[3]
                        self.end_turn = float(block[4]) + np.deg2rad(10)
                        print("Block orientation:",np.rad2deg(self.end_turn))

                        self.click_to_grab()
                        if block[1]=="Small":
                            self.camera.last_click[0] = small_drop_off[small_block_count][0]
                            self.camera.last_click[1] = small_drop_off[small_block_count][1]
                            self.block_size = "Small"
                            self.click_to_drop()
                            
                        if block[1]=="Large":
                            self.camera.last_click[0] = large_drop_off[large_block_count][0]
                            self.camera.last_click[1] = large_drop_off[large_block_count][1]
                            self.block_size = "Large"
                            self.click_to_drop()
                            
                        self.end_orentation_grab = np.pi/2
                        self.end_orentation_drop = np.pi/2
            prev_color = color
        self.rxarm.sleep()

        self.next_state = "idle"
    
    def place_the_block(self,block,x_point,y_point):
            self.camera.last_click[0] = block[2]
            self.camera.last_click[1] = block [3]
            self.end_turn = block[4]
            self.click_to_grab()
            self.camera.last_click[0] = x_point
            self.camera.last_click[1] = y_point
            self.click_to_drop()

    def compete2(self):
        self.status_message = "State: Compete - Entering light em up Mode"
        if(not self.camera.camera_calibrated):
            self.calibrate()

        # all the possible block colors
        colors = ['red','orange','yellow','green','blue','violet'] 

        # Distance between each placement of blocks
        large_block_length = 45
        small_block_length = 40

        # drop off point for blocks
        large_block_drop_point = [750,530]
        small_block_drop_point = [530,530]

        #Starting coordinates for blocks
        large_block_pt_x = large_block_drop_point[0]
        large_block_pt_y = large_block_drop_point[1]

        small_block_pt_x = small_block_drop_point[0]
        small_block_pt_y = small_block_drop_point[1]
        
        #Remove all stacks of blocks
        self.sweeper()

        # storing info about detected blocks
        detected_blocks = self.camera.block_detections

        #initializing arm
        self.rxarm.initialize()

        #Changing speed between change of positions
        self.rxarm.moving_time = 1
        self.rxarm.accel_time = 0.3
    
        for color in colors:
            for block in detected_blocks:
                block_color = block[0]
                block_shape = block[1]

                if block_shape == 'Large' and block_color == color:
                    self.block_size = "Large"
                    self.place_the_block(block,large_block_pt_x,large_block_pt_y)
                    large_block_pt_x += large_block_length
                if block_shape == 'Small' and block_color == color:
                    self.block_size = "Small"
                    self.place_the_block(block,small_block_pt_x,small_block_pt_y)
                    small_block_pt_x -= small_block_length
                
                #restarting orentation as straight down
                self.end_orentation_grab = np.pi/2
                self.end_orentation_drop = np.pi/2

        #ending arm in sleep position
        self.rxarm.sleep()  
        self.next_state = "idle"

    def move_over(self,position):
        joint_angles = position
        dh_params = np.array([[0 , -np.pi/2, 103.91, np.pi/2],
                          [205.73, 0, 0, np.arctan(50/200) -np.pi/2], #beta
                          [200.0 ,0 , 0,  (np.pi/2) -np.arctan(50/200) ],  # gamma
                          [0, -np.pi/2, 0, -np.pi/2],
                          [0, 0, 174.15, 0.0]])
        links = None

        transform = kinematics.FK_dh(dh_params, joint_angles, links)

        transform[1][3] -= 100
        pose = np.array([transform[0][3],transform[1][3],transform[2][3],self.end_orentation_drop])
        inv_present,ik_vals = kinematics.IK_geometric(None,pose)
        ik_vals = np.column_stack((ik_vals.reshape(1,4),0))
        
        ik_vals = ik_vals.flatten().tolist()
        
        return ik_vals    
    

    def compete3(self):
        self.status_message = "State: Compete - Entering to the sky Mode"
        if(not self.camera.camera_calibrated):
            self.calibrate()
        
        drop_off = np.array([640,315])
        self.rxarm.initialize()

        self.rxarm.moving_time = 2
        self.rxarm.accel_time = 0.5

        # Put blocks at
        # [250,-125],[200,-75],[200,25],[300,25],[300,-75],[425,50],[425,150],[425,250]

        # blocks = self.camera.block_detections:
        for block in self.camera.block_detections:
            self.block_size = "Large"
            self.camera.last_click[0] = block[2]
            self.camera.last_click[1] = block[3]
            self.end_turn = block[4]
            self.click_to_grab()
            self.camera.last_click[0] = drop_off[0]
            self.camera.last_click[1] = drop_off[1]
            self.click_to_drop()
            current_position = self.rxarm.get_positions()
            next = self.move_over(current_position)
            self.rxarm.set_positions(next)
            time.sleep(self.rxarm.moving_time+0.2)

            self.end_orentation_grab = np.pi/2
            self.end_orentation_drop = np.pi/2
        self.rxarm.arm.go_to_sleep_pose()

        self.next_state = "idle"
    
    def compete4(self):
        self.status_message = "State: Compete - Entering Free throw Mode"
        time.sleep(1)
        self.calibrate()
        time.sleep(3)
        # Detecting the ball based on shape and color (circle and orange)
        #if (self.camera.blockDetector() =="red"):
        #self.rxarm.initialize()
        time.sleep(3)
        # Implement medthod here
        #self.rxarm.initialize()
        time.sleep(3)
        self.next_state = "idle"
    
    def compete5(self):
        self.status_message = "State: Compete - Entering Bonus event Mode"
        time.sleep(1)
        self.calibrate()
        time.sleep(3)
        # Detecting the ball based on shape and color (circle and orange)
        #if (self.camera.blockDetector() =="red"):
        #self.rxarm.initialize()
        time.sleep(3)
        # Implement medthod here
        #self.rxarm.initialize()
        time.sleep(3)
        self.next_state = "idle"





class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)