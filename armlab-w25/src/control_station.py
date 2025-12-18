#!/usr/bin/python
"""!
Main GUI for Arm lab
"""
import os, sys
script_path = os.path.dirname(os.path.realpath(__file__))

import argparse
import cv2
import numpy as np
import rclpy
import time
from functools import partial

from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QFileDialog

from resource.ui import Ui_MainWindow
from rxarm import RXArm, RXArmThread
from camera import Camera, VideoThread
from state_machine import StateMachine, StateMachineThread
""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class Gui(QMainWindow):
    """!
    Main GUI Class

    Contains the main function and interfaces between the GUI and functions.
    """
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        """ Groups of ui commonents """
        self.joint_readouts = [
            self.ui.rdoutBaseJC,
            self.ui.rdoutShoulderJC,
            self.ui.rdoutElbowJC,
            self.ui.rdoutWristAJC,
            self.ui.rdoutWristRJC,
        ]
        self.joint_slider_rdouts = [
            self.ui.rdoutBase,
            self.ui.rdoutShoulder,
            self.ui.rdoutElbow,
            self.ui.rdoutWristA,
            self.ui.rdoutWristR,
        ]
        self.joint_sliders = [
            self.ui.sldrBase,
            self.ui.sldrShoulder,
            self.ui.sldrElbow,
            self.ui.sldrWristA,
            self.ui.sldrWristR,
        ]
        """Objects Using Other Classes"""
        self.camera = Camera()
        print("Creating rx arm...")
        self.rxarm = RXArm()
        print("Done creating rx arm instance.")
        self.sm = StateMachine(self.rxarm, self.camera)
        """
        Attach Functions to Buttons & Sliders
        TODO: NAME AND CONNECT BUTTONS AS NEEDED
        """
        # Video
        self.ui.videoDisplay.setMouseTracking(True)
        self.ui.videoDisplay.mouseMoveEvent = self.trackMouse
        self.ui.videoDisplay.mousePressEvent = self.calibrateMousePress

        # Buttons
        # Handy lambda function falsethat can be used with Partial to only set the new state if the rxarm is initialized
        #nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state if self.rxarm.initialized else None)
        nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state)
        self.ui.btn_estop.clicked.connect(self.estop)
        self.ui.btn_init_arm.clicked.connect(self.initRxarm)
        self.ui.btn_torq_off.clicked.connect(
            lambda: self.rxarm.disable_torque())
        self.ui.btn_torq_on.clicked.connect(lambda: self.rxarm.enable_torque())
        self.ui.btn_sleep_arm.clicked.connect(lambda: self.rxarm.sleep())
        self.ui.btn_calibrate.clicked.connect(partial(nxt_if_arm_init, 'calibrate'))

        # User Buttons
        # TODO: Add more lines here to add more buttons
        # To make a button activate a state, copy the lines for btnUser3 but change 'execute' to whichever state you want
        self.ui.btnUser1.setText('Open Gripper')
        self.ui.btnUser1.clicked.connect(lambda: self.rxarm.gripper.release())
        self.ui.btnUser2.setText('Close Gripper')
        self.ui.btnUser2.clicked.connect(lambda: self.rxarm.gripper.grasp())
        self.ui.btnUser3.setText('Execute')
        self.ui.btnUser3.clicked.connect(partial(nxt_if_arm_init, 'execute'))

        # code written by us
        # this button will record the current position of the robot
        self.ui.btnUser4.setText('Record')
        self.ui.btnUser4.clicked.connect(partial(nxt_if_arm_init, 'record'))

        #this button will execute the points recorded by the record button
        self.ui.btnUser5.setText('Repeat')
        self.ui.btnUser5.clicked.connect(partial(nxt_if_arm_init, 'repeat'))
        # 

        #this button will execute grab state
        self.ui.btnUser6.setText('Grab')
        self.ui.btnUser6.clicked.connect(partial(nxt_if_arm_init, 'Grab'))
        # 

        #this button will execute Drop state
        self.ui.btnUser7.setText('Drop')
        self.ui.btnUser7.clicked.connect(partial(nxt_if_arm_init, 'Drop'))

        #Button for Sort and Stack competition
        self.ui.btnUser8.setText('Competition 1')
        self.ui.btnUser8.clicked.connect(partial(nxt_if_arm_init, 'Compete 1'))
        # 

        #Button for light em up competition
        self.ui.btnUser9.setText('Competition 2')
        self.ui.btnUser9.clicked.connect(partial(nxt_if_arm_init, 'Compete 2'))
        # 

        #Button for to the sky competition
        self.ui.btnUser10.setText('Competition 3')
        self.ui.btnUser10.clicked.connect(partial(nxt_if_arm_init, 'Compete 3'))
        # 

        #Button for Free throw competition
        self.ui.btnUser11.setText('Competition 4')
        self.ui.btnUser11.clicked.connect(partial(nxt_if_arm_init, 'Compete 4'))
        # 

        #Button for bonus event competition
        self.ui.btnUser12.setText('Competition 5')
        self.ui.btnUser12.clicked.connect(partial(nxt_if_arm_init, 'Compete 5'))
        # 


        # Sliders
        for sldr in self.joint_sliders:
            sldr.valueChanged.connect(self.sliderChange)
        self.ui.sldrMoveTime.valueChanged.connect(self.sliderChange)
        self.ui.sldrAccelTime.valueChanged.connect(self.sliderChange)
        # Direct Control
        self.ui.chk_directcontrol.stateChanged.connect(self.directControlChk)
        # Status
        self.ui.rdoutStatus.setText("Waiting for input")
        """initalize manual control off"""
        self.ui.SliderFrame.setEnabled(False)
        """Setup Threads"""

        # State machine
        self.StateMachineThread = StateMachineThread(self.sm)
        self.StateMachineThread.updateStatusMessage.connect(
            self.updateStatusMessage)
        self.StateMachineThread.start()
        self.VideoThread = VideoThread(self.camera)
        self.VideoThread.updateFrame.connect(self.setImage)
        self.VideoThread.start()
        self.ArmThread = RXArmThread(self.rxarm)
        self.ArmThread.updateJointReadout.connect(self.updateJointReadout)
        self.ArmThread.updateEndEffectorReadout.connect(
            self.updateEndEffectorReadout)
        self.ArmThread.start()

    """ Slots attach callback functions to signals emitted from threads"""

    @pyqtSlot(str)
    def updateStatusMessage(self, msg):
        self.ui.rdoutStatus.setText(msg)

    @pyqtSlot(list)
    def updateJointReadout(self, joints):
        for rdout, joint in zip(self.joint_readouts, joints):
            rdout.setText(str('%+.2f' % (joint * R2D)))

    # Distances should be in mm
    @pyqtSlot(list)
    def updateEndEffectorReadout(self, pos):
        self.ui.rdoutX.setText(str("%+.2f mm" % (pos[0])))
        self.ui.rdoutY.setText(str("%+.2f mm" % (pos[1])))
        self.ui.rdoutZ.setText(str("%+.2f mm" % (pos[2])))
        self.ui.rdoutPhi.setText(str("%+.2f rad" % (pos[3])))
        self.ui.rdoutTheta.setText(str("%+.2f rad" % (pos[4])))
        self.ui.rdoutPsi.setText(str("%+.2f rad" % (pos[5])))

    @pyqtSlot(QImage, QImage, QImage, QImage)
    def setImage(self, rgb_image, depth_image, tag_image, grid_image):
        """!
        @brief      Display the images from the camera.

        @param      rgb_image    The rgb image
        @param      depth_image  The depth image
        """
        if (self.ui.radioVideo.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(rgb_image))
        if (self.ui.radioDepth.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(depth_image))
        if (self.ui.radioUsr1.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(tag_image))
        if (self.ui.radioUsr2.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(grid_image))

    """ Other callback functions attached to GUI elements"""

    def estop(self):
        self.rxarm.disable_torque()
        self.sm.set_next_state('estop')

    def sliderChange(self):
        """!
        @brief Slider changed

        Function to change the slider labels when sliders are moved and to command the arm to the given position
        """
        for rdout, sldr in zip(self.joint_slider_rdouts, self.joint_sliders):
            rdout.setText(str(sldr.value()))

        self.ui.rdoutMoveTime.setText(
            str(self.ui.sldrMoveTime.value() / 10.0) + "s")
        self.ui.rdoutAccelTime.setText(
            str(self.ui.sldrAccelTime.value() / 20.0) + "s")
        self.rxarm.set_moving_time(self.ui.sldrMoveTime.value() / 10.0)
        self.rxarm.set_accel_time(self.ui.sldrAccelTime.value() / 20.0)

        # Do nothing if the rxarm is not initialized
        if self.rxarm.initialized:
            joint_positions = np.array(
                [sldr.value() * D2R for sldr in self.joint_sliders])
            # Only send the joints that the rxarm has
            self.rxarm.set_positions(joint_positions[0:self.rxarm.num_joints])

    def directControlChk(self, state):
        """!
        @brief      Changes to direct control mode

                    Will only work if the rxarm is initialized.

        @param      state  State of the checkbox
        """
        if state == Qt.Checked and self.rxarm.initialized:
            # Go to manual and enable sliders
            self.sm.set_next_state("manual")
            self.ui.SliderFrame.setEnabled(True)
        else:
            # Lock sliders and go to idle
            self.sm.set_next_state("idle")
            self.ui.SliderFrame.setEnabled(False)
            self.ui.chk_directcontrol.setChecked(False)

    def trackMouse(self, mouse_event):
        """!
        @brief      Show the mouse position in GUI

                    TODO: after implementing workspace calibration display the world coordinates the mouse points to in the RGB
                    video image.

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """

        # TODO: Modify this function to change the mouseover text.
        # You should make the mouseover text display the (x, y, z) coordinates of the pixel being hovered over

        pt = mouse_event.pos()
        if self.camera.DepthFrameRaw.any() != 0:
            if (self.camera.camera_calibrated):
                # in calibration mode
                # print("In Calibration Mode")
                # printing the mouse coordinates in pixels
                z = self.camera.DepthFrameRaw[pt.y()][pt.x()]
                self.ui.rdoutMousePixels.setText("(%.0f,%.0f,%.0f)" %
                                                (pt.x(), pt.y(), z))
                
                homography_pixels = np.array([[pt.x()],[pt.y()],[1]])
                image_pixel = self.camera.homography_to_pixel_cood(homography_pixels)
                
                z = self.camera.DepthFrameRaw[int(image_pixel[1])][int(image_pixel[0])]
                camera_cood = self.camera.image_cood_to_camera_cood(image_pixel)
                camera_cood = z*camera_cood
                camera_cood = np.row_stack((camera_cood,np.array([1])))

                world_cood = self.camera.camera_to_world_cood(camera_cood)
                world_cood[2] -=21*(real_world_cood[1]-406)/513
                self.ui.rdoutMouseWorld.setText("(%.0f,%.0f,%.0f)" %
                                                (real_world_cood[0], real_world_cood[1], real_world_cood[2]))

            else:
                z = self.camera.DepthFrameRaw[pt.y()][pt.x()]
                self.ui.rdoutMousePixels.setText("(%.0f,%.0f,%.0f)" %
                                                (pt.x(), pt.y(), z))
                
                pixels = np.array([[pt.x()],[pt.y()],[1]])
                #  U and V

                instrinic_matrix = np.array([[896.86, 0.000000, 660.52],
                                            [0.000000, 897.203, 381.419],
                                            [0,0,1]]) 
                # K matrix

                camera_cood = np.linalg.inv(instrinic_matrix)@pixels
                camera_cood = z*camera_cood
                # Xc Yc and Zc

                trans_x = np.array([[1,0,0,28],
                                    [0,1,0,0],
                                    [0,0,1,0],
                                    [0,0,0,1]])
                trans_y = np.array([[1,0,0,0],
                                    [0,1,0,120],
                                    [0,0,1,0],
                                    [0,0,0,1]])
                trans_z = np.array([[1,0,0,0],
                                    [0,1,0,0],
                                    [0,0,1,1030],
                                    [0,0,0,1]])
                rot_angle_rad = np.deg2rad(190)
                rot_x = np.array([[1,0,0,0],
                                [0,np.cos(rot_angle_rad),-np.sin(rot_angle_rad),0],
                                [0,np.sin(rot_angle_rad),np.cos(rot_angle_rad),0],
                                [0,0,0,1]])
                extrinsic_matrix = trans_x@trans_y@trans_z
                extrinsic_matrix = extrinsic_matrix@rot_x
                
                camera_cood_1 = np.zeros((4,1))
                camera_cood_1[:3,:] = camera_cood
                camera_cood_1[3,:] = 1

                real_world_cood = np.linalg.inv(extrinsic_matrix)@camera_cood_1

                self.ui.rdoutMouseWorld.setText("(%.0f,%.0f,%.0f)" %
                                                (real_world_cood[0], real_world_cood[1], real_world_cood[2]))

    def calibrateMousePress(self, mouse_event):
        """!
        @brief Record mouse click positions for calibration

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """
        """ Get mouse posiiton """


        pt = mouse_event.pos()
        self.camera.last_click[0] = pt.x()
        self.camera.last_click[1] = pt.y()

        z = self.camera.DepthFrameRaw[pt.y()][pt.x()]
        # # pt has weird format, so make a new array
        # click = np.array([pt.x(),pt.y(),z])
        self.camera.new_click = True
 

        # print ("PT",click)

    def initRxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.ui.SliderFrame.setEnabled(False)
        self.ui.chk_directcontrol.setChecked(False)
        self.rxarm.enable_torque()
        self.sm.set_next_state('initialize_rxarm')


### TODO: Add ability to parse POX config file as well
def main():
    """!
    @brief      Starts the GUI
    """
    app = QApplication(sys.argv)
    app_window = Gui()
    app_window.show()

    # Set thread priorities
    app_window.VideoThread.setPriority(QThread.HighPriority)
    app_window.ArmThread.setPriority(QThread.NormalPriority)
    app_window.StateMachineThread.setPriority(QThread.LowPriority)

    sys.exit(app.exec_())


# Run main if this file is being run directly
if __name__ == '__main__':
    main()
