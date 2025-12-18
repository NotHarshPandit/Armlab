#!/usr/bin/env python3

"""!
Class to represent the camera.
"""
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import time
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError

# added this
import argparse
import sys
import math

class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720,1280, 3)).astype(np.uint8)


        # mouse clicks & calibration variables
        self.camera_calibrated = False
        self.intrinsic_matrix = np.eye(3)
        self.extrinsic_matrix = np.eye(4)
        self.last_click = np.array([0, 0]) # This contains the last clicked position
        self.new_click = False # This is automatically set to True whenever a click is received. Set it to False yourself after processing a click
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275], [-250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])

        """to store the mouse co-ordinates"""
        self.click_detections = np.array([])

        # to store the homography matrix
        self.homography = np.eye(3)

    # function to convert from world coordinates to pixel coordinates:
    def world_to_pixel(self,point):
        if point.shape == (3,):
            point = point.reshape(3,1)
        point = np.row_stack((point,np.array([1])))

        camera_point = np.matmul(self.extrinsic_matrix,point)
        camera_point = camera_point[:3,:]

        pixel_point = np.matmul(self.intrinsic_matrix,camera_point)
        pixel_point = pixel_point/pixel_point[2]
        # u and v values
        pixel_point = pixel_point[:2,:] 
        return pixel_point

    # function to change from pixel coordinates to homography coordinates
    def pixel_to_homography(self,point):
        point = np.row_stack((point,np.array([1])))
        # print(self.homography.shape)
        # print(point.shape)
        homography_points = np.matmul(self.homography,point)
        homography_points = homography_points/homography_points[2]
        return homography_points
    
    def homography_to_pixel_cood(self,homography_pixels):
        # homography pixels are of the form [u v 1]
        pixel_cood = np.matmul(np.linalg.inv(self.homography),homography_pixels)
        # now they are of the form [u v lambda]
        pixel_cood = pixel_cood/pixel_cood[2]
        # now they are of the form [u/lambda v/lambda 1]
        return pixel_cood
    
    def  image_cood_to_camera_cood(self,image_pixel):
        return np.matmul(np.linalg.inv(self.intrinsic_matrix),image_pixel)

    def camera_to_world_cood(self,camera_cood):
        return np.matmul(np.linalg.inv(self.camera.extrinsic_matrix),camera_cood)


    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    def blockDetector(self, cv_image):
    # def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        # set the font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # python3 control_station.py -i self.camera.VideoFrame.copy() -d /path/to/depth_map.png -l 0 -u 255


        # list the colors
        colors = list((
            {'id': 'red', 'color': (127,19,30)},
                {'id': 'orange', 'color': (164,66,5)},
                {'id': 'yellow', 'color': (218,180,30)},
                {'id': 'green', 'color': (30,110,60)},
                {'id': 'blue', 'color': (5,60,110)},
                {'id': 'violet', 'color': (50,50,73)})
        )


        def retrieve_area_color(data, contour, labels):
            mask = np.zeros(data.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean = cv2.mean(data, mask=mask)[:3]
            min_dist = (np.inf, None)
            for label in labels:
                d = np.linalg.norm(label["color"] - np.array(mean))
                if d < min_dist[0]:
                    min_dist = (d, label["id"])
            return min_dist[1]

        # Copy image for drawing contour
        cnt_image = cv_image.copy()
        # cnt_image = self.VideoFrame.copy()
        depth_data = self.DepthFrameRaw.copy()
        rgb_image = self.VideoFrame.copy()

        #Prepare instrinsic and depth data
        K = self.intrinsic_matrix
        h, w = rgb_image.shape[:2]
        u = np.repeat(np.arange(w)[None, :], h, axis=0)
        v = np.repeat(np.arange(h)[:, None], w, axis=1)
        Z = depth_data #Raw depth data

        # Use extrinsic matrix for transformating depth data to world coordinates
        T_i = self.extrinsic_matrix
        T_f = np.array([
                        [1, 0,  0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 1000],   
                        [0, 0,  0, 1]])
        
        T_relative = np.dot(T_f, np.linalg.inv(T_i))


        # 3D transformation from image space to camera space
        X = (u - K[0,2]) * Z / K[0,0]
        Y = (v - K[1,2]) * Z / K[1,1]
        points_camera_frame = np.stack((X, Y, Z, np.ones_like(Z)), axis=-1)
        points_transformed = np.dot(points_camera_frame, T_relative.T)
        depth_transformed = points_transformed[..., 2]
        depth_data = cv2.warpPerspective(depth_transformed, self.homography, (w, h))

        #adjust mask focus area
        for i in range(len(depth_data)):
            depth_data[i]-=21*(i-406)/513
        # Mask focus area
        lower = -25
        upper = 1000


        """mask out arm & outside board"""
        mask = np.zeros_like(depth_data, dtype=np.uint8)
        modified_image = cv_image.copy()
        # modified_image = self.VideoFrame.copy()


        cv2.rectangle(mask, (125,90),(1150,700), 255, cv2.FILLED)
        cv2.rectangle(mask, (550,390),(735,700), 0, cv2.FILLED)

        cv2.rectangle(cnt_image, (125,90),(1150,700), (255, 0, 0), 2)
        cv2.rectangle(cnt_image, (550,400),(735,700), (255, 0, 0), 2)

        #edge detection
        # depth_data = depth_data.astype(np.uint8)
        # edges = cv2.Canny(depth_data,0,30)
        # thresh = cv2.bitwise_and(edges,edges, mask=mask)
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #orginal working code
        thresh = cv2.bitwise_and(cv2.inRange(depth_data, lower, upper), mask)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(cnt_image, contours, -1, (0,255,255), thickness=1)    
        
        if(self.camera_calibrated): 
            rgb_image = self.VideoFrame.copy()
            self.block_detections = np.empty((0,5))
            for contour in contours:
                color = retrieve_area_color(rgb_image, contour, colors)
                theta = cv2.minAreaRect(contour)[2]
                M = cv2.moments(contour)
                if (cv2.contourArea(contour)<100):
                    continue
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.putText(cnt_image, color, (cx-30, cy+40), font, 1.0, (0,0,0), thickness=2)
                cv2.putText(cnt_image, str(int(theta)), (cx, cy), font, 0.5, (255,255,255), thickness=2)
                cv2.circle(cnt_image,(cx,cy),5,(0,255,0),-1)

                # epsilon = 0.03*cv2.arcLength(contour,True)
                epsilon = 0.05*cv2.arcLength(contour,True)
                approx = cv2.approxPolyDP(contour,epsilon,True)
                num_corners=len(approx)
                if num_corners == 3:
                    shape = "Triangle"
                elif num_corners == 4:
                    # Check if it's a square or rectangle
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    if (abs(w-h)<10 or abs(aspect_ratio -1) < 0.15 )and(cv2.contourArea(contour)>1850 or cv2.contourArea(contour)<=850):
                        shape = "Square"
                        # print("Square aspect ration: ",aspect_ratio)
                        # print("Square w-h :", w-h)
                        # print("Area: ",cv2.contourArea(contour))
                    else:
                        # print("Rectangle aspect ration: ",aspect_ratio)
                        # print("Rectangle w-h :", w-h)
                        shape = "Rectangle"
                elif num_corners == 5:
                    shape = "Pentagon"
                elif num_corners > 5:
                    shape = "Circle"
                else:
                    shape = "Unknown"
                cv2.putText(cnt_image, shape, (cx-30, cy-40), font, 1.0, (0,0,0), thickness=2)
                if(cv2.contourArea(contour)>1850 and shape == "Square"):
                    a = np.array([[color,'Large',cx,cy,math.radians(theta)]])
                    self.block_detections = np.vstack((self.block_detections,a))
                if(cv2.contourArea(contour)<=850 and shape == "Square"):
                    a = np.array([[color,'Small',cx,cy,math.radians(theta)]])
                    self.block_detections = np.vstack((self.block_detections,a))
                

        self.VideoFrame = cnt_image
        return cnt_image

        

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        pass

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        modified_image = self.VideoFrame.copy()
        # Write your code here
        radius = 2
        colour = [255,0,0]
        thickness = 2

        # for gridpoints_x in self.grid_points[0]:
        #     for gridpoints_y in self.grid_points[1]:
        #         for gridpoint_x in gridpoints_x:
        #             for gridpoint_y in gridpoints_y:
        #                 points = np.array([gridpoint_x,gridpoint_y,1])
        for r in range(self.grid_points[0].shape[0]):
            for c in range(self.grid_points[0].shape[1]):
                points = np.array([self.grid_points[0][r][c],
                                    self.grid_points[1][r][c],
                                    1])
                pixel_point = self.world_to_pixel(points)
                pixel_point = self.pixel_to_homography(pixel_point)
                modified_image = cv2.circle(modified_image,(int(pixel_point[0]),int(pixel_point[1])),radius,colour,thickness)
                
        self.GridFrame = modified_image
     
    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection

        TODO: Use the tag detections output, to draw the corners/center/tagID of
        the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
        Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

        center of the tag: (detection.centre.x, detection.centre.y) they are floats
        id of the tag: detection.id
        """
        modified_image = self.VideoFrame.copy()
        # Write your code here


        for apriltags in msg.detections:
            # change the top-left corner and bottom-right corner
        # apriltags = msg.detection 

            rectangle_top_left_corner = (int(apriltags.corners[0].x),int(apriltags.corners[0].y))
            rectangle_bottom_right_corner = (int(apriltags.corners[2].x),int(apriltags.corners[2].y))
            
            # RGB colour of rectangle
            rectangle_color = (0,0,255)

            # thickness of the rectangle
            rectangle_thickness = 2
            # modified_image = cv2.rectangle(modified_image,(400,300),(600,100),rectangle_color,rectangle_thickness)
            # drawing rectangle on the image 
            modified_image = cv2.rectangle(modified_image,rectangle_top_left_corner,rectangle_bottom_right_corner,rectangle_color,rectangle_thickness)

            # center of the apriltag
            april_tag_center_x = int(apriltags.centre.x)
            april_tag_center_y = int(apriltags.centre.y)

            # color of the center
            center_colour = (0,255,0) #RGB

            # radius of the center to be drawn
            center_radius = 2

            # thickness of the circle drawn
            center_thickness = 5

            modified_image = cv2.circle(modified_image,(april_tag_center_x,april_tag_center_y),center_radius,center_colour,center_thickness)
            
            # text_font 
            text_font = cv2.FONT_HERSHEY_SIMPLEX

            # text scale: for the size of the font
            text_scale = 1
            
            # text to write
            text = apriltags.id

            # text colour
            text_colour = (255,0,0)

            # text thickness
            text_thickness = 5
            
            # offset from the center to put the text
            offset = 30
            modified_image = cv2.putText(modified_image,"ID: "+str(text),(april_tag_center_x+offset,april_tag_center_y+offset),text_font,text_scale,text_colour,text_thickness,cv2.LINE_AA)
    # end of for loop

        self.TagImageFrame = modified_image

class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # We modified here

            cv_image = cv2.warpPerspective(cv_image, self.camera.homography, (cv_image.shape[1], cv_image.shape[0]))
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = self.camera.blockDetector(cv_image)
        # self.camera.blockDetector()
        # self.camera.VideoFrame = drawn_cv_image


class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')  
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        # print(self.camera.intrinsic_matrix)


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        # self.camera.DepthFrameRaw = self.camera.DepthFrameRaw / 2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
        
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                grid_frame = self.camera.convertQtGridFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once() # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        
        self.executor.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()