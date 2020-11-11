#!/usr/bin/env python3
import numpy as np
import os
import math
import cv2
from renderClass import Renderer
from dt_apriltags import Detector

import rospy
import yaml
import sys
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

import rospkg 


"""

This is a template that can be used as a starting point for the CRA1 exercise.
You need to project the model file in the 'models' directory on an AprilTag.
To help you with that, we have provided you with the Renderer class that render the obj file.

"""

class ARNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ARNode, self).__init__(node_name=node_name,node_type=NodeType.GENERIC)
        self.veh = rospy.get_namespace().strip("/")

        rospack = rospkg.RosPack()

        # Initialize an instance of Renderer giving the model in input.
        rospy.loginfo("[ARNode]: Initializing Renderer ...")
        self.renderer = Renderer(rospack.get_path('augmented_reality_apriltag') + '/src/models/duckie.obj')

        
        # April Tag Detector
        rospy.loginfo("[ARNode]: Initializing Detector ...")
        self.at_detector = Detector(searchpath=['apriltags'], families='tag36h11', nthreads=1, quad_decimate=1.0,
                               quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)

        # CV Bridge
        rospy.loginfo("[ARNode]: Initializing CV Bridge ...")
        self.bridge = CvBridge()


        # Intrinsics
        rospy.loginfo("[ARNode]: Loading Camera Intrinsics ...")
        if(not os.path.isfile(f'/data/config/calibrations/camera_intrinsic/{self.veh}.yaml')):
            rospy.logwarn(f'[AugmentedRealityBasics]: Could not find {self.veh}.yaml. Loading default.yaml')
            camera_intrinsic = self.readYamlFile(f'/data/config/calibrations/camera_intrinsic/default.yaml')
        else:
            camera_intrinsic = self.readYamlFile(f'/data/config/calibrations/camera_intrinsic/{self.veh}.yaml')
        self.K = np.array(camera_intrinsic['camera_matrix']['data']).reshape(3, 3)


        # Subscribers
        rospy.loginfo("[ARNode]: Initializing Subscribers ...")
        self.image_subscriber = rospy.Subscriber('camera_node/image/compressed', CompressedImage, self.callback)


        # Publishers
        rospy.loginfo("[ARNode]: Initializing Publishers ...")
        self.mod_img_pub = rospy.Publisher('ar_node/image/compressed', CompressedImage, queue_size=1)



    @staticmethod
    def projection_matrix(intrinsic, homography):
        """
            Write here the compuatation for the projection matrix, namely the matrix
            that maps the camera reference frame to the AprilTag reference frame.
        """

        P_list = []

        for H in homography:

            R2D_t = np.linalg.inv(intrinsic).dot(H)  # [R2D_t] = [r1, r2, t]
            R2D_t = R2D_t / np.linalg.norm(R2D_t[:, 0]) #normalize
            r1 = R2D_t[:, 0]
            r2 = R2D_t[:, 1]
            t = R2D_t[:, 2]

            # r3 must be orthogonal to r1 and r2
            r3 = np.cross(r1, r2) 

            R3D = np.column_stack((r1, r2, r3))

            # since R3D is not a proper rotation matrix yet, we have to orthonormalize it with a polar decomposition: A = RP = 
            W, U, Vt = cv2.SVDecomp(R3D)
            R3D = U.dot(Vt)

            R3D_t = np.column_stack((R3D, t)) # R3D_t = [r1, r2, r3, t]

            P = intrinsic.dot(R3D_t)

            P_list.append(P)

        return P_list


    def callback(self, imgmsg):

        # Convert img message to cv2
        img = self.readImage(imgmsg)

        # Convert to greyscale
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect april tags
        tags = self.at_detector.detect(img_grey, estimate_tag_pose=False, camera_params=None, tag_size=None)
        # self.visualize_at_detection(img, tags) #debug visualization, marks april tags

        # Get homographies of april tags
        homography_list = [tag.homography for tag in tags]

        # Compute P from homographies
        P_list = self.projection_matrix(self.K, homography_list)

        # Render duckies on april tags
        for P in P_list:
            img = self.renderer.render(img, P)

        # Publish modified image
        modified_imgmsg = self.bridge.cv2_to_compressed_imgmsg(img)
        modified_imgmsg.header.stamp = rospy.Time.now()
        self.mod_img_pub.publish(modified_imgmsg)

        return


    def readImage(self, msg_image):
        """
            Convert images to OpenCV images
            Args:
                msg_image (:obj:`CompressedImage`) the image from the camera node
            Returns:
                OpenCV image
        """
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg_image)
            return cv_image
        except CvBridgeError as e:
            self.log(e)
            return []

    def readYamlFile(self,fname):
        """
            Reads the 'fname' yaml file and returns a dictionary with its input.

            You will find the calibration files you need in:
            `/data/config/calibrations/`
        """
        if(not os.path.isfile(fname)):
            rospy.logwarn("[ARNode]: Could not find file in %s" %fname)
            return

        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return

    @staticmethod
    def visualize_at_detection(img, tags):
        """
        Visualizes detected april tags for debugging.
        """
        for tag in tags:
            for index in range(len(tag.corners)):
                cv2.line(img, tuple(tag.corners[index - 1, :].astype(int)), tuple(tag.corners[index, :].astype(int)),
                         (0, 255, 0))

            cv2.putText(img, str(tag.tag_id),
                        org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        color=(0, 0, 255))

    def on_shutdown(self):
        super(ARNode, self).on_shutdown()



if __name__ == '__main__':
    # Initialize the node
    camera_node = ARNode(node_name='augmented_reality_apriltag_node')
    # Keep it spinning to keep the node alive
    rospy.loginfo("[ARNode]: Node is up and running ...")
    rospy.spin()
