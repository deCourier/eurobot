#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import detector
import cv2
import numpy as np
import subprocess
import calibration
import time
import color_detection


def get_cups_str(detected_field_cups, initial_field_cups):
    centers = []
    output_string = ""

    if len(detected_field_cups) != 0:

        for i in range(len(detected_field_cups)):
            center_x = int(
                detected_field_cups[i][0][0] + (detected_field_cups[i][1][0] - detected_field_cups[i][0][0]) / 2)
            center_y = int(
                detected_field_cups[i][0][1] + (detected_field_cups[i][1][1] - detected_field_cups[i][0][1]) / 2)
            centers.append((center_x, center_y))

        presence = False

        for j in range(len(initial_field_cups)):

            for i in range(len(centers)):
                if initial_field_cups[j][0][0] < centers[i][0] < initial_field_cups[j][1][0]:

                    if initial_field_cups[j][0][1] < centers[i][1] < initial_field_cups[j][1][1]:
                        presence = True

            if presence:
                output_string += "1"
                presence = False

            else:
                output_string += "0"

    else:
        return "00000000"

    return output_string


class CameraNode:
    def __init__(self):
        self.node = rospy.init_node('camera', anonymous=True)
        self.K = np.asarray(rospy.get_param("K"))
        self.D = np.asarray(rospy.get_param("D"))
        self.DIM = tuple(rospy.get_param("DIM"))
        self.projection_matrix = np.asarray(rospy.get_param("PROJECTION_MATRIX"))
        self.template_path = rospy.get_param("TEMPLATE_PATH")

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.DIM[0])
        self.cap.set(4, self.DIM[1])
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K, self.DIM,
                                                                   cv2.CV_16SC2)

        self.model_cfg_path = rospy.get_param("MODEL_CONFIG_PATH")
        self.model_weights_path = rospy.get_param("MODEL_WEIGHTS_PATH")
        self.classes = rospy.get_param("CLASSES")
        self.cup_detector = detector.Detect(self.model_cfg_path, self.model_weights_path, self.classes, DIM=(1605, 1285), confidence=0.5)

        self.seq_publisher = rospy.Publisher('/sequence', String, queue_size=1)
        self.compass_publisher = rospy.Publisher('/wind_direction', String, queue_size=1)
        self.cups_publisher = rospy.Publisher('/field_cups', String, queue_size=1)
        self.reef_publisher = rospy.Publisher('/reef_presence', String, queue_size=1)
        self.field_publisher = rospy.Publisher('/field_presence', String, queue_size=1)

        rospy.Subscriber('/main_robot/stm/start_status', String, self.start_status_callback_main, queue_size=1)
        rospy.Subscriber('/secondary_robot/stm/start_status', String, self.start_status_callback_secondary, queue_size=1)

        self.timer = -1
        self.fps_timer = -1
        self.seq = ""
        self.compass = ""
        self.reef = ""
        self.field = ""
        self.start_status = ""
        self.matrix_projection = 0
        self.crop_mask = 0

        self.find_feature_matrix()

        rospy.logwarn("INITIALIZATION COMPLETED")
        rospy.logwarn("CAMERA LOOP STARTED")
        rospy.logwarn("WAITING START STATUS")
        self.run()

    def start_status_callback_main(self, data):
        self.start_status = data.data

    def start_status_callback_secondary(self, data):
        self.start_status = data.data

    def run(self):
        start_flag = False

        while not rospy.is_shutdown():

            if self.start_status == "1" and not start_flag:
                self.timer = time.time()
                rospy.logwarn('MATCH STARTED')
                start_flag = True

            self.fps_timer = time.time()
            ret, frame = self.cap.read()
            undistorted = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
            undistorted_croped = cv2.warpPerspective(undistorted, self.matrix_projection, (2448, 1740))
            undistorted_croped = color_detection.crop(undistorted_croped)
            detected_field_cups, detected_reef_cups = self.cup_detector.detect(undistorted_croped)

            if start_flag and (time.time() - self.timer) > 0:

                if self.field == "":
                    initial_field_cups = detected_field_cups
                    initial_field_cups.sort(key=lambda k: k[0])

                self.field = get_cups_str(detected_field_cups, initial_field_cups)

                if self.reef == "":
                    initial_reef_cups = detected_reef_cups
                    initial_reef_cups.sort(key=lambda k: k[0])

                self.reef = get_cups_str(detected_reef_cups, initial_reef_cups)

                if self.seq == "":
                    seq_frame = cv2.warpPerspective(undistorted, self.matrix_projection, (2448, 1740))
                    self.seq = color_detection.findColorsHSV(seq_frame)

                if self.compass == "" and (time.time() - self.timer) > 30:
                    compas_frame = cv2.warpPerspective(undistorted, self.matrix_projection, (2448, 1740))
                    self.compass = color_detection.findCompas(compas_frame)

                self.compass_publisher.publish(self.compass)
                self.seq_publisher.publish(self.seq)
                self.reef_publisher.publish(self.reef)
                self.field_publisher.publish(self.field)

                rospy.loginfo("compass: {}".format(self.compass))
                rospy.loginfo("sequence cups: {}".format(self.seq))
                rospy.loginfo("field cups: {}".format(self.field))
                rospy.loginfo("reef cups: {}".format(self.reef))
                rospy.loginfo("time: {}".format(time.time() - self.timer))
                rospy.loginfo("fps: {}".format(1 / (time.time() - self.fps_timer)))

            if start_flag and (time.time() - self.timer) > 110:
                rospy.logwarn("MATCH ENDED")
                return 0

    def find_feature_matrix(self):
        frame_num = 0

        while frame_num < 10:
            ret, frame = self.cap.read()
            frame_num += 1

        undistorted = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        seq_frame = cv2.warpPerspective(undistorted, self.projection_matrix, (3000, 2000))
        matrix_feature = calibration.siftFeatures(seq_frame, self.template_path)
        self.matrix_projection = np.dot(matrix_feature, self.projection_matrix)
        croped_img = calibration.crop(cv2.warpPerspective(undistorted, self.matrix_projection, (2448, 1740)))
        self.crop_mask = cv2.warpPerspective(croped_img, np.linalg.inv(self.matrix_projection), self.DIM)
        self.crop_mask = cv2.cvtColor(self.crop_mask, cv2.COLOR_BGR2GRAY)
        _, self.crop_mask = cv2.threshold(self.crop_mask, 1, 1, cv2.THRESH_BINARY)


if __name__ == '__main__':

    try:
        CameraNode()
    except rospy.ROSInterruptException:
        pass

    rospy.spin()
