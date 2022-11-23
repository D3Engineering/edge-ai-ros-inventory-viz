#!/usr/bin/env python3
import rospy
import sys
import cv2
import numpy as np
from d3_apriltag.msg import AprilTagDetection, AprilTagDetectionArray
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseWithCovariance, PoseWithCovarianceStamped, Point, Quaternion
import std_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial import distance
from pylibdmtx.pylibdmtx import decode


# from pyzbar.pyzbar import decode

class datamatrix_visualizer:
    def __init__(self, camera_info):
        self.d3_blue_color = (239, 174, 0)
        self.camera_info = camera_info
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/imx390/image_raw_rgb', Image, self.image_callback, queue_size=1)
        self.display_width = 1920
        self.display_height = 1080
        item_image_size = 100
        item_image_locs = [{'dtop': 200, 'dleft': 75, 'inCorn': []}, {'dtop': 500, 'dleft': 75, 'inCorn': []}, {'dtop': 800, 'dleft': 75, 'inCorn': []},
                           {'dtop': 20, 'dleft': 500, 'inCorn': []}, {'dtop': 20, 'dleft': 910, 'inCorn': []}, {'dtop': 20, 'dright': 500, 'inCorn': []},
                           {'dbottom': 20, 'dleft': 500, 'inCorn': []}, {'dbottom': 20, 'dleft': 910, 'inCorn': []}, {'dbottom': 20, 'dright': 500, 'inCorn': []},
                           {'dtop': 200, 'dright': 75, 'inCorn': []}, {'dtop': 500, 'dright': 75, 'inCorn': []}, {'dtop': 800, 'dright': 75, 'inCorn': []}]
        for iil in item_image_locs:
            if 'dtop' in iil and 'dleft' in iil:
                iil['inCorn'] = [(iil['dleft']+item_image_size, iil['dtop']), # Inner Top Corner
                    (iil['dleft']+item_image_size, iil['dtop']+item_image_size)] # Inner Bottom Corner
            elif 'dtop' in iil and 'dright' in iil:
                iil['inCorn'] = [(self.display_width-iil['dright']-item_image_size, iil['dtop']), # Inner Top Corner
                    (self.display_width-iil['dright']-item_image_size, iil['dtop']+item_image_size)] # Inner Bottom Corner
            elif 'dbottom' in iil and 'dleft' in iil:
                iil['inCorn'] = [(iil['dleft'], self.display_height-iil['dbottom']-item_image_size), # Inner Left Corner
                    (iil['dleft']+item_image_size, self.display_height-iil['dbottom']-item_image_size)] # Inner Right Corner
            elif 'dbottom' in iil and 'dright' in iil:
                iil['inCorn'] = [(self.display_width-iil['dright']-item_image_size, self.display_height-iil['dbottom']-item_image_size), # Inner Left Corner
                                 (self.display_width-iil['dright'], self.display_height-iil['dbottom']-item_image_size)] # Inner Right Corner
            else:
                print("Error during item_image_locs initialization... Exiting")
                exit()
        self.item_image_locs = item_image_locs
        tti = {'D1': "Bearings", 'D2': "Belts", 'D3': "Bolts", 'D4': "Buttons", 'D5': "Chains", 'D6': "Gears",
               'D7': "Motors", 'D8': "Nuts", 'D9': "Shafts", 'D10': "Solder", 'D11': "Sprockets", 'D12': "Switches",
               'D13': "Washers", 'D14': "Wires"}
        self.tag_to_item = tti
        self.respath = "/root/j7ros_home/ros_ws/src/robotics_sdk/ros1/drivers/edge-ai-ros-inventory-viz/res/"
        self.current_image = None
        cv2.namedWindow("Robot Monitor", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Robot Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def get_blank_image(self, num_channels=3):
        blank_image = np.zeros([self.display_height, self.display_width, num_channels], dtype=np.uint8)
        blank_image.fill(255)
        return blank_image

    def image_overlay_pos(self, bg_image, ol_image, ol_dtop=None, ol_dbottom=None, ol_dleft=None, ol_dright=None):
        bgh, bgw = bg_image.shape[:2]
        olh, olw = ol_image.shape[:2]
        olsh, olsw, oleh, olew = [None] * 4
        if ol_dtop is not None:
            olsh = ol_dtop
            oleh = olsh + olh
        elif ol_dbottom is not None:
            oleh = bgh - ol_dbottom
            olsh = oleh - olh
        else:
            raise Exception("ol_dtop or ol_dbottom must be specified")
        if ol_dleft is not None:
            olsw = ol_dleft
            olew = olsw + olw
        elif ol_dright is not None:
            olew = bgw - ol_dright
            olsw = olew - olw
        else:
            raise Exception("ol_dleft or ol_dright must be specified")
        bg_image[olsh:oleh, olsw:olew] = ol_image
        drawn_image = dict()
        drawn_image['left'] = olsw
        drawn_image['right'] = olew
        drawn_image['top'] = olsh
        drawn_image['bottom'] = oleh
        return bg_image, drawn_image

    def scale_image(self, image, scale_percent):
        sw = int(image.shape[1] * scale_percent / 100)
        sh = int(image.shape[0] * scale_percent / 100)
        scaled_image = cv2.resize(image, (sw, sh))
        return scaled_image

    def get_blank_slide(self):
        blank_slide = self.get_blank_image()
        blank_slide, image_pos = self.image_overlay_pos(blank_slide,
                                                        self.scale_image(cv2.imread(self.respath + "D3TI.jpg"), 50),
                                                        ol_dright=0, ol_dbottom=0)
        return blank_slide

    def image_overlay_center(self, bg_image, ol_image):
        bgh, bgw = bg_image.shape[:2]
        olh, olw = ol_image.shape[:2]
        olsh, olsw = int((bgh / 2) - (olh / 2)), int((bgw / 2) - (olw / 2))
        oleh = olsh + olh
        olew = olsw + olw
        bg_image[olsh:oleh, olsw:olew] = ol_image
        return bg_image, (olsw, olsh)

    def scale_image_and_detections(self, image, detections, scale_percent):
        scaled_image = self.scale_image(image, scale_percent)
        scaled_detections = []
        for d in detections:
            sd = dict()
            sd['data'] = str(d.data, 'utf-8')
            sd['rect'] = dict()
            sd['rect']['left'] = int(d.rect.left * scale_percent / 100)
            sd['rect']['top'] = int(d.rect.top * scale_percent / 100)
            sd['rect']['width'] = int(d.rect.width * scale_percent / 100)
            sd['rect']['height'] = int(d.rect.height * scale_percent / 100)
            scaled_detections.append(sd)
        return scaled_image, scaled_detections

    def draw_detected_objects(self, image, drawn_rects):
        for d in drawn_rects.values():
            image = cv2.rectangle(image,
                                          (d['left'], d['top']), (d['right'], d['bottom']),
                                          self.d3_blue_color, 2)
            cv2.waitKey(500)
            cv2.imshow("Robot Monitor", image)
            lowest_corner_distance = 1920
            lowest_corner_obj = None
            lowest_corner_incorn = None
            corner_of_rect = None
            for iil in self.item_image_locs:
                for inCorn in iil['inCorn']:
                    tldist = distance.euclidean((d['left'], d['top'], 0), (inCorn[0], inCorn[1], 0))
                    if tldist < lowest_corner_distance:
                        lowest_corner_distance = tldist
                        lowest_corner_obj = iil
                        lowest_corner_incorn = inCorn
                        corner_of_rect = (d['left'], d['top'])
                    trdist = distance.euclidean((d['right'], d['top'], 0), (inCorn[0], inCorn[1], 0))
                    if trdist < lowest_corner_distance:
                        lowest_corner_distance = trdist
                        lowest_corner_obj = iil
                        lowest_corner_incorn = inCorn
                        corner_of_rect = (d['right'], d['top'])
                    bldist = distance.euclidean((d['left'], d['bottom'], 0), (inCorn[0], inCorn[1], 0))
                    if bldist < lowest_corner_distance:
                        lowest_corner_distance = bldist
                        lowest_corner_obj = iil
                        lowest_corner_incorn = inCorn
                        corner_of_rect = (d['left'], d['bottom'])
                    brdist = distance.euclidean((d['right'], d['bottom'], 0), (inCorn[0], inCorn[1], 0))
                    if brdist < lowest_corner_distance:
                        lowest_corner_distance = brdist
                        lowest_corner_obj = iil
                        lowest_corner_incorn = inCorn
                        corner_of_rect = (d['right'], d['bottom'])
            self.item_image_locs.remove(lowest_corner_obj)

            lco = lowest_corner_obj
            ol_coords = None
            if 'dtop' in lco and 'dleft' in lco:
                image, ol_coords = self.image_overlay_pos(image, cv2.imread(self.respath + self.tag_to_item[d['data']] + ".png"), ol_dtop=lco['dtop'],
                                       ol_dleft=lco['dleft'])

            elif 'dtop' in lco and 'dright' in lco:
                image, ol_coords = self.image_overlay_pos(image, cv2.imread(self.respath + self.tag_to_item[d['data']] + ".png"), ol_dtop=lco['dtop'],
                                       ol_dright=lco['dright'])
            elif 'dbottom' in lco and 'dleft' in lco:
                image, ol_coords = self.image_overlay_pos(image, cv2.imread(self.respath + self.tag_to_item[d['data']] + ".png"), ol_dbottom=lco['dbottom'],
                                       ol_dleft=lco['dleft'])
            elif 'dbottom' in lco and 'dright' in lco:
                image, ol_coords = self.image_overlay_pos(image, cv2.imread(self.respath + self.tag_to_item[d['data']] + ".png"), ol_dbottom=lco['dbottom'],
                                       ol_dright=lco['dright'])
            else:
                print("Error during item_image_locs drawing... Exiting")
                exit()
            cv2.rectangle(image,
                          (ol_coords['left'], ol_coords['top']), (ol_coords['right'], ol_coords['bottom']),
                          self.d3_blue_color, 2)
            image = cv2.line(image, lowest_corner_incorn, corner_of_rect, self.d3_blue_color, 2)
            cv2.waitKey(750)
            cv2.imshow("Robot Monitor", image)

    def run_detect(self, image, expected_num_codes, timeout=3000):
        print("Running Data Matrix Detection for " + str(expected_num_codes) + " Codes and a timeout of " + str(timeout))
        detections = decode(self.current_image, timeout=timeout)
        rospy.loginfo(detections)
        print(str(len(detections)) + " Data Matrix codes detected")
        if len(detections) < expected_num_codes:
            print("Did not reach Target of " + str(expected_num_codes))
            return self.run_detect(image, expected_num_codes, timeout+1000)
        for d in detections:
            if str(d.data, 'utf-8') not in self.tag_to_item:
                print("Found bad value for tag: " + str(d.data, 'utf-8'))
                return self.run_detect(image, expected_num_codes, timeout+1000)
        return detections



    def image_callback(self, image):
        rospy.loginfo("Got Image")
        try:
            self.image_sub.unregister()
            self.current_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

            display_image = self.get_blank_slide()
            cv2.imshow("Robot Monitor", display_image)
            cv2.waitKey(100)
            display_image, temp = self.image_overlay_center(display_image, cv2.imread(self.respath + "Scanning.png"))
            cv2.imshow("Robot Monitor", display_image)
            cv2.waitKey(100)
            detections = self.run_detect(self.current_image, 4)
            display_image = self.get_blank_slide()
            scaled_image, detections = self.scale_image_and_detections(self.current_image, detections, 75)
            display_image, cam_ol_coords = self.image_overlay_center(display_image, scaled_image)
            text_drawn_at_y = 50
            # display_image = cv2.putText(display_image, str(len(detections)) + " Data Matrix codes Detected",
            #                             (0, text_drawn_at_y), cv2.FONT_HERSHEY_DUPLEX, 1.0, self.d3_blue_color, 2)
            drawn_rects = dict()
            cv2.imshow("Robot Monitor", display_image)
            for d in detections:
                text_drawn_at_y += 50
                # display_image = cv2.putText(display_image, str(d['data']) + " -> " + self.tag_to_item[d['data']],
                #                             (0, text_drawn_at_y), cv2.FONT_HERSHEY_DUPLEX, 1.0, self.d3_blue_color, 2)
                top = display_image.shape[0] - cam_ol_coords[1] - d['rect']['top']
                left = cam_ol_coords[0] + d['rect']['left']
                bottom = top - d['rect']['height']
                right = left + d['rect']['width']
                # display_image = cv2.rectangle(display_image,
                #                               (left, top), (right, bottom),
                #                               self.d3_blue_color, 2)
                draw_data = dict()
                draw_data['data'] = d['data']
                draw_data['top'] = top
                draw_data['left'] = left
                draw_data['bottom'] = bottom
                draw_data['right'] = right
                draw_data['width'] = d['rect']['width']
                draw_data['height'] = d['rect']['height']
                drawn_rects[d['data']] = draw_data
                # cv2.waitKey(1000)
                cv2.imshow("Robot Monitor", display_image)
            self.draw_detected_objects(display_image, drawn_rects)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            exit()
        except CvBridgeError as e:
            rospy.loginfo(e)


def main(args):
    rospy.init_node('datamatrix_visualizer', anonymous=True)

    # Get camera name from parameter server
    camera_name = rospy.get_param("~camera_name", "camera")
    camera_info_topic = "/{}/camera_info".format(camera_name)
    rospy.loginfo("Waiting on camera_info: %s" % camera_info_topic)

    # Wait until we have valid calibration data before starting
    camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
    camera_info = np.array(camera_info.K, dtype=np.float32).reshape((3, 3))
    rospy.loginfo("Camera intrinsic matrix: %s" % str(camera_info))
    dmtx = datamatrix_visualizer(camera_info)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting Down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Starting DataMatrix Detction/Visualizer Node")
    main(sys.argv)
