#!/usr/bin/env python3
import rospy
import sys
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Int32
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from pylibdmtx.pylibdmtx import decode


class inventory_visualizer:
    """
    inventory_visualizer is the audience display component of the Inventory Demo - it takes in the Robot State, as well
    as images from the left camera and detection targets from the Demo node (for DataMatrix detection and visualization)
    as well as pre-processed camera/radar fusion images from the PointCloud and Tracking/CNN nodes (for fusion display)
    """

    def __init__(self, camera_info):
        print("Init Inventory Visualizer")
        # Prepare Fullscreen OpenCV Window - doing it at this stage allows us to easily stop and re-open the visualizer
        # without the robot continuing (if Demo's WAIT_FOR_PC flag is True) if the window crashes
        cv2.namedWindow("Robot Monitor", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Robot Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # Resources that will be used by the visualizer are all stored in the res/ directory of this node
        self.respath = "/root/j7ros_home/ros_ws/src/robotics_sdk/ros1/drivers/edge-ai-ros-inventory-viz/res/"
        # Create a OpenCV FreeType font, so we can use Proxima Nova with variable text
        self.font = cv2.freetype.createFreeType2()
        self.font.loadFontData(fontFileName=self.respath + 'ProximaNova-Bold.otf', id=0)
        # Prepare and Display Initial Slide
        self.state = "STARTUP|STARTUP"
        self.display_width = 1920
        self.display_height = 1080
        self.current_image = None
        cv2.imshow("Robot Monitor", self.get_blank_slide())
        cv2.waitKey(100)
        self.d3_blue_color = (239, 174, 0)
        self.camera_info = camera_info
        self.bridge = CvBridge()
        self.viz_resp_pub = rospy.Publisher('/viz_resp', String, queue_size=10)  # Used to talk to Demo node
        # Inform the Inventory Demo node that the Visualizer has completed Initialization and the Robot can continue
        self.viz_resp_pub.publish("ACK")
        # Subscribe to the Robot State, PointCloud result, and Tracking/CNN result
        self.state_sub = rospy.Subscriber('/robot_state', String, self.state_callback, queue_size=5)
        self.pcl_sub = rospy.Subscriber('/front/imx390/image_fused_pcl', Image, self.pcl_img_callback, queue_size=5)
        self.trk_sub = rospy.Subscriber('/front/imx390/image_fused_trk', Image, self.trk_img_callback, queue_size=5)
        self.expected_codes = 0
        item_image_size = 100  # The size (in px) that we expect the item images (for DataMatrix visualization) to be
        # The locations on screen that the item images should be drawn, ctr (center) is left blank, computed next
        # dtop is 'distance from top', dleft is 'distance from left', and so on. All parameters are in units of pixels.
        item_image_locs = [{'dtop': 200, 'dleft': 75, 'ctr': []},
                           {'dtop': 800, 'dleft': 75, 'ctr': []},
                           {'dbottom': 20, 'dleft': 500, 'ctr': []},
                           {'dbottom': 20, 'dright': 500, 'ctr': []},
                           {'dtop': 200, 'dright': 75, 'ctr': []},
                           {'dtop': 800, 'dright': 75, 'ctr': []}]
        # Compute the center coordinates for each item image location using the sides provided above and the known size
        # of the item images (item_image_size)
        for iil in item_image_locs:
            if 'dtop' in iil and 'dleft' in iil:
                iil['ctr'] = (iil['dleft'] + int(item_image_size / 2), iil['dtop'] + int(item_image_size / 2))  # Center
            elif 'dtop' in iil and 'dright' in iil:
                iil['ctr'] = (self.display_width - iil['dright'] - int(item_image_size / 2),
                              iil['dtop'] + int(item_image_size / 2))  # Center
            elif 'dbottom' in iil and 'dleft' in iil:
                iil['ctr'] = (iil['dleft'] + int(item_image_size / 2),
                              self.display_height - iil['dbottom'] - int(item_image_size / 2))  # Center
            elif 'dbottom' in iil and 'dright' in iil:
                iil['ctr'] = (self.display_width - iil['dright'] - int(item_image_size / 2),
                              self.display_height - iil['dbottom'] - int(item_image_size / 2))  # Center
            else:
                print("Error during item_image_locs initialization... Exiting")
                exit()
        self.item_image_locs = item_image_locs  # Make the item_image_locs object available across the class
        # Translations from the DataMatrix encoded data to the image file name that should be used
        # when that code is detected. When tag 'D1' is detected, we display respath+'Bearings.png'
        tti = {'D1': "Bearings", 'D2': "Belts", 'D3': "Bolts", 'D4': "Buttons", 'D5': "Chains", 'D6': "Gears",
               'D7': "Motors", 'D8': "Nuts", 'D9': "Shafts", 'D10': "Solder", 'D11': "Sprockets", 'D12': "Switches",
               'D13': "Washers", 'D14': "Wires"}
        self.tag_to_item = tti  # Make the tag to image object available across the class
        print("Init Complete")

    def get_blank_image(self, num_channels=3):
        """
        Create a blank OpenCV Image
        
        :param num_channels: the number of channels the image should have (default: 3 (BGR))
        :return: a blank (white) OpenCV Image
        """
        blank_image = np.zeros([self.display_height, self.display_width, num_channels], dtype=np.uint8)
        blank_image.fill(255)
        return blank_image

    def image_overlay_pos(self, bg_image, ol_image, ol_dtop=None, ol_dbottom=None, ol_dleft=None, ol_dright=None):
        """
        Overlay an OpenCV Image on another OpenCV Image by position from the sides of the background image
        Note: if ol_dtop and ol_dbottom are both specified, ol_dtop will take precedence. Likewise, if ol_dleft and
        ol_dright are both specified, ol_dleft will take precedence.
        
        :param bg_image: The background OpenCV Image
        :param ol_image: The OpenCV Image to be overlayed
        :param ol_dtop: Distance (in px) from the top of bg_image that the top of ol_image should be at
        :param ol_dbottom: Distance (in px) from the bottom of bg_image that the bottom of ol_image should be at
        :param ol_dleft:  Distance (in px) from the left of bg_image that the left of ol_image should be at
        :param ol_dright: Distance (in px) from the right of bg_image that the right of ol_image should be at
        :return: tuple of the resulting OpenCV Image, and a dictionary with the pixel locations of each side of ol_image 
        """
        bgh, bgw = bg_image.shape[:2]  # Background Height, Background Width
        olh, olw = ol_image.shape[:2]  # Overlay Height, Overlay Width
        olsh, olsw, oleh, olew = [None] * 4  # Overlay Start Height, Start Width, End Height, End Width
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

    def draw_text(self, bg_image, text, font_size, origin, color=(0, 0, 0)):
        """
        Draw Text on an OpenCV Image in Proxima Nova
        
        :param bg_image: The OpenCV Image to draw text on
        :param text: The text to draw
        :param font_size: The font height, in pixels
        :param origin: The coordinates on bg_image to draw the text at
        :param color: The color of the text to be drawn (default: (0,0,0)/black)
        :return: The resulting OpenCV Image
        """
        self.font.putText(img=bg_image, text=text, org=origin, fontHeight=font_size, color=color,
                          thickness=-1, line_type=cv2.LINE_AA, bottomLeftOrigin=False)
        return bg_image

    def scale_image(self, image, scale_percent):
        """
        Scale an OpenCV Image by a Percentage
        
        :param image: The OpenCV Image to scale 
        :param scale_percent: The percentage to scale the image by
        :return: The resulting OpenCV Image
        """
        sw = int(image.shape[1] * scale_percent / 100)
        sh = int(image.shape[0] * scale_percent / 100)
        scaled_image = cv2.resize(image, (sw, sh))
        return scaled_image

    def state_to_display_name(self):
        """
        Convert the machine-readable Robot State to a Display String for the Audience

        :return: The Robot's State summarized in a short phrase
        """
        print(self.state)
        objective_name = self.state.split("|")[0]
        state_name = self.state.split("|")[1]
        display_name = "State: "
        if state_name == "STARTUP":
            display_name += "Starting Up..."
        elif state_name == "DRIVE":
            display_name += "Driving to " + objective_name
        elif state_name == "SCAN":
            display_name += "Scanning Inventory at " + objective_name
        elif state_name == "TRACK":
            display_name += "Tracking " + objective_name
        elif state_name == "DONE":
            display_name += "Routine Complete!"
        else:
            display_name += "Unknown State: " + state_name + ", Objective: " + objective_name
        return display_name

    def get_blank_slide(self):
        """
        Get a blank 'slide' prefilled with the D3 and TI logos, and Robot State for displaying Demo Visuals

        :return: the resulting OpenCV Image
        """
        blank_slide = self.get_blank_image()
        blank_slide, image_pos = self.image_overlay_pos(blank_slide,
                                                        self.scale_image(cv2.imread(self.respath + "D3.png"), 100),
                                                        ol_dleft=0, ol_dbottom=0)
        blank_slide, image_pos = self.image_overlay_pos(blank_slide,
                                                        self.scale_image(cv2.imread(self.respath + "TI.png"), 100),
                                                        ol_dright=0, ol_dbottom=0)
        blank_slide = self.draw_text(blank_slide, self.state_to_display_name(), 100, (20, 0))
        return blank_slide

    def image_overlay_center(self, bg_image, ol_image):
        """
        Overlay an OpenCV Image in the center of another OpenCV Image

        :param bg_image: the background OpenCV Image
        :param ol_image: the OpenCV Image to be overlayed
        :return: tuple of the resulting OpenCV Image, the coordinates of the top left of ol_image in bg_image
        """
        bgh, bgw = bg_image.shape[:2]  # Background Height, Background Width
        olh, olw = ol_image.shape[:2]  # Overlay Height, Overlay Width
        olsh, olsw = int((bgh / 2) - (olh / 2)), int((bgw / 2) - (olw / 2))  # Overlay Start Height, Overlay Start Width
        oleh = olsh + olh  # Overlay End Height
        olew = olsw + olw  # Overlay End Width
        bg_image[olsh:oleh, olsw:olew] = ol_image
        return bg_image, (olsw, olsh)

    def scale_image_and_detections(self, image, detections, scale_percent):
        """
        Scale an OpenCV Image and DataMatrix Detection locations in unison

        :param image: the OpenCV Image to scale
        :param detections: the DataMatrix Detections to scale
        :param scale_percent: the percentage to scale by
        :return: tuple of the resulting OpenCV Image, and the scaled detections
        """
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
            sd['rect']['centerx'] = sd['rect']['left'] + int(sd['rect']['width'] / 2)
            sd['rect']['centery'] = sd['rect']['top'] + int(sd['rect']['height'] / 2)
            scaled_detections.append(sd)
        return scaled_image, scaled_detections

    def draw_detected_objects(self, image, drawn_rects):
        """
        Draw the detected DataMatrix items in the item_image_locs boxes on the display, connect them to the center of
        the DataMatrix detection box with a line to visualize the detection process (delays included for visual effect),
        ensuring the lines don't cross by using the Hungarian algorithm

        :param image: the OpenCV Image to draw on
        :param drawn_rects: the locations of rectangles drawn around the deteccted DataMatrix codes
        :return: None, as the resulting OpenCV Image is displayed incrementally throughout this function
        """
        original_image = image.copy()
        itembox_ctrs = []
        detection_ctrs = []
        for iil in self.item_image_locs:
            itembox_ctrs.append(iil['ctr'])
        drawn_rect_arr = []
        for d in drawn_rects.values():
            print(d)
            detection_ctrs.append(d['ctr'])
            drawn_rect_arr.append(d)
        if len(detection_ctrs) == 0:
            print("Skipping Drawing of Objects, as none were Detected")
        else:
            costmat = distance.cdist(itembox_ctrs, detection_ctrs, 'euclidean')
            solved_lsa = linear_sum_assignment(costmat)
            lsa_rows = solved_lsa[0]
            lsa_cols = solved_lsa[1]
            for i in range(len(lsa_cols)):
                sol = (lsa_rows[i], lsa_cols[i])
                print(sol)
                print("Solution: ")
                lco = self.item_image_locs[sol[0]]
                print(lco)
                d = drawn_rect_arr[sol[1]]
                print(d)
                image = cv2.rectangle(image, (d['left'], d['top']), (d['right'], d['bottom']), self.d3_blue_color, 2)
                cv2.waitKey(500)
                cv2.imshow("Robot Monitor", image)
                image = cv2.line(image, lco['ctr'], d['ctr'], self.d3_blue_color, 2, lineType=cv2.LINE_AA)
                image[d['bottom'] + 2:d['top'] - 2, d['left'] + 2:d['right'] - 2] = original_image[
                                                                                    d['bottom'] + 2:d['top'] - 2,
                                                                                    d['left'] + 2:d['right'] - 2]
                ol_coords = None
                if 'dtop' in lco and 'dleft' in lco:
                    image, ol_coords = self.image_overlay_pos(image, cv2.imread(
                        self.respath + self.tag_to_item[d['data']] + ".png"), ol_dtop=lco['dtop'],
                                                              ol_dleft=lco['dleft'])
                elif 'dtop' in lco and 'dright' in lco:
                    image, ol_coords = self.image_overlay_pos(image, cv2.imread(
                        self.respath + self.tag_to_item[d['data']] + ".png"), ol_dtop=lco['dtop'],
                                                              ol_dright=lco['dright'])
                elif 'dbottom' in lco and 'dleft' in lco:
                    image, ol_coords = self.image_overlay_pos(image, cv2.imread(
                        self.respath + self.tag_to_item[d['data']] + ".png"), ol_dbottom=lco['dbottom'],
                                                              ol_dleft=lco['dleft'])
                elif 'dbottom' in lco and 'dright' in lco:
                    image, ol_coords = self.image_overlay_pos(image, cv2.imread(
                        self.respath + self.tag_to_item[d['data']] + ".png"), ol_dbottom=lco['dbottom'],
                                                              ol_dright=lco['dright'])
                else:
                    print("Error during item_image_locs drawing... Exiting")
                    exit()
                image = cv2.rectangle(image,
                                      (ol_coords['left'], ol_coords['top']), (ol_coords['right'], ol_coords['bottom']),
                                      self.d3_blue_color, 2)
                cv2.waitKey(750)
                cv2.imshow("Robot Monitor", image)

    def run_detect(self, image, expected_num_codes, timeout=3000):
        """
        Run DataMatrix Detection on an image

        :param image: the OpenCV Image to run Detection on
        :param expected_num_codes: the number of codes we expect in the image
        :param timeout: the timeout (in ms) to limit the Detection with
        :return: The pylibdmtx detection objects, or None, if we don't reach the expected_num_codes target, or if a code
        reads with data not included in the tag to image object.
        """
        print(
            "Running Data Matrix Detection for " + str(expected_num_codes) + " Codes and a timeout of " + str(timeout))
        detections = decode(image, timeout=timeout)
        rospy.loginfo(detections)
        print(str(len(detections)) + " Data Matrix codes detected")
        if len(detections) < expected_num_codes:
            print("Did not reach Target of " + str(expected_num_codes))
            return None
        for d in detections:
            if str(d.data, 'utf-8') not in self.tag_to_item:
                print("Found bad value for tag: " + str(d.data, 'utf-8'))
                return None
        return detections

    def pcl_img_callback(self, image):
        """
        Process an Image Message received from the PointCloud Fusion Node, display it if the Robot State allows

        :param image: the ROS Image Message from a Subscriber
        :return: None
        """
        rospy.loginfo("Got PointCloud Visualized Image")
        if self.state.endswith("SCAN") or self.state.endswith("TRACK"):
            rospy.loginfo("State is SCAN or TRACK, ignoring received PointCloud Image")
        else:
            try:
                pcl_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
                display_image = self.get_blank_slide()
                display_image, ol_coords = self.image_overlay_center(display_image, self.scale_image(pcl_image, 150))
                rospy.loginfo("Updating PointCloud Frame")
                cv2.imshow("Robot Monitor", display_image)
                cv2.waitKey(50)
            except CvBridgeError as e:
                rospy.loginfo(e)

    def trk_img_callback(self, image):
        """
        Process an Image Message received from the Tracking/CNN Fusion Node, display it if the Robot State allows

        :param image: the ROS Image Message from a Subscriber
        :return: None
        """
        rospy.loginfo("Got Tracker Visualized Image")
        if not self.state.endswith("TRACK"):
            rospy.loginfo("State is not TRACK, ignoring received PointCloud Image")
        else:
            try:
                trk_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
                display_image = self.get_blank_slide()
                display_image, ol_coords = self.image_overlay_center(display_image, self.scale_image(trk_image, 150))
                rospy.loginfo("Updating Tracking Frame")
                cv2.imshow("Robot Monitor", display_image)
                cv2.waitKey(50)
            except CvBridgeError as e:
                rospy.loginfo(e)

    def dmtx_cam_img_recv(self, image):
        """
        Process and display an Image Message received from the Left Camera during our Scanning Routine

        :param image: the ROS Image Message from our wait_for_message call
        :return: False if no DataMatrix codes were detected, True if we completed the visualization of Scanning
        """
        rospy.loginfo("Got DataMatrix Image")
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')

            display_image = self.get_blank_slide()
            cv2.imshow("Robot Monitor", display_image)
            cv2.waitKey(100)
            display_image, temp = self.image_overlay_center(display_image, cv2.imread(self.respath + "Scanning.png"))
            cv2.imshow("Robot Monitor", display_image)
            cv2.waitKey(100)
            detections = self.run_detect(self.current_image, self.expected_codes)
            if detections is None:
                return False
            display_image = self.get_blank_slide()
            scaled_image, detections = self.scale_image_and_detections(self.current_image, detections, 75)
            display_image, cam_ol_coords = self.image_overlay_center(display_image, scaled_image)
            text_drawn_at_y = 50
            drawn_rects = dict()
            cv2.imshow("Robot Monitor", display_image)
            for d in detections:
                text_drawn_at_y += 50
                top = display_image.shape[0] - cam_ol_coords[1] - d['rect']['top']
                left = cam_ol_coords[0] + d['rect']['left']
                bottom = top - d['rect']['height']
                right = left + d['rect']['width']
                draw_data = dict()
                draw_data['data'] = d['data']
                draw_data['top'] = top
                draw_data['left'] = left
                draw_data['bottom'] = bottom
                draw_data['right'] = right
                draw_data['width'] = d['rect']['width']
                draw_data['height'] = d['rect']['height']
                draw_data['ctr'] = (left + int(d['rect']['width'] / 2), top - int(d['rect']['height'] / 2))
                drawn_rects[d['data']] = draw_data
                cv2.imshow("Robot Monitor", display_image)
            self.draw_detected_objects(display_image, drawn_rects)
            cv2.waitKey(5000)
            return True
        except CvBridgeError as e:
            rospy.loginfo(e)

    def state_callback(self, state):
        """
        Process an updated Robot State and update the Visualizer

        :param state: the ROS String message containing the robot State
        :return: None
        """
        rospy.loginfo("State Received: " + state.data)
        self.state = state.data
        print(self.state)
        if self.state.endswith("SCAN"):  # Enter Scanning Routine
            print("Waiting for dmtx_count")
            self.expected_codes = rospy.wait_for_message('/dmtx_count', Int32).data
            print("Waiting for image_raw_rgb")
            camera_image = rospy.wait_for_message('/left/imx390/image_raw_rgb', Image)
            while not self.dmtx_cam_img_recv(camera_image):
                print("Waiting for image_raw_rgb again")
                camera_image = rospy.wait_for_message('/left/imx390/image_raw_rgb', Image)
                self.expected_codes -= 1
            self.viz_resp_pub.publish("ACK")
        else:  # Get a Blank Slide, other Subscribers will add the appropriate Fusion image based on self.state
            print("State is not SCAN, skipping Scan Routine")
            display_image = self.get_blank_slide()
            cv2.imshow("Robot Monitor", display_image)
        cv2.waitKey(100)
        print("Completed Callback")


def main(args):
    rospy.init_node('inventory_visualizer', anonymous=True)

    # Get camera name from parameter server
    camera_name = rospy.get_param("~camera_name", "camera")
    camera_info_topic = "/{}/camera_info".format(camera_name)
    rospy.loginfo("Waiting on camera_info: %s" % camera_info_topic)

    # Wait until we have valid calibration data before starting
    camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)
    camera_info = np.array(camera_info.K, dtype=np.float32).reshape((3, 3))
    rospy.loginfo("Camera intrinsic matrix: %s" % str(camera_info))
    # Start the Inventory Visualizer
    inv_viz = inventory_visualizer(camera_info)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting Down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Starting Inventory Visualizer Node")
    main(sys.argv)
