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
from scipy.spatial.transform import Rotation as R
from pylibdmtx.pylibdmtx import decode
#from pyzbar.pyzbar import decode

class datamatrix_visualizer:
	def __init__(self, camera_info):
		self.camera_info = camera_info
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber('/imx390/image_raw_rgb', Image, self.image_callback, queue_size=1)
		self.image_pub = rospy.Publisher('/imx390/image_fused', Image, queue_size=1)
		self.current_image = None

	def image_callback(self, image):
		rospy.loginfo("Got Image")
		try:
			self.current_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='bgr8')
			#scale_percent = 50
			#width = int(self.current_image.shape[1] * scale_percent / 100)
			#height = int(self.current_image.shape[0] * scale_percent / 100)
			#scaled_image = cv2.resize(self.current_image, (width, height))
			detections = decode(self.current_image, timeout=1500)
			rospy.loginfo(detections)
			print(str(len(detections)) + " Data Matrix codes detected")
			for d in detections:
				top = 1096-d.rect.top
				left = d.rect.left
				self.current_image = cv2.rectangle(self.current_image, (left, top), (left+d.rect.width, top-d.rect.height), (255, 0, 0), 2)
				self.current_image = cv2.putText(self.current_image, str(d.data), (left, top-d.rect.height), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 1)
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.current_image, encoding='bgr8'))
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
