import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

def main():
    rospy.init_node("env_cam_obs_node")
    obs_pub = rospy.Publisher("/env/cam", Image, queue_size=10)
    rate = rospy.Rate(30)

    cam = cv2.VideoCapture(2)
    bridge = CvBridge()

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam.set(cv2.CAP_PROP_FPS, 30)

    while not rospy.is_shutdown():
        ret, frame = cam.read()

        grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Camera', grayscaled)

        msg = bridge.cv2_to_imgmsg(grayscaled, encoding="mono8")
        obs_pub.publish(msg)
        rate.sleep()

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()