#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from intera_interface import Limb, Gripper
from sawyer_pykdl import sawyer_kinematics
from utils import *

def get_observation(limb, kin, gripper):
    joint_pos = get_joint_positions(limb)
    joint_pos_cos = np.cos(joint_pos)
    joint_pos_sin = np.sin(joint_pos)
    joint_vel = get_joint_velocities(limb)

    pose = limb.endpoint_pose()
    eef_pos = np.array(list(pose['position']))
    eef_quat = np.array(list(pose['orientation']))

    gripper_qpos = np.array([gripper.get_position()])

    obs = np.concatenate([
        joint_pos,
        joint_pos_cos,
        joint_pos_sin,
        joint_vel,
        eef_pos,
        eef_quat,
        gripper_qpos
    ])

    return obs

def main():
    rospy.init_node("env_robot_obs_node")
    rate = rospy.Rate(10)

    limb = Limb('right')
    kin = sawyer_kinematics('right')
    gripper = Gripper('right_gripper')

    obs_pub = rospy.Publisher("/env/robot_state", Float64MultiArray, queue_size=10)

    while not rospy.is_shutdown():
        obs = get_observation(limb, kin, gripper)
        msg = Float64MultiArray()
        msg.data = obs.tolist()
        obs_pub.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    main()
