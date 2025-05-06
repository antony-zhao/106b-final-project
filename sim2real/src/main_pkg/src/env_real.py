
# this is used to get the dynamics (inertia matrix, manipulator jacobian, etc) from the robot
# in the current position, UNLESS you specify other joint angles.  see the source code
# https://github.com/ucb-ee106/baxter_pykdl/blob/master/src/sawyer_pykdl/sawyer_pykdl.py
# for info on how to use each method
import numpy as np
import rospy
import intera_interface
from intera_interface import Gripper
from sawyer_pykdl import sawyer_kinematics
from utils import *
import time

rospy.init_node("env_node")

limb = intera_interface.Limb('right')
kin = sawyer_kinematics('right')

"""
obs 
'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 
'robot0_joint_vel', 
'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_quat_site', 
'robot0_gripper_qpos', 'robot0_gripper_qvel'
"""

joint_pos = get_joint_positions(limb)
joint_pos_cos = np.cos(joint_pos)
joint_pos_sin = np.sin(joint_pos)

joint_vel = get_joint_velocities(limb)

eef_pos = np.array(list(limb.endpoint_pose()['position']))
eef_quat = np.array(list(limb.endpoint_pose()['orientation']))

gripper = Gripper('right_gripper')
gripper_qpos = np.array([gripper.get_position()])

print(joint_pos, joint_pos_cos, joint_pos_sin)
print(joint_vel)
print(eef_pos, eef_quat)
print(gripper_qpos)