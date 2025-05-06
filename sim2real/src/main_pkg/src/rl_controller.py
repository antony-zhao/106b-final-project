import rospy
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from intera_interface import Limb

from cv_bridge import CvBridge
import torch
from collections import deque
import time

from utils import *
from rl.rnd import PPONetwork
from rl.robosuite_testing import RobosuitePolicy, RobosuiteValue

class SawyerRLController:

    def __init__(self, model_path, cam_dim=100, framestack=4, action_size=7, ctrl_freq=10, device="cuda"):
        self.device = device
        self.bridge = CvBridge()
        self.cam_dim = cam_dim
        self.framestack = framestack
        self.ctrl_itv = 1 / ctrl_freq

        # RL Agent Model
        policy = RobosuitePolicy(action_size, camera_dim=cam_dim, framestack=framestack)
        value = RobosuiteValue(camera_dim=cam_dim, framestack=framestack)
        self.model = PPONetwork(policy, value)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Env Setup
        self.obs_buffer = deque(maxlen=self.framestack)

        # Sawyer Interface
        self.limb = Limb("right")

        # ROS Subscribers
        self.sub_cam = Subscriber("env/cam", Image)
        self.sub_robot = Subscriber("env/robot_state", Float64MultiArray)
        self.ts_env = TimeSynchronizer([self.sub_cam, self.sub_robot], queue_size=10)
        self.ts_env.registerCallback(self.callback)
        self.last_time = 0    # used to ensure control frequency

    def callback(self, img_msg, state_msg):
        now = time.time()
        if now - self.last_time < self.ctrl_itv:
            return
        self.last_time = now
        
        frame = bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
        cv_img = cv2.resize(frame, (self.cam_dim, self.cam_dim))/255.
        frame = np.expand_dims(frame, axis=-1)

        proprio = np.array(state_msg.data, dtype=np.float32)

        # transform obs
        dim = self.cam_dim * self.cam_dim
        new_channel = np.zeros(dim)
        new_channel[:proprio.size] = proprio
        new_channel = new_channel.reshape(self.cam_dim, self.cam_dim, 1)
        obs = np.concatenate([frame, new_channel], axis=-1)

        # stack obs
        self.obs_buffer.append(obs)
        if len(self.obs_buffer) < self.framestack:
            return 
        obs_stack = np.stack(self.obs_buffer, axis=0)
        obs_stack = obs_stack.transpose(0, 3, 1, 2).reshape(-1, self.cam_dim, self.cam_dim)
        obs_tensor = torch.tensor(obs_stack).unsqueeze(0).float().to(self.device)

        # obtain action from agent
        with torch.no_grad():
            action, _ = self.model.policy_network.policy_fn(obs, det=True)
            action = torch.tanh(action).cpu().numpy()[0]
            print(action)

        # control sawyer
        # self.limb.set_joint_velocities(joint_array_to_dict(action, self.limb))


if __name__ == "__main__":
    rospy.init_node("sawyer_rl_controller")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "../../../ppo_2460.pt"
    controller = SawyerRLController(
        model_path,
        cam_dim=100,
        framestack=4,
        action_size=7,
        ctrl_freq=10,
        device=device,
    )
    rospy.spin()

