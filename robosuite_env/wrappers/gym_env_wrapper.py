import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
from collections import OrderedDict
from robosuite.wrappers import Wrapper

# == Camera names for Panda:
# frontview
# birdview
# agentview
# sideview
# robot0_robotview
# robot0_eye_in_hand

class GymWrapper(Wrapper, gym.Env):
    def __init__(self, env) -> None:
        super().__init__(env=env)
        self.env = env
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__
        
        low, high = self.env.action_spec

        self.action_space = spaces.Box(low=low, high=high)
        obs_spec = self.env.observation_spec()
        # robot_joint_state keys of oringinal observation space
        robot_state_original_keys = ['robot0_joint_pos_cos', 'robot0_joint_vel']
        robot_joint_state_dim = 0
        for key in robot_state_original_keys:
            robot_joint_state_dim += obs_spec[key].shape[0]
        
        # gripper_state keys of oringinal observation space
        gripper_state_original_keys = ['robot0_gripper_qpos', 'robot0_gripper_qvel', 'eef_force', 'eef_torque']
        gripper_state_dim = 0
        for key in gripper_state_original_keys:
            gripper_state_dim += obs_spec[key].shape[0]
        
        # object_state keys of oringinal observation space
        # object_state_dim = obs_spec['object-state'].shape[0]
        
        obs_space = OrderedDict({
            "robot_joint_state": spaces.Box(low=np.array([-np.inf]*robot_joint_state_dim), high=np.array([np.inf]*robot_joint_state_dim),dtype=np.float32),
            "gripper_state": spaces.Box(low=np.array([-np.inf]*gripper_state_dim), high=np.array([np.inf]*gripper_state_dim),dtype=np.float32),
            # "object_state": spaces.Box(low=np.array([-np.inf]*object_state_dim), high=np.array([np.inf]*object_state_dim),dtype=np.float32),
            # "rgb_image": spaces.Box(low=np.zeros(obs_spec['agentview_image'].shape), high=np.ones(obs_spec['agentview_image'].shape),dtype=np.float32),
            # "depth_image": spaces.Box(low=np.zeros(obs_spec['agentview_depth'].shape), high=np.ones(obs_spec['agentview_depth'].shape),dtype=np.float32),
            # "segmentation_image": spaces.Box(low=np.ones(obs_spec['agentview_segmentation_element'].shape), high=np.ones(obs_spec['agentview_segmentation_element'].shape),dtype=np.float32)
        })

        self.observation_space = spaces.Dict(sorted(obs_space.items()))
    
    def _process_observation(self, obs):
        new_obs = {}
        robot_joint_state = np.concatenate((obs['robot0_joint_pos_cos'], obs['robot0_joint_vel']), axis=-1)
        gripper_state = np.concatenate((obs['robot0_gripper_qpos'], obs['robot0_gripper_qvel'], obs['eef_force'], obs['eef_torque']), axis=-1)
        new_obs["robot_joint_state"] = robot_joint_state
        new_obs["gripper_state"] = gripper_state
        # new_obs["object_state"] = obs["object-state"] 
        # new_obs["rgb_image"] = np.flip(obs["agentview_image"])
        # new_obs["depth_image"] = np.flip(obs["agentview_depth"])
        # new_obs["segmentation_image"] = np.flip(obs["agentview_segmentation_element"])

        return new_obs 
        
    def reset(self):
        obs = self.env.reset()
        return self._process_observation(obs)
    
    def seed(self, seed=None):
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        ob_dict, reward, done, info = self.env.step(action)
        return self._process_observation(ob_dict), reward, done, info
    
    def render(self, mode='human'):
        self.env.render()

class GymGoalWrapper(Wrapper, gym.GoalEnv):
    def __init__(self, env) -> None:
        super().__init__(env=env)
        self.env = env
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__
        
        low, high = self.env.action_spec

        self.action_space = spaces.Box(low=low, high=high)
        obs_spec = self.env.observation_spec()

        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=(7,), dtype="float32"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=(7,), dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs_spec['robot0_proprio-state'].shape, dtype="float32"
                ),
            )
        )
    
    def _process_observation(self, obs): 
        achieved_goal_pose = self.env.robots[0]._hand_pose.copy()
        if self.env.robot_configs[0]["controller_config"]["type"] == "JOINT_VELOCITY":          
            propio_obs = obs['robot0_proprio-state'].copy()
            gripper_pos = self.env.robots[0].sim.data.get_site_xpos("gripper0_grip_site").copy()
            dt = self.env.robots[0].sim.model.opt.timestep * self.env.robots[0].sim.data.get_site_xpos("gripper0_grip_site")
            grip_velp = self.env.robots[0].sim.data.get_site_xvelp("gripper0_grip_site") * dt
            achieved_goal_pos = self.env.robots[0].sim.data.get_site_xpos("gripper0_grip_site").copy()
            achieved_goal_quat = self.env.robots[0].sim.data.get_site_xmat("gripper0_grip_site").copy()
            return {
                "observation": propio_obs,
                "achieved_goal": np.concatenate((achieved_goal_pos,achieved_goal_quat), axis=0),
                "desired_goal": self.goal.copy(),
            }
        elif self.env.robot_configs[0]["controller_config"]["type"] == "OSC_POSITION":

            robot_joint_vel = obs['robot0_joint_vel'].copy()
            robot_join_pos_cos = obs['robot0_joint_pos_cos'].copy()
            achieved_goal_pos,_ = self.env.gripper_pose_in_world(achieved_goal_pose)
            gripper_pos = self.env.robots[0].sim.data.get_site_xpos("gripper0_grip_site").copy()
            dt = self.env.robots[0].sim.model.opt.timestep * self.env.robots[0].sim.data.get_site_xpos("gripper0_grip_site")
            grip_velp = self.env.robots[0].sim.data.get_site_xvelp("gripper0_grip_site") * dt

            return {
                "observation": np.concatenate((robot_joint_vel,robot_join_pos_cos,gripper_pos,grip_velp), axis=0),
                "achieved_goal": gripper_pos.copy(),
                "desired_goal": self.goal.copy(),
            }
        

        
    def reset(self):
        obs = self.env.reset()
        return self._process_observation(obs)
    
    def seed(self, seed=None):
        # Seed the generator
        if seed is not None:
            try:
                np.random.seed(seed)
            except:
                TypeError("Seed must be an integer type!")
        seed = self.env.seed(seed)
        return seed
    
    def _is_success(self, achieved_goal, desired_goal):
        if self.env.robot_configs[0]["controller_config"]["type"] == "JOINT_VELOCITY":
            d = self.goal_euclidian_distance(desired_goal[:3], achieved_goal[:3])
            d_angle = self.goal_angle_distance(desired_goal[3:], achieved_goal[3:])
            return ((d < self.distance_threshold).astype(np.float32) and (d_angle < self.angle_threshold).astype(np.float32))
        elif self.env. robot_configs[0]["controller_config"]["type"] == "OSC_POSITION":
            d = self.goal_euclidian_distance(desired_goal, achieved_goal)
            return (d < self.distance_threshold).astype(np.float32)
    def step(self, action):
        ob_dict, reward, done, info = self.env.step(action)
        obs = self._process_observation(ob_dict)
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }
        return obs, reward, done, info
    
    def render(self, mode='human'):
        self.env.render()
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        if self.env.robot_configs[0]["controller_config"]["type"] == "JOINT_VELOCITY":
            d = self.goal_euclidian_distance(desired_goal[:, :3], achieved_goal[:, :3])
            d_angle = self.goal_angle_distance(desired_goal[:, 3:], achieved_goal[:, 3:])

            compute_reward_return = -np.multiply((d > self.distance_threshold).astype(np.float32), (d_angle > self.angle_threshold).astype(np.float32))
            # Sparse reward
            return compute_reward_return

            # Dense reward
        elif self.env. robot_configs[0]["controller_config"]["type"] == "OSC_POSITION":
            d = self.goal_euclidian_distance(desired_goal, achieved_goal)
            return -(d > self.distance_threshold).astype(np.float32)

