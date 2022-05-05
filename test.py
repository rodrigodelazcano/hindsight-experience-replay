import numpy as np
import gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch
import robosuite as suite
from robosuite_env.wrappers.gym_env_wrapper import GymGoalWrapper
from robosuite_env.reach_eef import ReachEEF 
from robosuite.environments.base import register_env
from rl_modules.models import actor
import robosuite.utils.transform_utils as T

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env.horizon
    return params
register_env(ReachEEF)
env = suite.make(
            env_name="ReachEEF",
            robots="Panda",
            controller_configs=suite.load_controller_config(default_controller='OSC_POSITION'),
            use_camera_obs=False,
            use_object_obs=False,
            has_renderer=True,
            has_offscreen_renderer=False,
            control_freq=10,
            horizon=5000,
            # initialization_noise = {
            #     "magnitude": 0.5,
            #     "type": "gaussian"
            # }
            # camera_names=["agentview"],
            # camera_heights=128,
            # camera_widths=128,
            # camera_depths=True,
            # camera_segmentations='element'
        )
env = GymGoalWrapper(env)
env.seed(1)
env_params = get_env_params(env)
# model = actor(env_params)
# checkpoint = torch.load('model.pt')
# model.load_state_dict(checkpoint[-1])

obs = env.reset()

while True:
    # action = model(obs)
    action = np.array([0,0,0.00,0])
    obs, reward, done, info = env.step(action)
    # print('ENV GRIPPER QUAT')
    # print(T.mat2euler(env.robots[0].sim.data.get_site_xmat("gripper0_grip_site").copy()))
    # print('OBSERVATION')
    # print(obs)
    # print('REWARD')
    # print(reward)
    # print('DONE')
    # print(done)
    # print('INFO')
    # print(info)
    env.render()
    if done:
        env.reset()
