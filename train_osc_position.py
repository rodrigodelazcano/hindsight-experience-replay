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

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = 200
    return params

def launch(args):
    register_env(ReachEEF)
    env = suite.make(
                env_name="ReachEEF",
                robots="Panda",
                controller_configs=suite.load_controller_config(default_controller='OSC_POSITION'),
                use_camera_obs=False,
                use_object_obs=False,
                has_renderer=False,
                has_offscreen_renderer=False,
                control_freq=10,
                horizon=200,
                initialization_noise = {
                    "magnitude": 0.5,
                    "type": "gaussian"
                }
                # camera_names=["agentview"],
                # camera_heights=128,
                # camera_widths=128,
                # camera_depths=True,
                # camera_segmentations='element'
            )

    env = GymGoalWrapper(env)
    env_params = get_env_params(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters

    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
