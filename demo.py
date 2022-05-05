import torch
from rl_modules.models import actor
from arguments import get_args
import gym
import numpy as np
import robosuite as suite
from robosuite_env.wrappers.gym_env_wrapper import GymGoalWrapper
from robosuite_env.reach_eef import ReachEEF 
from robosuite.environments.base import register_env
import time


# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

if __name__ == '__main__':
    args = get_args()
    # load the model param
    model_path = args.save_dir + args.env_name + '/model.pt'
    o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)

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
            horizon=75,
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
    
    # create the environment
    env = GymGoalWrapper(env)
    env.seed(300)
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    print('ENV PARAMS')
    print(env_params)
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    for i in range(args.demo_length):
        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        for t in range(env.horizon):
            env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            action[-1] = -1
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
