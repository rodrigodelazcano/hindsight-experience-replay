import numpy as np
import os
import argparse
import robosuite as suite
from robosuite.utils.observables import Observable, sensor
from wrappers.gym_env_wrapper import GymWrapper
from model.actor_critic import Actor, Critic
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from stable_baselines3.common.buffers import DictReplayBuffer
import matplotlib.pyplot as plt


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")


    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")


    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1000),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--random-start-steps", type=int, default=25e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    args = parser.parse_args()

    return args

def convert_dict_obs_to_tensor(obs):
    torch_obs = {}
    for key, value in obs.items().copy():
        torch_obs[key] = torch.Tensor(value).to(device)
    
    return torch_obs

if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


    env= suite.make(
        env_name="NutAssemblyRound",
        robots="Panda",
        controller_configs=suite.load_controller_config(default_controller='JOINT_VELOCITY'),
        use_camera_obs=True,
        use_object_obs=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        control_freq=20,
        horizon=300,
        camera_names=["agentview"],
        camera_heights=128,
        camera_widths=128,
        camera_depths=True,
        camera_segmentations='element'
    )

    pf = env.robots[0].robot_model.naming_prefix
    modality = f"{pf}proprio"

    @sensor(modality=modality)
    def eef_force(obs_cache):
        return env.robots[0].ee_force

    @sensor(modality=modality)
    def eef_torque(obs_cache):
        return env.robots[0].ee_torque

    gripper_force_torque_obs = [("eef_force", eef_force),
                                ("eef_torque", eef_torque)]
    
    for name, sensor in gripper_force_torque_obs:
        observable = Observable(
        name=name,
        sensor=sensor,
        sampling_rate=env.control_freq,
        )
        env.add_observable(observable)
    
    env = GymWrapper(env)

    # envs = gym.vector.SyncVectorEnv([lambda: GymWrapper(env)])
    replay_buffer = DictReplayBuffer(args.buffer_size, env.observation_space, env.action_space, device=device, handle_timeout_termination=False)

    # == POLICY NETWORKS ==
    actor = Actor(env.observation_space, env.action_space).to(device)
    q_function = Critic(env.observation_space, env.action_space).to(device)

    # == TARGET NETWORKS ==
    actor_target = Actor(env.observation_space, env.action_space)
    q_function_target = Critic(env.observation_space, env.action_space)
    
    # Initialize same weights for policy and target
    actor_target.load_state_dict(actor.state_dict())
    q_function_target.load_state_dict(q_function.state_dict())

    # Initialize optimizers
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)
    q_function_optimizer = optim.Adam(list(q_function.parameters()), lr=args.learning_rate)

    obs = env.reset()
    done = False
    for step in range(args.total_timesteps):
        if done:
            obs = env.reset()
        if step < args.random_start_steps:
            action = env.action_space.sample()
        else:
            torch_obs = convert_dict_obs_to_tensor(obs)

            action = actor(torch_obs)
            action = action + np.random.normal(0, env.action_space.high[0]*args.exploration_noise, 
                                                size=env.action_space.shape[0]).clip(env.action_space.low[0], env.action_space.high[0])
        

        next_obs, reward, done, info = env.step(action)
        replay_buffer.add(obs, next_obs, action, reward, done, info)

        obs = next_obs

        if step > args.random_start_steps:
            sample_states = replay_buffer.sample(args.batch_size)

            with torch.no_grad():
                next_state_action = actor_target(sample_states.next_observations).clamp(env.action_space.low[0], env.action_space.high[0])
                next_state_q_value = q_function_target(data.next_observations, next_state_action)
                target_q_value = sample_states.rewards.flatten() + (1 - sample_states.dones.flatten()) * args.gamma * (next_state_q_value).view(-1)

            current_q_value = q_function(sample_states.observations, sample_states.actions).view(-1)
            q_function_loss = F.mse(target_q_value, current_q_value)

            q_function_optimizer.zero_grad()
            q_function_loss.backward()
            
            nn.util.clip_grad_norm_(list(q_function.parameters()), args.max_grad_norm)
            q_function_optimizer.step()

            # update policy after certain number of steps
            if step % args.policy_frequency == 0:
                actor_loss = -q_function(sample_states.observations, actor(sample_states.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.bakcward()
                nn.utils.clip_grad_norm_(list(actor.parameters()), args.max_grad_norm)
                actor_optimizer.step()

                # update the target networks with Polyak averaging
                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(q_function.parameters(), q_function_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
