
import gym
import robosuite as suite
from wrappers.gym_env_wrapper import GymWrapper, GymGoalWrapper
from robosuite.utils.observables import Observable, sensor
from robosuite.environments.base import register_env
import numpy as np
from reach_eef import ReachEEF

register_env(ReachEEF)


env = suite.make(
        env_name="ReachEEF",
        robots="Panda",
        controller_configs=suite.load_controller_config(default_controller='JOINT_VELOCITY'),
        use_camera_obs=False,
        use_object_obs=False,
        has_renderer=True,
        has_offscreen_renderer=False,
        control_freq=20,
        horizon=100,
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

env = GymGoalWrapper(env)
env = gym.wrappers.RecordEpisodeStatistics(env)
# low, high = env.action_spec
done = False
env.reset()
action = [0]*8
# action[7] = 0
i = 0
while True:
    if done:
        env.reset()
        i+=1
        if i%2 == 0:
            action[7] = -0.1
        else:
            action[7] = 0.002
    
    # action = env.action_space.sample()
    # action = np.random.uniform(low, high)
    # print(action)
    ob_dict, reward, done, info = env.step(action)

    env.render()
    
#     # print('OBSERVATION DICT')
#     # print(ob_dict)


# ## Train the reach primitive with self-play
# ## Alice will try to create new poses that Bob will try to get to
# ## The goal is a position + orintation tuple (orientation in euler angles if possible)
# ## Implement Behavioral Clonning and HER
# ## Is it possible DDPG?


