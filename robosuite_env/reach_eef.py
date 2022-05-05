from turtle import distance
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from scipy.spatial.transform import Rotation as R
import robosuite.utils.transform_utils as T
from mujoco_py.generated import const
import numpy as np
from gym.utils import seeding


class ReachEEF(SingleArmEnv):
    def __init__(self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        distance_threshold = 0.01,
        angle_threshold = 0.05,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        single_object_mode=0,
        nut_type=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=50,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,):

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.82))

        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold

        super().__init__(robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,)


    def reset(self):
        super().reset()
        self._initialize_goal()
        observations = (
            self.viewer._get_observations(force_update=True)
            if self.viewer_get_obs
            else self._get_observations(force_update=True)
        )

        # Return new observations
        return observations

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            xml="arenas/table_arena.xml"
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=None,
        )
    def _check_gripper_pos_inside_workspace(self, pos):
        inside_workspace = True
        if pos[0] < -0.35:
            inside_workspace = False
        if pos[1] > 0.35 or pos[1] < -0.35:
            inside_workspace = False
        if pos[2] < 0.85:
            inside_workspace = False
        
        return inside_workspace

    def _initialize_goal(self):
        if self.robot_configs[0]["controller_config"]["type"] == "JOINT_VELOCITY":
            # collisions = self.get_contacts(self.robots[0].robot_model)
            # hand_pose = self.robots[0]._hand_pose.copy()
            # gripper_pos_in_world, _ = self.gripper_pose_in_world(hand_pose)
            # inside_workspace = self._check_gripper_pos_inside_workspace(gripper_pos_in_world)
            # init_qpos = np.array(self.robots[0].init_qpos)
            
            # # x > -0.35, y = [-0.35, 0.35], z > 0.85 => cartesian bounding box for workspace
            # while len(collisions) != 0 or not inside_workspace:
            #     noise = np.random.randn(len(self.robots[0].init_qpos)) * self.robots[0].initialization_noise["magnitude"]

            #     init_qpos_goal = init_qpos + noise
            #     self.robots[0].sim.data.qpos[self.robots[0]._ref_joint_pos_indexes] = init_qpos_goal
            #     self.robots[0].sim.forward()
            #     collisions = self.get_contacts(self.robots[0].robot_model)
            
            #     hand_pose = self.robots[0]._hand_pose.copy()
            #     gripper_pos_in_world, _ = self.gripper_pose_in_world(hand_pose)
            #     inside_workspace = self._check_gripper_pos_inside_workspace(gripper_pos_in_world)

            # self.goal_pos = self.robots[0]._hand_pos.copy()
            # self.goal_quat = self.robots[0]._hand_quat.copy()
            # self.goal_pose = self.robots[0]._hand_pose.copy()
            # self.goal = np.concatenate((self.goal_pos,self.goal_quat), axis=0)      
            
            # # Set joint initial qpos to solve goal
            # self.robots[0].sim.data.qpos[self.robots[0]._ref_joint_pos_indexes] = init_qpos
            # self.robots[0].sim.forward()
            pos_goal = self.goal = self.goal_pos = self.np_random.uniform(
                    [-0.25,-0.25,0.83], [0.25,0.25,1.1])
            quat_goal = self.robots[0].sim.data.get_site_xmat("gripper0_grip_site").copy()
            self.goal = np.concatenate((pos_goal,quat_goal))
        elif self.robot_configs[0]["controller_config"]["type"] == "OSC_POSITION":
            # init_gripper_pos,_ = self.gripper_pose_in_world(self.robots[0]._hand_pose.copy())
            # self.init_gripper_pos = init_gripper_pos
            self.goal = self.goal_pos = self.np_random.uniform(
                    [-0.25,-0.25,0.83], [0.25,0.25,1.1])
    def gripper_pose_in_world(self, pose):

        base_pos_in_world = self.robots[0].sim.data.get_body_xpos(self.robots[0].robot_model.root_body)
        base_rot_in_world = self.robots[0].sim.data.get_body_xmat(self.robots[0].robot_model.root_body).reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        eef_goal_pose_in_world = T.pose_in_A_to_pose_in_B(pose, base_pose_in_world)
        
        eef_goal_pos_in_world = eef_goal_pose_in_world[:3, 3]
        eef_goal_orn_in_world = eef_goal_pose_in_world[:3, :3]

        return eef_goal_pos_in_world, eef_goal_orn_in_world

    def goal_euclidian_distance(self, state_A, state_B):
        return np.linalg.norm(state_A - state_B, axis=-1)

    def goal_angle_distance(self, state_A, state_B):
        if len(state_A.shape) > 1:
            error_gripper_orn = np.zeros((state_A.shape[0],))
            for i in range(state_A.shape[0]):
                error_gripper_orn[i] = abs(np.sum(T.get_orientation_error(state_A[i,:], state_B[i,:])))
        else:
            error_gripper_orn = abs(np.sum(T.get_orientation_error(state_A, state_B)))

        return error_gripper_orn
    
    def reward(self, action=None):
        if self.robot_configs[0]["controller_config"]["type"] == "JOINT_VELOCITY":
            d = self.goal_euclidian_distance(self.goal[:3], self.robots[0].sim.data.get_site_xpos("gripper0_grip_site").copy())
            d_angle = self.goal_angle_distance(self.goal[3:], self.robots[0].sim.data.get_site_xmat("gripper0_grip_site").copy())

            # Sparse reward
            return -((d > self.distance_threshold).astype(np.float32) and (d_angle > self.angle_threshold).astype(np.float32))
        elif self.robot_configs[0]["controller_config"]["type"] == "OSC_POSITION":
            d = self.goal_euclidian_distance(self.goal_pos, self.robots[0].sim.data.get_site_xpos("gripper0_grip_site").copy())
            return -(d > self.distance_threshold).astype(np.float32)
        # Dense reward
    
    def render(self, mode='human'):

        
        if self.robot_configs[0]["controller_config"]["type"] == "JOINT_VELOCITY":
            render_goal_pos, render_goal_orn = self.gripper_pose_in_world(self.goal_pose)
            self.viewer.viewer.add_marker(pos=render_goal_pos, #position of the arrow\
                        size=np.array([0.005,0.005,0.4]), #size of the arrow
                        mat=render_goal_orn, # orientation as a matrix
                        rgba=np.array([230.,0.,64.,1.]),#color of the arrow
                        type=const.GEOM_ARROW,
                        label=str('GOAL'))
        elif self.robot_configs[0]["controller_config"]["type"] == "OSC_POSITION":
            gripper_pos = self.robots[0].sim.data.get_site_xpos("gripper0_grip_site")
            self.viewer.viewer.add_marker(pos=self.goal, #position of the arrow\
                        size=np.array([0.01,0.01,0.01]), #size of the arrow
                        # mat=render_goal_orn, # orientation as a matrix
                        rgba=np.array([230.,0.,64.,1.]),#color of the arrow
                        type=const.GEOM_SPHERE,
                        label=str('GOAL'))

            self.viewer.viewer.add_marker(pos=self.robots[0].sim.data.get_site_xpos("gripper0_grip_site").copy(), #position of the arrow\
                        size=np.array([0.01,0.01,0.01]), #size of the arrow
                        # mat=render_goal_orn, # orientation as a matrix
                        rgba=np.array([0.,230.,64.,1.]),#color of the arrow
                        type=const.GEOM_SPHERE,
                        label=str('GOAL'))
                    
        super().render()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]
        

