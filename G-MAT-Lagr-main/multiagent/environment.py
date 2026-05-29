import argparse
import gym
from gym import spaces
import numpy as np
import math
import random
from typing import Callable, List, Tuple, Dict, Union, Optional
from multiagent.core import World, Agent
from multiagent.multi_discrete import MultiDiscrete
from onpolicy import global_var as glv
from .guide_policy import guide_policy, set_JS_curriculum, limit_action_inf_norm
import csv

# update bounds to center around agent
cam_range = 8
INFO = []

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentBaseEnv(gym.Env):
    """
    Base environment for all multi-agent environments
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        args: argparse.Namespace,
        world: World,
        reset_callback: Callable = None,
        reward_callback: Callable = None,
        cost_callback: Callable = None,
        observation_callback: Callable = None,
        info_callback: Callable = None,
        done_callback: Callable = None,
        shared_viewer: bool = True,
        discrete_action: bool = True,
        scenario_name: str = "navigation",
    ) -> None:
        
        self.args = args
        self.use_policy = args.use_policy
        self.gp_type = args.gp_type
        self.use_CL = args.use_curriculum
        self.CL_ratio = 0
        self.Cp = args.guide_cp
        self.js_ratio = args.js_ratio
        self.JS_thre = 0  # step of guide steps
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        self.num_agents = len(
            world.policy_agents
        )  # for compatibility with offpolicy baseline envs
        # scenario callbacks

        # terminate
        self.is_terminate = False

        # record episode information, only monte_carlo_test is True
        self.monte_carlo_test = args.monte_carlo_test
        self.round = 1
        self.last_step = 0
        self.collision_th = 1 # 2
        self.data_ = ()
        self.INFO_flag = 0
        self.collision_num = 0
        self.reward_all = np.array([0]*self.n) # for data record
        self.cost_all = np.array([0]*self.n) # for data record
        if self.args.gp_type == 'formation':
            self.formation_error = 0
        elif self.args.gp_type == 'encirclement':
            self.dist_error = 0
            self.angle_error = 0

        self.world = world
        self.world_length = self.world.world_length
        self.current_step = 0
        self.agents = self.world.policy_agents


        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.cost_callback = cost_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        self.scenario_name = scenario_name
        self.policy_u = guide_policy
        # environment parameters
        self.discrete_action_space = False
        # self.discrete_action_space = discrete_action

        # if true, action is a number 0...N,
        # otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous,
        # action will be performed discretely
        self.force_discrete_action = (
            world.discrete_action if hasattr(world, "discrete_action") else False
        )
        # if true, every agent has the same reward
        self.shared_reward = (
            world.collaborative if hasattr(world, "collaborative") else False
        )
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = (
            []
        )  # adding this for compatibility with MAPPO code
        share_obs_dim = 0
        for agent in self.agents:
            total_action_space = []

            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = MultiDiscrete([[0, 19], [0, 19]])  # -1~1, 20 values
            if agent.movable:
                total_action_space.append(u_action_space)

            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(
                    low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32
                )

            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete,
                # so simplify to MultiDiscrete action space
                if all(
                    [
                        isinstance(act_space, spaces.Discrete)
                        for act_space in total_action_space
                    ]
                ):
                    act_space = MultiDiscrete(
                        [[0, act_space.n - 1] for act_space in total_action_space]
                    )
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space
            # for original MPE Envs like simple_spread, simple_reference, etc.
            if "simple" in self.scenario_name:
                obs_dim = len(observation_callback(agent=agent, world=self.world))
            else:
                obs_dim = len(observation_callback(agent=agent, world=self.world))
            share_obs_dim += obs_dim
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32
                )
            )

            agent.action.c = np.zeros(self.world.dim_c)

        self.share_observation_space = [
            spaces.Box(
                low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32
            )
            for _ in range(self.n)
        ]

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def step(self, action_n: List):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    # get info used for benchmarking
    def _get_info(self, agent: Agent) -> Dict:
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent: Agent) -> np.ndarray:
        if self.observation_callback is None:
            return np.zeros(0)
        # for original MPE Envs like simple_spread, simple_reference, etc.
        if "simple" in self.scenario_name:
            return self.observation_callback(agent=agent, world=self.world)
        else:
            return self.observation_callback(agent=agent, world=self.world)

    # get shared observation for the environment
    def _get_shared_obs(self) -> np.ndarray:
        if self.shared_obs_callback is None:
            return None
        return self.shared_obs_callback(self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent: Agent) -> bool:
        if self.done_callback is None:
            if self.current_step >= self.world_length:
                return True
            else:
                return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent: Agent) -> float:
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # get cost for a particular agent
    def _get_cost(self, agent: Agent) -> float:
        if self.cost_callback is None:
            return 0.0
        return self.cost_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, policy_u, agent: Agent, action_space, time: Optional = None) -> None:
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index : (index + s)])
                index += s
            action = act
        else:
            action = [action]

        
        if agent.movable:
            
            action_mapping = np.linspace(-1, 1, 20)
            ux = np.dot(action[0], action_mapping)
            uy = np.dot(action[1], action_mapping)

            network_output = np.array([ux, uy])
            policy_output = (policy_u.T)[0]

            if agent.done:
                # agent decellerate to zero
                target_v = np.linalg.norm(agent.state.p_vel)
                if target_v < 1e-3:
                    acc = np.array([0,0])
                else:
                    acc = -agent.state.p_vel/target_v*agent.max_accel
                network_output[0], network_output[1] = acc[0], acc[1]
                policy_output = network_output
                agent.state.p_vel = agent.state.p_vel * 1/2
                # agent.state.p_vel = np.array([0,0])
                # print("agent done, decellerate to zero")

            if self.use_CL == True:
                if self.CL_ratio < self.Cp:
                    if self.current_step < self.JS_thre:
                        agent.action.u = policy_output
                    else:
                        agent.action.u = network_output
                else:
                    act = network_output
                    agent.action.u = limit_action_inf_norm(act, 1)
            elif self.use_policy:
                agent.action.u = policy_output
            else: 
                act = network_output
                agent.action.u = limit_action_inf_norm(act, 1) 


    def _set_CL(self, CL_ratio):
        glv.set_value('CL_ratio', CL_ratio)
        self.CL_ratio = glv.get_value('CL_ratio')

    # reset rendering assets
    def _reset_render(self) -> None:
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode: str = "human", close: bool = False) -> List:
        if self.monte_carlo_test:
            # print(self.current_step)
            # if self.current_step == self.world_length-1 and self.INFO_flag == 0: # 一个episode结束都还没有成功
            #     self.data_ = self.data_ + (self.round, int(self.is_terminate), self.current_step,)
            #     # INFO.append(data_)  # 增加行
            #     # print("round:{} current_step:{} is_terminate:{}".format(self.round, self.current_step, self.is_terminate))
            #     self.INFO_flag = 1
            #     self.round += 1
            # elif self.is_terminate == True and self.INFO_flag == 0: # 一个episode中途成功了
            #     self.data_ = self.data_ + (self.round, int(self.is_terminate), self.current_step,)
            #     if self.args.gp_type == 'encirclement':
            #         self.data_ = self.data_ + (self.angle_error/self.n, self.dist_error/self.n, )
            #     # INFO.append(data_)  # 增加行
            #     # print("round:{} current_step:{} is_terminate:{}".format(self.round, self.current_step, self.is_terminate))
            #     self.INFO_flag = 1
            #     self.round += 1

            if self.current_step == 0:
                # reset parameters (some variables also reset in the reset() function)
                self.data_ = ()
                # self.is_terminate = False
                # self.INFO_flag = 0
                # self.collision_num = 0
                # self.reward_all = np.array([0]*self.n)
                # self.cost_all = np.array([0]*self.n)
                if self.args.gp_type == 'formation':
                    self.formation_error = 0
                elif self.args.gp_type == 'encirclement':
                    self.dist_error = 0
                    self.angle_error = 0
            elif self.current_step == self.world_length-1:
                # 记录rew和num_collision
                for i, agent in enumerate(self.agents):
                    suc_ = 1 if agent.finish_step < self.world_length else 0
                    self.data_ = self.data_ + (self.round, agent.id, suc_, agent.finish_step, agent.min_dist,  agent.reward_num, agent.collision_num, )
                    INFO.append(self.data_)  # 增加行
                    self.data_ = ()

                print("round:{}".format(self.round))
                
                self.round += 1

        else:
            if close:
                # close any existic renderers
                for i, viewer in enumerate(self.viewers):
                    if viewer is not None:
                        viewer.close()
                    self.viewers[i] = None
                return []

            if mode == "human":
                alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                message = ""
                for agent in self.world.agents:
                    comm = []
                    for other in self.world.agents:
                        if other is agent:
                            continue
                        if np.all(other.state.c == 0):
                            word = "_"
                        else:
                            word = alphabet[np.argmax(other.state.c)]
                        message += other.name + " to " + agent.name + ": " + word + "   "
                # print(message)

            for i in range(len(self.viewers)):
                # create viewers (if necessary)
                if self.viewers[i] is None:
                    # import rendering only if we need it
                    # (and don't import for headless machines)
                    # from gym.envs.classic_control import rendering
                    from multiagent import rendering

                    self.viewers[i] = rendering.Viewer(700, 700)

            # create rendering geometry
            if self.render_geoms is None:
                # import rendering only if we need it
                # (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from multiagent import rendering

                self.render_geoms = []
                self.render_geoms_xform = []
                self.line = {}
                self.comm_geoms = []

                for entity in self.world.entities:
                    if entity.name=="obstacle":
                        radius = entity.R
                    else:
                        radius = entity.size
                    geom = rendering.make_circle(radius)  # drawing entity 
                    xform = rendering.Transform()

                    entity_comm_geoms = []

                    if "agent" in entity.name:
                        geom.set_color(*entity.color, alpha=0.5)

                        if not entity.silent:
                            dim_c = self.world.dim_c
                            # make circles to represent communication
                            for ci in range(dim_c):
                                comm = rendering.make_circle(entity.size / dim_c)
                                comm.set_color(1, 1, 1)
                                comm.add_attr(xform)
                                offset = rendering.Transform()
                                comm_size = entity.size / dim_c
                                offset.set_translation(
                                    ci * comm_size * 2 - entity.size + comm_size, 0
                                )
                                comm.add_attr(offset)
                                entity_comm_geoms.append(comm)

                    else:
                        geom.set_color(*entity.color)
                        if entity.channel is not None:
                            dim_c = self.world.dim_c
                            # make circles to represent communication
                            for ci in range(dim_c):
                                comm = rendering.make_circle(entity.size / dim_c)
                                comm.set_color(1, 1, 1)
                                comm.add_attr(xform)
                                offset = rendering.Transform()
                                comm_size = entity.size / dim_c
                                offset.set_translation(
                                    ci * comm_size * 2 - entity.size + comm_size, 0
                                )
                                comm.add_attr(offset)
                                entity_comm_geoms.append(comm)
                    geom.add_attr(xform)
                    self.render_geoms.append(geom)
                    self.render_geoms_xform.append(xform)
                    self.comm_geoms.append(entity_comm_geoms)

                for wall in self.world.walls:
                    corners = (
                        (wall.axis_pos - 0.5 * wall.width, wall.endpoints[0]),
                        (wall.axis_pos - 0.5 * wall.width, wall.endpoints[1]),
                        (wall.axis_pos + 0.5 * wall.width, wall.endpoints[1]),
                        (wall.axis_pos + 0.5 * wall.width, wall.endpoints[0]),
                    )
                    if wall.orient == "H":
                        corners = tuple(c[::-1] for c in corners)
                    geom = rendering.make_polygon(corners)
                    if wall.hard:
                        geom.set_color(*wall.color)
                    else:
                        geom.set_color(*wall.color, alpha=0.5)
                    self.render_geoms.append(geom)

                # add geoms to viewer
                # for viewer in self.viewers:
                #     viewer.geoms = []
                #     for geom in self.render_geoms:
                #         viewer.add_geom(geom)

                for viewer in self.viewers:
                    viewer.geoms = []
                    for geom in self.render_geoms:
                        viewer.add_geom(geom)
                    for entity_comm_geoms in self.comm_geoms:
                        for geom in entity_comm_geoms:
                            viewer.add_geom(geom)

            results = []
            for i in range(len(self.viewers)):
                from multiagent import rendering

                if self.shared_viewer:
                    pos = np.zeros(self.world.dim_p)
                else:
                    pos = self.agents[i].state.p_pos
                # self.viewers[i].set_bounds(
                #     pos[0] - cam_range,
                #     pos[0] + cam_range,
                #     pos[1] - cam_range,
                #     pos[1] + cam_range,
                # )
                self.viewers[i].set_bounds(-10, 10, -5, 15)

                # save traj data 
                # print(self.args.save_data)
                if self.args.save_data:
                    data_ = ()
                    for j, ego in enumerate(self.world.egos):
                        data_ = data_ + (j, ego.state.p_pos[0], ego.state.p_pos[1], ego.state.p_vel[0], ego.state.p_vel[1])
                    for j, dob in enumerate(self.world.dynamic_obstacles):
                        data_ = data_ + (j, dob.state.p_pos[0], dob.state.p_pos[1], dob.state.p_vel[0], dob.state.p_vel[1])
                    for j, obs in enumerate(self.world.obstacles):
                        data_ = data_ + (j, obs.state.p_pos[0], obs.state.p_pos[1], obs.R)
                    for j, target in enumerate(self.world.targets):
                        data_ = data_ + (j, target.state.p_pos[0], target.state.p_pos[1], target.state.p_vel[0], target.state.p_vel[1])
                    
                    INFO.append(data_)


                # update geometry positions
                for e, entity in enumerate(self.world.entities):
                    self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                    if "agent" in entity.name:
                        self.render_geoms[e].set_color(*entity.color, alpha=0.5)

                        if not entity.silent:
                            for ci in range(self.world.dim_c):
                                color = 1 - entity.state.c[ci]
                                self.comm_geoms[e][ci].set_color(color, color, color)
                    else:
                        self.render_geoms[e].set_color(*entity.color)
                        if entity.channel is not None:
                            for ci in range(self.world.dim_c):
                                color = 1 - entity.channel[ci]
                                self.comm_geoms[e][ci].set_color(color, color, color)

                # plot target points
                if 'navigation' in self.gp_type:
                    m = len(self.render_geoms)
                    for k, ego in enumerate(self.world.egos):
                        geom = rendering.make_moving_circle(radius=ego.R, pos=ego.goal)  # entity.size
                        geom.set_color(*ego.goal_color)
                        self.render_geoms.append(geom)
                        self.render_geoms[m+k] = self.viewers[i].draw_moving_circle(radius=ego.R, color=ego.goal_color, pos=ego.goal)

                if 'formation' in self.gp_type:
                    m = len(self.render_geoms)
                    for k, ego in enumerate(self.world.egos):
                        if ego.is_leader:
                            geom = rendering.make_moving_circle(radius=0.1, pos=ego.goal)
                            geom.set_color(*ego.goal_color)
                            self.render_geoms.append(geom)
                            self.render_geoms[m] = self.viewers[i].draw_moving_circle(radius=0.1, color=ego.goal_color, pos=ego.goal)

                m = len(self.line)
                for k, agent in enumerate(self.world.agents):
                    if not agent.done:
                        self.line[m+k] = self.viewers[i].draw_line(agent.state.p_pos, agent.state.p_pos+agent.state.p_vel*1.0)
                        self.line[m+k].set_color(*agent.color, alpha=0.5)

                # render the graph connections
                # if hasattr(self.world, "graph_mode"):
                #     if self.world.graph_mode:
                #         edge_list = self.world.edge_list.T
                #         assert edge_list is not None, "Edge list should not be None"
                #         for entity1 in self.world.entities:
                #             for entity2 in self.world.entities:
                #                 e1_id, e2_id = entity1.global_id, entity2.global_id
                #                 if e1_id == e2_id:
                #                     continue
                #                 # if edge exists draw a line
                #                 if [e1_id, e2_id] in edge_list.tolist():
                #                     src = entity1.state.p_pos
                #                     dest = entity2.state.p_pos
                #                     self.viewers[i].draw_line(start=src, end=dest)

                # render to display or array
                results.append(self.viewers[i].render(return_rgb_array=mode == "rgb_array"))

            return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent: Agent) -> List:
        receptor_type = "polar"
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == "polar":
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == "grid":
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx


class MultiAgentGraphEnv(MultiAgentBaseEnv):
    metadata = {"render.modes": ["human", "rgb_array"]}
    """
        Parameters:
        –––––––––––
        world: World
            World for the environment. Refer `multiagent/core.py`
        reset_callback: Callable
            Reset function for the environment. Refer `reset()` in 
            `multiagent/navigation_graph.py`
        reward_callback: Callable
            Reward function for the environment. Refer `reward()` in 
            `multiagent/navigation_graph.py`
        observation_callback: Callable
            Observation function for the environment. Refer `observation()` 
            in `multiagent/navigation_graph.py`
        graph_observation_callback: Callable
            Observation function for graph_related stuff in the environment. 
            Refer `graph_observation()` in `multiagent/navigation_graph.py`
        id_callback: Callable
            A function to get the id of the agent in graph
            Refer `get_id()` in `multiagent/navigation_graph.py`
        info_callback: Callable
            Reset function for the environment. Refer `info_callback()` in 
            `multiagent/navigation_graph.py`
        done_callback: Callable
            Reset function for the environment. Refer `done()` in 
            `multiagent/navigation_graph.py`
        update_graph: Callable
            A function to update the graph structure in the environment
            Refer `update_graph()` in `multiagent/navigation_graph.py`
        shared_viewer: bool
            If we want a shared viewer for rendering the environment or 
            individual windows for each agent as the ego
        discrete_action: bool
            If the action space is discrete or not
        scenario_name: str
            Name of the scenario to be loaded. Refer `multiagent/custom_scenarios.py`
    """

    def __init__(
        self,
        args: argparse.Namespace,
        world: World,
        reset_callback: Callable = None,
        reward_callback: Callable = None,
        cost_callback: Callable = None,
        observation_callback: Callable = None,
        graph_observation_callback: Callable = None,
        id_callback: Callable = None,
        info_callback: Callable = None,
        done_callback: Callable = None,
        update_graph: Callable = None,
        shared_viewer: bool = True,
        discrete_action: bool = True,
        scenario_name: str = "navigation",
    ) -> None:
        super(MultiAgentGraphEnv, self).__init__(
            args,
            world,
            reset_callback,
            reward_callback,
            cost_callback,
            observation_callback,
            info_callback,
            done_callback,
            shared_viewer,
            discrete_action,
            scenario_name,
        )
        self.update_graph = update_graph
        self.graph_observation_callback = graph_observation_callback
        self.id_callback = id_callback
        self.set_graph_obs_space()

    def set_graph_obs_space(self):
        self.node_observation_space = []
        self.adj_observation_space = []
        self.edge_observation_space = []
        self.agent_id_observation_space = []
        self.share_agent_id_observation_space = []
        num_agents = len(self.agents)
        for agent in self.agents:
            node_obs, adj = self.graph_observation_callback(agent, self.world)
            node_obs_dim = node_obs.shape  # (13, 6)
            adj_dim = adj.shape  # (13, 13)
            edge_dim = 1  # NOTE hardcoding edge dimension
            agent_id_dim = 1  # NOTE hardcoding agent id dimension
            self.node_observation_space.append(
                spaces.Box(
                    low=-np.inf, high=+np.inf, shape=node_obs_dim, dtype=np.float32
                )
            )
            self.adj_observation_space.append(
                spaces.Box(low=-np.inf, high=+np.inf, shape=adj_dim, dtype=np.float32)
            )
            self.edge_observation_space.append(
                spaces.Box(
                    low=-np.inf, high=+np.inf, shape=(edge_dim,), dtype=np.float32
                )
            )
            self.agent_id_observation_space.append(
                spaces.Box(
                    low=-np.inf, high=+np.inf, shape=(agent_id_dim,), dtype=np.float32
                )
            )
            self.share_agent_id_observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(num_agents * agent_id_dim,),
                    dtype=np.float32,
                )
            )
        
        # print("node_observation_space: ", self.node_observation_space[0])  # (13, 6)
        # print("edge_observation_space: ", self.edge_observation_space[0])  # (1, )
        # print("adj_observation_space: ", self.adj_observation_space[0])  # (13, 13)

    def step(self, action_n: List) -> Tuple[List, List, List, List, List, List, List]:
        if self.update_graph is not None:
            self.update_graph(self.world)
        self.current_step += 1
        obs_n, reward_n, cost_n, done_n, info_n = [], [], [], [], []
        node_obs_n, adj_n, agent_id_n = [], [], []
        self.world.current_time_step += 1
        self.agents = self.world.policy_agents
        self.JS_thre = int(self.world_length*self.js_ratio*set_JS_curriculum(self.CL_ratio/self.Cp, self.gp_type))

        # print("step: ", self.current_step)

        # set action for each agent
        if self.use_CL or self.use_policy:
            policy_u = self.policy_u(self.world, self.gp_type)
        else:
            policy_u = np.zeros((self.num_agents, 2, 1))
        for i, agent in enumerate(self.agents):  # adversaries only
            self._set_action(action_n[i], policy_u[i], agent, self.action_space[i])

        # advance world state
        self.world.step()

        # record observation for each agent
        done_check = []
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            agent_id_n.append(self._get_id(agent))
            node_obs, adj = self._get_graph_obs(agent)
            node_obs_n.append(node_obs)
            adj_n.append(adj)
            done_n.append(self._get_done(agent))
            reward_n.append([self._get_reward(agent)])
            cost_n.append([self._get_cost(agent)])
            done_check.append(agent.done)
            info = {}
            env_info = self._get_info(agent)
            info.update(env_info)  # nothing fancy here, just appending dict to dict
            info_n.append(info)

            if self.monte_carlo_test:
                # check collision
                for ego in self.world.egos:
                    if ego == agent: pass
                    else:
                        d_ij = np.linalg.norm(agent.state.p_pos - ego.state.p_pos)
                        d_ij_abs = d_ij - agent.R - ego.R
                        if d_ij < agent.R + ego.R:
                            self.collision_num += 1
                            agent.collision_num += 1
                        if d_ij_abs < agent.min_dist:
                            agent.min_dist = d_ij_abs
                for obs in self.world.obstacles:
                    d_ij = np.linalg.norm(agent.state.p_pos - obs.state.p_pos)
                    d_ij_abs = d_ij - agent.R - obs.R
                    if d_ij < agent.R + obs.R:
                        self.collision_num += 1
                        agent.collision_num += 1
                    if d_ij_abs < agent.min_dist:
                        agent.min_dist = d_ij_abs
                for dobs in self.world.dynamic_obstacles:
                    d_ij = np.linalg.norm(agent.state.p_pos - dobs.state.p_pos)
                    d_ij_abs = d_ij - agent.R - dobs.R
                    if d_ij < agent.R + dobs.R:
                        self.collision_num += 1
                        agent.collision_num += 1
                    if d_ij_abs < agent.min_dist:
                        agent.min_dist = d_ij_abs

                agent.reward_num += reward_n[-1][0]

                if agent.done and agent.collision_num < 1 and agent.finish_step == self.world_length: # 最后一项保证只存一次
                    agent.finish_step = self.current_step

                if self.args.gp_type == 'formation':
                    self.formation_error += self.world.formation_error
                elif self.args.gp_type == 'encirclement':
                    self.dist_error = self.world.dist_error
                    self.angle_error = self.world.angle_error

        # supervise dones number and check terminate
        terminate = []
        check_terminate = done_check if self.monte_carlo_test else done_n # for recording 
        if 'formation' in self.gp_type:
            if any(check_terminate):
                terminate = [True] * self.n
            else:
                terminate = [False] * self.n
        elif 'encirclement' in self.gp_type:
            terminate = check_terminate
            # if self.world.targets[0].done:
            #     print("step{} target done".format(self.current_step))
        elif 'navigation' in self.gp_type:
            if all(check_terminate):
                terminate = [True] * self.n
            else:
                terminate = [False] * self.n
            
        self.is_terminate = True if all(terminate) and self.collision_num/self.n <= self.collision_th else False  # col num < thre, we also think success
        
        if self.is_terminate:
            # done_n = [True] * self.n
            # this will affect the data recording
            pass

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        cost = np.sum(cost_n)
        if self.shared_reward:
            reward_n = [[reward]] * self.n  # NOTE this line is similar to PPOEnv
            cost_n = [[cost]] * self.n  # NOTE this line is similar to PPOEnv
        # self.reward_all += np.array(reward_n)  # record total reward over a trajectory
        # self.cost_all += np.array(cost_n)  # record total cost over a trajectory

        # print("reward_n: ", reward_n)
        # print("node_obs_n: ", node_obs_n[0].shape)
        # print("obs_n: ", obs_n[0].shape)
        # print("adj_n: ", adj_n[0].shape)
        # print("agent_id_n: ", agent_id_n[0].shape)

        return obs_n, agent_id_n, node_obs_n, adj_n, reward_n, cost_n, done_n, info_n

    def reset(self) -> Tuple[List, List, List, List]:
        self.current_step = 0
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n, node_obs_n, adj_n, agent_id_n = [], [], [], []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            agent_id_n.append(self._get_id(agent))
            node_obs, adj = self._get_graph_obs(agent)
            node_obs_n.append(node_obs)
            adj_n.append(adj)
            # follow is the initialization for monte carlo test
            if self.monte_carlo_test:
                agent.min_dist = 100
                agent.reward_num = 0
                agent.collision_num = 0
                agent.done = False
                agent.finish_step = self.world_length
        return obs_n, agent_id_n, node_obs_n, adj_n

    def _get_graph_obs(self, agent: Agent):
        if self.graph_observation_callback is None:
            return None, None, None
        return self.graph_observation_callback(agent, self.world)

    def _get_id(self, agent: Agent):
        if self.id_callback is None:
            return None
        return self.id_callback(agent)

'''
class MultiAgentOrigEnv(MultiAgentBaseEnv):
    metadata = {"render.modes": ["human", "rgb_array"]}
    """
        Parameters:
        –––––––––––
        world: World
            World for the environment. Refer `multiagent/core.py`
        reset_callback: Callable
            Reset function for the environment. Refer `reset()` in 
            `multiagent/navigation.py`
        reward_callback: Callable
            Reward function for the environment. Refer `reward()` in 
            `multiagent/navigation.py`
        observation_callback: Callable
            Observation function for the environment. Refer `observation()` 
            in `multiagent/navigation.py`
        info_callback: Callable
            Reset function for the environment. Refer `info_callback()` in 
            `multiagent/navigation.py`
        done_callback: Callable
            Reset function for the environment. Refer `done()` in 
            `multiagent/navigation.py`
        shared_viewer: bool
            If we want a shared viewer for rendering the environment or 
            individual windows for each agent as the ego
        discrete_action: bool
            If the action space is discrete or not
        scenario_name: str
            Name of the scenario to be loaded. Refer `multiagent/custom_scenarios.py`
    """

    def __init__(
        self,
        world: World,
        reset_callback: Callable = None,
        reward_callback: Callable = None,
        observation_callback: Callable = None,
        info_callback: Callable = None,
        done_callback: Callable = None,
        shared_viewer: bool = True,
        discrete_action: bool = True,
        scenario_name: str = "navigation",
    ) -> None:
        super(MultiAgentOrigEnv, self).__init__(
            world,
            reset_callback,
            reward_callback,
            observation_callback,
            info_callback,
            done_callback,
            shared_viewer,
            discrete_action,
            scenario_name,
        )

    def step(self, action_n: List) -> Tuple[List, List, List, List]:
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        self.world.current_time_step += 1
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward = self._get_reward(agent)
            reward_n.append(reward)
            done_n.append(self._get_done(agent))
            info = {"individual_reward": reward}
            env_info = self._get_info(agent)
            info.update(env_info)  # nothing fancy here, just appending dict to dict
            info_n.append(info)

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    def reset(self) -> Tuple[List, Union[None, np.ndarray]]:
        self.current_step = 0
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n


class MultiAgentPPOEnv(MultiAgentBaseEnv):
    metadata = {"render.modes": ["human", "rgb_array"]}
    """
        Parameters:
        –––––––––––
        world: World
            World for the environment. Refer `multiagent/core.py`
        reset_callback: Callable
            Reset function for the environment. Refer `reset()` in 
            `multiagent/navigation.py`
        reward_callback: Callable
            Reward function for the environment. Refer `reward()` in 
            `multiagent/navigation.py`
        observation_callback: Callable
            Observation function for the environment. Refer `observation()` 
            in `multiagent/navigation.py`
        info_callback: Callable
            Reset function for the environment. Refer `info_callback()` in 
            `multiagent/navigation.py`
        done_callback: Callable
            Reset function for the environment. Refer `done()` in 
            `multiagent/navigation.py`
        shared_obs_callback: Callable
            If we want to concatenate common environment state along with
            the concatenation of the indidual agent states. This will return 
            a master state of the environment. Refer 'shared_observation()` in 
            `multiagent/navigation.py`
        shared_viewer: bool
            If we want a shared viewer for rendering the environment or 
            individual windows for each agent as the ego
        discrete_action: bool
            If the action space is discrete or not
        scenario_name: str
            Name of the scenario to be loaded. Refer `multiagent/custom_scenarios.py`
    """

    def __init__(
        self,
        world: World,
        reset_callback: Callable = None,
        reward_callback: Callable = None,
        observation_callback: Callable = None,
        info_callback: Callable = None,
        done_callback: Callable = None,
        shared_viewer: bool = True,
        discrete_action: bool = True,
        scenario_name: str = "navigation",
    ) -> None:
        super(MultiAgentPPOEnv, self).__init__(
            world,
            reset_callback,
            reward_callback,
            observation_callback,
            info_callback,
            done_callback,
            shared_viewer,
            discrete_action,
            scenario_name,
        )

    def step(self, action_n: List) -> Tuple[List, List, List, List]:
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        self.world.current_time_step += 1
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward = self._get_reward(agent)
            reward_n.append(reward)
            done_n.append(self._get_done(agent))
            info = {"individual_reward": reward}
            env_info = self._get_info(agent)
            info.update(env_info)  # nothing fancy here, just appending dict to dict
            info_n.append(info)

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [
                [reward]
            ] * self.n  # NOTE this line is different compared to origEnv

        return obs_n, reward_n, done_n, info_n

    def reset(self) -> Tuple[List, Union[None, np.ndarray]]:
        self.current_step = 0
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n



class MultiAgentGPGEnv(MultiAgentGraphEnv):
    metadata = {"render.modes": ["human", "rgb_array"]}
    """ 
        Multi-agent Graph Policy Gradient Environment compatible with author's 
        official implementation: https://github.com/arbaazkhan2/gpg_labeled
    """

    def __init__(
        self,
        world: World,
        reset_callback: Callable = None,
        reward_callback: Callable = None,
        observation_callback: Callable = None,
        graph_observation_callback: Callable = None,
        id_callback: Callable = None,
        info_callback: Callable = None,
        done_callback: Callable = None,
        update_graph: Callable = None,
        shared_viewer: bool = True,
        discrete_action: bool = True,
        scenario_name: str = "navigation_gpg",
    ) -> None:
        super(MultiAgentGPGEnv, self).__init__(
            world,
            reset_callback,
            reward_callback,
            observation_callback,
            graph_observation_callback,
            id_callback,
            info_callback,
            done_callback,
            update_graph,
            shared_viewer,
            discrete_action,
            scenario_name,
        )

    def step(self, action_n: List) -> Tuple[List, List, List, List, List, List, List]:
        if self.update_graph is not None:
            self.update_graph(self.world)
        self.current_step += 1
        obs_n, adj_n, reward_n, done_n, info_n = [], [], [], [], []
        self.world.current_time_step += 1
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            node_obs, adj = self._get_graph_obs(agent)
            adj_n.append(adj)
            reward = self._get_reward(agent)
            reward_n.append(reward)
            done_n.append(self._get_done(agent))
            info = {"individual_reward": reward}
            env_info = self._get_info(agent)
            info.update(env_info)  # nothing fancy here, just appending dict to dict
            info_n.append(info)

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        reward_n = [
            reward
        ] * self.n  # NOTE this is so that all agents get the same reward for GPG
        done_n = np.array(done_n)

        # since adj and reward is same for all agents, just return adj_n[0]
        # done only if all agents are done

        return obs_n, adj_n[0], reward_n[0], done_n.all(), info_n

    def reset(self) -> Tuple[List, List, List, List]:
        self.current_step = 0
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n, adj_n = [], []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            node_obs, adj = self._get_graph_obs(agent)
            adj_n.append(adj)
        # since adj for all agents are same, only return adj_n[0]
        return obs_n, adj_n[0]


class MultiAgentCADRLEnv(MultiAgentBaseEnv):
    metadata = {"render.modes": ["human", "rgb_array"]}
    """
        Collision Avoidance with Deep RL Environment compatible with author's 
        official implementation: https://github.com/mit-acl/cadrl_ros
    """

    def __init__(
        self,
        config_args,
        phase,
        world: World,
        reset_callback: Callable = None,
        reward_callback: Callable = None,
        observation_callback: Callable = None,
        info_callback: Callable = None,
        done_callback: Callable = None,
        shared_viewer: bool = True,
        discrete_action: bool = True,
        scenario_name: str = "navigation",
    ) -> None:
        super(MultiAgentCADRLEnv, self).__init__(
            world,
            reset_callback,
            reward_callback,
            observation_callback,
            info_callback,
            done_callback,
            shared_viewer,
            discrete_action,
            scenario_name,
        )
        # self.radius = config_args.radius
        # self.v_pref = config_args.v_pref
        # self.kinematic = config_args.kinematic
        # self.agent_num = config_args.num_agents
        # self.xmin = config_args.xmin
        # self.xmax = config_args.xmax
        # self.ymin = config_args.ymin
        # self.ymax = config_args.ymax
        # self.crossing_radius = config_args.crossing_radius
        # self.max_time = config_args.max_time
        # self.agents = [None, None]
        # self.counter = 0
        assert phase in ["train", "test"]
        self.phase = phase
        # self.test_counter = 0

    def step(self, action_n: List) -> Tuple[List, List, List, List]:
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        self.world.current_time_step += 1
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward = self._get_reward(agent)
            reward_n.append(reward)
            done_n.append(self._get_done(agent))
            info = {"individual_reward": reward}
            env_info = self._get_info(agent)
            info.update(env_info)  # nothing fancy here, just appending dict to dict
            info_n.append(info)

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    # def reset(self, case=None) -> Tuple[List, Union[None, np.ndarray]]:
    #     self.current_step = 0
    #     # reset world
    #     self.reset_callback(self.world)
    #     # reset renderer
    #     self._reset_render()
    #     # record observations for each agent
    #     obs_n = []
    #     self.agents = self.world.policy_agents
    #     for agent in self.agents:
    #         obs_n.append(self._get_obs(agent))

    #     cr = self.crossing_radius
    #     self.agents[0] = CADRLAgent(-cr, 0, cr, 0, self.radius, self.v_pref, 0, self.kinematic)
    #     if self.phase == 'train':
    #         angle = random.random() * math.pi
    #         while math.sin((math.pi - angle)/2) < 0.3/2:
    #             angle = random.random() * math.pi
    #     else:
    #         if case is not None:
    #             angle = (case % 10) / 10 * math.pi
    #             self.test_counter = case
    #         else:
    #             angle = (self.test_counter % 10) / 10 * math.pi
    #             self.test_counter += 1
    #     x = cr * math.cos(angle)
    #     y = cr * math.sin(angle)
    #     theta = angle + math.pi
    #     self.agents[1] = CADRLAgent(x, y, -x, -y, self.radius, self.v_pref, theta, self.kinematic)
    #     self.counter = 0

    #     return [self.compute_joint_state(0), self.compute_joint_state(1)]
    #     # return obs_n

    def reset(self) -> Tuple[List, Union[None, np.ndarray]]:
        self.current_step = 0
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # def compute_joint_state(self, agent_idx):
    #     if self.agents[agent_idx].done:
    #         return None
    #     else:
    #         from baselines.cadrl.cadrl_navigation.utils_cadrl import JointState
    #         return JointState(*(self.agents[agent_idx].get_full_state() +
    #                           self.agents[1-agent_idx].get_observable_state()))

    # def check_boundary(self, agent_idx):
    #     agent = self.agents[agent_idx]
    #     return self.xmin < agent.px < self.xmax and self.ymin < agent.py < self.ymax

    # def compute_reward(self, agent_idx, actions):
    #     """
    #     When performing one-step lookahead, only one action is known, the position of the other agent is approximate
    #     When called by step(), both actions are known, the position of the other agent is exact
    #     """
    #     agent = self.agents[agent_idx]
    #     other_agent = self.agents[1-agent_idx]
    #     # simple collision detection is done by checking the beginning and end position
    #     dmin = float('inf')
    #     dmin_time = 1
    #     for time in [0, 0.5, 1]:
    #         pos = agent.compute_position(time, actions[agent_idx])
    #         other_pos = other_agent.compute_position(time, actions[1-agent_idx])
    #         distance = math.sqrt((pos[0]-other_pos[0])**2 + (pos[1]-other_pos[1])**2)
    #         if distance < dmin:
    #             dmin = distance
    #             dmin_time = time
    #     final_pos = agent.compute_position(1, actions[agent_idx])
    #     reached_goal = math.sqrt((final_pos[0] - agent.pgx)**2 + (final_pos[1] - agent.pgy)**2) < self.radius

    #     if dmin < self.radius * 2:
    #         reward = -0.25
    #         end_time = dmin_time
    #     else:
    #         end_time = 1
    #         if dmin < self.radius * 2 + 0.2:
    #             reward = -0.1 - dmin/2
    #         elif reached_goal:
    #             reward = 1
    #         else:
    #             reward = 0

    #     return reward, end_time


# TODO: merge env.py into CADRL MPE env here? for reset.


class MultiAgentDGNEnv(MultiAgentGraphEnv):
    metadata = {"render.modes": ["human", "rgb_array"]}
    """ 
        Multi-agent Graph Convolutional RL Environment compatible with author's 
        official implementation: https://github.com/jiechuanjiang/pytorch_DGN
    """

    def __init__(
        self,
        world: World,
        reset_callback: Callable = None,
        reward_callback: Callable = None,
        observation_callback: Callable = None,
        graph_observation_callback: Callable = None,
        id_callback: Callable = None,
        info_callback: Callable = None,
        done_callback: Callable = None,
        update_graph: Callable = None,
        shared_viewer: bool = True,
        discrete_action: bool = True,
        scenario_name: str = "navigation",
    ) -> None:
        super(MultiAgentDGNEnv, self).__init__(
            world,
            reset_callback,
            reward_callback,
            observation_callback,
            graph_observation_callback,
            id_callback,
            info_callback,
            done_callback,
            update_graph,
            shared_viewer,
            discrete_action,
            scenario_name,
        )

    def step(self, action_n: List) -> Tuple[List, List, List, List, List, List, List]:
        if self.update_graph is not None:
            self.update_graph(self.world)
        self.current_step += 1
        obs_n, adj_n, reward_n, done_n, info_n = [], [], [], [], []
        self.world.current_time_step += 1
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            node_obs, adj = self._get_graph_obs(agent)
            adj_n.append(adj)
            reward = self._get_reward(agent)
            reward_n.append(reward)
            done_n.append(self._get_done(agent))
            info = {"individual_reward": reward}
            env_info = self._get_info(agent)
            info.update(env_info)  # nothing fancy here, just appending dict to dict
            info_n.append(info)

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        reward_n = [
            reward
        ] * self.n  # NOTE this is so that all agents get the same reward
        done_n = np.array(done_n)

        # since adj and reward is same for all agents, just return adj_n[0]
        # done only if all agents are done

        return obs_n, adj_n[0], reward_n, done_n.all(), info_n

    def reset(self) -> Tuple[List, List, List, List]:
        self.current_step = 0
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n, adj_n = [], []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            node_obs, adj = self._get_graph_obs(agent)
            adj_n.append(adj)
        # since adj for all agents are same, only return adj_n[0]
        return obs_n, adj_n[0]


class MultiAgentDGN_ATOCEnv(MultiAgentGraphEnv):
    metadata = {"render.modes": ["human", "rgb_array"]}
    """ 
        Multi-agent Graph Convolutional RL Environment compatible with author's 
        official implementation: https://github.com/jiechuanjiang/pytorch_DGN
    """

    def __init__(
        self,
        world: World,
        reset_callback: Callable = None,
        reward_callback: Callable = None,
        observation_callback: Callable = None,
        graph_observation_callback: Callable = None,
        id_callback: Callable = None,
        info_callback: Callable = None,
        done_callback: Callable = None,
        update_graph: Callable = None,
        shared_viewer: bool = True,
        discrete_action: bool = True,
        scenario_name: str = "navigation",
    ) -> None:
        super(MultiAgentDGN_ATOCEnv, self).__init__(
            world,
            reset_callback,
            reward_callback,
            observation_callback,
            graph_observation_callback,
            id_callback,
            info_callback,
            done_callback,
            update_graph,
            shared_viewer,
            discrete_action,
            scenario_name,
        )

    def step(self, action_n: List) -> Tuple[List, List, List, List, List, List, List]:
        if self.update_graph is not None:
            self.update_graph(self.world)
        self.current_step += 1
        obs_n, adj_n, reward_n, done_n, info_n = [], [], [], [], []
        self.world.current_time_step += 1
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            node_obs, adj = self._get_graph_obs(agent)
            adj_n.append(adj)
            reward = self._get_reward(agent)
            reward_n.append(reward)
            done_n.append(self._get_done(agent))
            info = {"individual_reward": reward}
            env_info = self._get_info(agent)
            info.update(env_info)  # nothing fancy here, just appending dict to dict
            info_n.append(info)

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        reward_n = [
            reward
        ] * self.n  # NOTE this is so that all agents get the same reward
        done_n = np.array(done_n)

        # since adj and reward is same for all agents, just return adj_n[0]
        # done only if all agents are done

        return obs_n, adj_n[0], reward_n[0], done_n.all(), info_n

    def reset(self) -> Tuple[List, List, List, List]:
        self.current_step = 0
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n, adj_n = [], []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            node_obs, adj = self._get_graph_obs(agent)
            adj_n.append(adj)
        # since adj for all agents are same, only return adj_n[0]
        return obs_n, adj_n[0]


class MultiAgentOffPolicyEnv(MultiAgentBaseEnv):
    metadata = {"render.modes": ["human", "rgb_array"]}
    """
        This Environment is only for the off-policy baselines
        The only difference is the way in which the `rewards` and `dones` 
        are returned. Here they are returned as a list of `dones` and `rewards` 
        instead of just scalars
        Parameters:
        –––––––––––
        world: World
            World for the environment. Refer `multiagent/core.py`
        reset_callback: Callable
            Reset function for the environment. Refer `reset()` in 
            `multiagent/navigation.py`
        reward_callback: Callable
            Reward function for the environment. Refer `reward()` in 
            `multiagent/navigation.py`
        observation_callback: Callable
            Observation function for the environment. Refer `observation()` 
            in `multiagent/navigation.py`
        info_callback: Callable
            Reset function for the environment. Refer `info_callback()` in 
            `multiagent/navigation.py`
        done_callback: Callable
            Reset function for the environment. Refer `done()` in 
            `multiagent/navigation.py`
        shared_viewer: bool
            If we want a shared viewer for rendering the environment or 
            individual windows for each agent as the ego
        discrete_action: bool
            If the action space is discrete or not
    """

    def __init__(
        self,
        world: World,
        reset_callback: Callable = None,
        reward_callback: Callable = None,
        observation_callback: Callable = None,
        info_callback: Callable = None,
        done_callback: Callable = None,
        shared_viewer: bool = True,
        discrete_action: bool = True,
        scenario_name: str = "navigation",
    ) -> None:
        super(MultiAgentOffPolicyEnv, self).__init__(
            world,
            reset_callback,
            reward_callback,
            observation_callback,
            info_callback,
            done_callback,
            shared_viewer,
            discrete_action,
            scenario_name,
        )

    def step(self, action_n: List) -> Tuple[List, List, List, List]:
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        self.world.current_time_step += 1
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward = self._get_reward(agent)
            reward_n.append([reward])
            done_n.append([self._get_done(agent)])
            info = {"individual_reward": reward}
            env_info = self._get_info(agent)
            info.update(env_info)  # nothing fancy here, just appending dict to dict
            info_n.append(info)

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [[reward]] * self.n

        return obs_n, reward_n, done_n, info_n

    def reset(self) -> Tuple[List, Union[None, np.ndarray]]:
        self.current_step = 0
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n


class MultiAgentMPNNEnv(MultiAgentOrigEnv):
    metadata = {"render.modes": ["human", "rgb_array"]}
    """ 
        This Environment is only for the MPNN baselines
        discrete_action: bool
            If the action space is discrete or not
    """

    def __init__(
        self,
        world: World,
        reset_callback: Callable = None,
        reward_callback: Callable = None,
        observation_callback: Callable = None,
        info_callback: Callable = None,
        done_callback: Callable = None,
        shared_viewer: bool = True,
        discrete_action: bool = False,
        scenario_name: str = "navigation",
    ) -> None:
        super(MultiAgentMPNNEnv, self).__init__(
            world,
            reset_callback,
            reward_callback,
            observation_callback,
            info_callback,
            done_callback,
            shared_viewer,
            discrete_action,
            scenario_name,
        )
        self.discrete_action_space = True
        self.discrete_action_input = discrete_action


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {"runtime.vectorized": True, "render.modes": ["human", "rgb_array"]}

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        shared_obs_n = []
        reward_n = []
        done_n = []
        info_n = {"n": []}
        i = 0
        for env in self.env_batch:
            obs, shared_obs, reward, done, _ = env.step(action_n[i : (i + env.n)], time)
            i += env.n
            obs_n += obs
            shared_obs_n += shared_obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, shared_obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        shared_obs_n = []
        for env in self.env_batch:
            obs, shared_obs = env.reset()
            obs_n += obs
            shared_obs_n += shared_obs
        return obs_n, shared_obs

    # render environment
    def render(self, mode="human", close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n

'''