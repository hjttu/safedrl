"""
3 egos
4 obstacles
4 dynamic obstacles
"""
from typing import Optional, Tuple, List
import argparse
from numpy import ndarray as arr
import os, sys
import numpy as np
from onpolicy import global_var as glv
from scipy import sparse

sys.path.append(os.path.abspath(os.getcwd()))
from multiagent.custom_scenarios.util import *

from multiagent.core import World, Agent, Entity, Target, Obstacle, DynamicObstacle
from multiagent.scenario import BaseScenario

entity_mapping = {"agent": 0, "target": 1, "obstacle": 2, "dynamic_obstacle": 3}

class Scenario(BaseScenario):

    def __init__(self) -> None:
        super().__init__()
        self.band_init = 0.1
        self.band_target = 0.1
        self.d_lft_band = self.band_target

    def make_world(self, args: argparse.Namespace) -> World:
        # pull params from args
        self.cp = args.cp
        self.use_CL = args.use_curriculum
        self.num_egos = args.num_agents
        self.num_target = args.num_target
        self.num_obs = args.num_obstacle
        self.num_dynamic_obs = args.num_dynamic_obs
        if not hasattr(args, "max_edge_dist"):
            self.max_edge_dist = 1
            print("_" * 60)
            print(
                f"Max Edge Distance for graphs not specified. "
                f"Setting it to {self.max_edge_dist}"
            )
            print("_" * 60)
        else:
            self.max_edge_dist = args.max_edge_dist
        ####################
        world = World()
        # graph related attributes
        world.cache_dists = True  # cache distance between all entities
        world.graph_mode = True
        world.graph_feat_type = args.graph_feat_type
        world.world_length = args.episode_length
        world.collaborative = True
        self.world_size = np.sqrt(16*args.num_agents/3)

        world.max_edge_dist = self.max_edge_dist
        world.egos = [Agent() for i in range(self.num_egos)]
        world.obstacles = [Obstacle() for i in range(self.num_obs)]
        world.targets = [Target() for i in range(self.num_target)]
        world.dynamic_obstacles = [DynamicObstacle() for i in range(self.num_dynamic_obs)]
        world.agents = world.egos + world.dynamic_obstacles  # cannot add targets here, which is static
        
        # add agents
        global_id = 0
        for i, ego in enumerate(world.egos):
            ego.id = i
            ego.size = 0.1
            ego.R = ego.size
            ego.color = np.array([0.95, 0.45, 0.45])
            ego.goal_color = np.array([0.95, 0.95, 0.00])
            ego.max_speed = 2.0
            ego.max_accel = 2.0
            ego.delta = 0
            ego.global_id = global_id
            global_id += 1

        for i, target in enumerate(world.targets):
            target.id = i
            target.size = 0.1
            target.R = target.size
            target.color = np.array([0.95, 0.95, 0.00])
            target.max_speed = 0
            target.max_accel = 0
            target.global_id = global_id
            global_id += 1

        for i, obs in enumerate(world.obstacles):
            obs.id = i
            obs.size = 0.15
            obs.R = obs.size
            obs.delta = 0
            obs.color = np.array([0.45, 0.45, 0.95])
            obs.global_id = global_id
            global_id += 1

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world: World) -> None:
        # metrics to keep track of
        world.current_time_step = 0
        # to track time required to reach goal
        world.times_required = -1 * np.ones(self.num_egos)
        # track distance left to the goal
        world.dist_left_to_goal = -1 * np.ones(self.num_egos)
        # number of times agents collide with stuff
        world.num_obstacle_collisions = np.zeros(self.num_egos)
        world.num_agent_collisions = np.zeros(self.num_egos)

        # Randomly place agents and landmarks
        for obstacle in world.obstacles:
            obstacle.state.p_pos = 0.8 * np.random.uniform(
                -self.world_size / 2, self.world_size / 2, world.dim_p
            )
            obstacle.state.p_vel = np.zeros(world.dim_p)

        goal_pos = np.array([[1., 2.], [0., 2.0], [-1., 2.]])
        init_pos_ego = np.array([[-1., 0.], [0.0, 0.0], [1., 0.0]])
        init_pos_ego = init_pos_ego + np.random.randn(*init_pos_ego.shape)*0.01
        for i, ego in enumerate(world.egos):
            ego.done = False
            ego.state.p_pos = init_pos_ego[i]
            ego.state.p_vel = np.array([0.0, 0.0])
            ego.state.V = np.linalg.norm(ego.state.p_vel)
            ego.state.phi = np.pi
            ego.goal = goal_pos[i]

        for i, target in enumerate(world.targets):
            target.state.p_pos = goal_pos[i]
            target.state.p_vel = np.array([0.0, 0.0])
            target.R = 0.1
            target.delta = 0.0
            target.done = False
            target.size = 0.1

        init_pos_obs = np.array([[-1., 1.], [0.0, 1.0], [1., 1.0]])
        for i, obs in enumerate(world.obstacles):
            obs.done = False
            obs.state.p_pos = init_pos_obs[i]
            obs.state.p_vel = np.array([0.0, 0.0])
            obs.R = 0.2
            obs.delta = 0.0

        world.calculate_distances()
        self.update_graph(world)

    def set_CL(self, CL_ratio, world):
        obstacles = world.obstacles
        dynamic_obstacles = world.dynamic_obstacles
        start_CL = 0.0
        # if start_CL < CL_ratio < self.cp:
        #     for i, obs in enumerate(obstacles):
        #         obs.R = self.sizes_obs[i]*(CL_ratio-start_CL)/(self.cp-start_CL)
        #         obs.delta = 0.1*(CL_ratio-start_CL)/(self.cp-start_CL)
        #     for i, d_obs in enumerate(dynamic_obstacles):
        #         d_obs.R = d_obs.size*(CL_ratio-start_CL)/(self.cp-start_CL)
        #         d_obs.delta = 0.1*(CL_ratio-start_CL)/(self.cp-start_CL)
        #     # for i, obs in enumerate(obstacles):
        #     #     obs.R = self.sizes_obs[i]
        #     #     obs.delta = 0.1
        #     # for i, d_obs in enumerate(dynamic_obstacles):
        #     #     d_obs.R = d_obs.size
        #     #     d_obs.delta = 0.1
        # elif CL_ratio >= self.cp:
        #     for i, obs in enumerate(obstacles):
        #         obs.R = self.sizes_obs[i]
        #         obs.delta = 0.1
        #     for i, d_obs in enumerate(dynamic_obstacles):
        #         d_obs.R = d_obs.size
        #         d_obs.delta = 0.1
        # else:
        #     for i, obs in enumerate(obstacles):
        #         obs.R = 0.05
        #         obs.delta = 0.05
        #     for i, d_obs in enumerate(dynamic_obstacles):  
        #         d_obs.R = 0.05
        #         d_obs.delta = 0.05

        # for i, obs in enumerate(obstacles):
        #     obs.R = self.sizes_obs[i]
        #     obs.delta = 0.1
        # for i, d_obs in enumerate(dynamic_obstacles):
        #     d_obs.R = d_obs.size
        #     d_obs.delta = 0.1

        # if CL_ratio < self.cp:
        #     self.d_lft_band = self.band_init - (self.band_init - self.band_target)*CL_ratio/self.cp
        # else:
        #     self.d_lft_band = self.band_target

    def info_callback(self, agent: Agent, world: World) -> Tuple:
        if agent.collide:
            if self.is_obstacle_collision(agent.state.p_pos, agent.R, world):
                world.num_obstacle_collisions[agent.id] += 1
            for a in world.agents:
                if a is agent:
                    continue
                if self.is_collision(agent, a):
                    world.num_agent_collisions[agent.id] += 1

        agent_info = {
            "Dist_to_goal": world.dist_left_to_goal[agent.id],
            "Time_req_to_goal": world.times_required[agent.id],
            "Num_agent_collisions": world.num_agent_collisions[agent.id],
            "Num_obst_collisions": world.num_obstacle_collisions[agent.id],
        }

        return agent_info


    # check collision of entity with obstacles
    def is_obstacle_collision(self, pos, entity_size: float, world: World) -> bool:
        # pos is entity position "entity.state.p_pos"
        collision = False
        for obstacle in world.obstacles:
            delta_pos = obstacle.state.p_pos - pos
            dist = np.linalg.norm(delta_pos)
            dist_min = obstacle.R + entity_size
            if dist < dist_min:
                collision = True
                break
        return collision

    def check_agent_collision(self, pos, agent_size, agent_added) -> bool:
        collision = False
        if len(agent_added):
            for agent in agent_added:
                delta_pos = agent.state.p_pos - pos
                dist = np.linalg.norm(delta_pos)
                if dist < (agent.size + agent_size):
                    collision = True
                    break
        return collision

    def is_target_collision(self, pos, size: float, target_list: list) -> bool:
        collision = False
        for target in target_list:
            delta_pos = target.state.p_pos - pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = size + target.size
            if dist < dist_min:
                collision = True
                break
        return collision

    # check collision of agent with another agent
    def is_collision(self, agent1: Agent, agent2: Agent) -> bool:
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.linalg.norm(delta_pos)
        dist_min = agent1.R + agent2.R + (agent1.delta + agent2.delta)*0.2
        return True if dist < dist_min else False

    # done condition for each agent
    def done(self, agent: Agent, world: World) -> bool:
        dist = np.linalg.norm(agent.state.p_pos - agent.goal)
        if dist < self.d_lft_band:
            agent.done = True
            return True
        else:  agent.done = False
        return False

    def cost(self, agent: Agent, world: World) -> float:
        # the cost is the number of collisions with obstacles and other agents
        # similar to Num_agent_collisions in info_callback
        if self.use_CL:
            self.set_CL(glv.get_value('CL_ratio'), world)
        cost = 0.0
        if agent.collide:
            # cost for colliding with obstacles
            for obstacle in world.obstacles:
                delta_pos = obstacle.state.p_pos - agent.state.p_pos
                dist = np.linalg.norm(delta_pos)
                dist_min = obstacle.R + agent.R
                if dist < dist_min:
                    cost += 1.0
            # cost for colliding with other agents
            for a in world.agents:
                if a is agent:
                    continue
                if self.is_collision(agent, a):
                    cost += 1.0
        
        # if abs(agent.state.p_pos[0]) > self.world_size/2 or abs(agent.state.p_pos[1]) > self.world_size/2:
        #     cost += 1.0

        return cost

    def reward(self, agent: Agent, world: World) -> float:
        if self.use_CL:
            self.set_CL(glv.get_value('CL_ratio'), world)
        
        egos = world.egos
        obstacles = world.obstacles
        # dynamic_obstacles = world.dynamic_obstacles
        
        
        goal = world.targets[agent.id]
        rew = 0
        penalty = 2
        # collision_flag = False
        # for ego in egos:
        #     if ego == agent: pass
        #     else:
        #         if self.is_collision(agent, ego):
        #             rew += -1*penalty
        #             collision_flag = True
        # for obs in obstacles:
        #     if self.is_collision(agent, obs):
        #         rew += -1*penalty
        #         collision_flag = True

        
        # if abs(agent.state.p_pos[0]) > self.world_size/1.8 or abs(agent.state.p_pos[1]) > self.world_size/1.8:
        #     rew -= 2.0

        dist_to_goal = np.sqrt(
            np.sum(np.square(agent.state.p_pos - goal.state.p_pos))
        )
        if dist_to_goal < agent.size:
            rew += 10
            agent.done = True
        else:
            rew -= dist_to_goal

        # agent.done = self.done(agent, world)
        # print("step:", world.world_step)
        # print("ego id:", agent.id, "collide:", collision_flag)
        '''
        cost = 0.0
        if agent.collide:
            # cost for colliding with obstacles
            for obstacle in world.obstacles:
                delta_pos = obstacle.state.p_pos - agent.state.p_pos
                dist = np.linalg.norm(delta_pos)
                dist_min = obstacle.R + agent.R
                if dist < dist_min:
                    cost += 1.0
            # cost for colliding with other agents
            for a in world.agents:
                if a is agent:
                    continue
                if self.is_collision(agent, a):
                    cost += 1.0
        rew = -cost
        '''

        return rew

    def observation(self, agent: Agent, world: World) -> arr:
        """
        Return:
            [agent_pos, agent_vel, goal_pos]
        """
        goal_pos = []
        id = agent.id
        goal_pos.append(world.targets[id].state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_pos, agent.state.p_vel] + goal_pos)  # dim = 6

    def get_id(self, agent: Agent) -> arr:
        return np.array([agent.global_id])

    def graph_observation(self, agent: Agent, world: World) -> Tuple[arr, arr]:
        num_entities = len(world.entities)
        # node observations
        node_obs = []
        if world.graph_feat_type == "global":
            for i, entity in enumerate(world.entities):
                node_obs_i = self._get_entity_feat_global(entity, world)
                node_obs.append(node_obs_i)
        elif world.graph_feat_type == "relative":
            for i, entity in enumerate(world.entities):
                node_obs_i = self._get_entity_feat_relative(agent, entity, world)
                node_obs.append(node_obs_i)
        else:
            raise ValueError(f"Graph Feature Type {world.graph_feat_type} not supported")

        node_obs = np.array(node_obs)
        adj = world.cached_dist_mag

        return node_obs, adj

    def update_graph(self, world: World):
        """
        Construct a graph from the cached distances.
        Nodes are entities in the environment
        Edges are constructed by thresholding distances
        """
        dists = world.cached_dist_mag
        # just connect the ones which are within connection
        # distance and do not connect to itself
        connect = np.array((dists <= self.max_edge_dist) * (dists > 0)).astype(int)
        sparse_connect = sparse.csr_matrix(connect)
        sparse_connect = sparse_connect.tocoo()
        row, col = sparse_connect.row, sparse_connect.col
        edge_list = np.stack([row, col])
        world.edge_list = edge_list
        if world.graph_feat_type == "global":
            world.edge_weight = dists[row, col]
        elif world.graph_feat_type == "relative":
            world.edge_weight = dists[row, col]
        
        # print(f"Edge List: {len(world.edge_list.T)}")
        # print(f"Edge List: {world.edge_list}")
        # print(f"cached_dist_vect: {world.cached_dist_vect}")
        # print(f"cached_dist_mag: {world.cached_dist_mag}")

    def _get_entity_feat_global(self, entity: Entity, world: World) -> arr:
        """
        Returns: ([velocity, position, goal_pos, entity_type])
        in global coords for the given entity
        """
        if self.use_CL:
            self.set_CL(glv.get_value('CL_ratio'), world)

        pos = entity.state.p_pos
        vel = entity.state.p_vel
        Radius = entity.R
        id = entity.id
        if "agent" in entity.name:
            entity_type = entity_mapping["agent"]
        elif "target" in entity.name:
            entity_type = entity_mapping["target"]
        elif "dynamic_obstacle" in entity.name:
            entity_type = entity_mapping["dynamic_obstacle"]
        elif "obstacle" in entity.name:
            entity_type = entity_mapping["obstacle"]
        else:
            raise ValueError(f"{entity.name} not supported")

        return np.hstack([pos, vel, entity_type])

    def _get_entity_feat_relative(self, agent: Agent, entity: Entity, world: World) -> arr:
        """
        Returns: ([relative_velocity, relative_position, entity_type])
        in relative coords for the given entity
        """
        if self.use_CL:
            self.set_CL(glv.get_value('CL_ratio'), world)

        pos = agent.state.p_pos - entity.state.p_pos
        vel = agent.state.p_vel - entity.state.p_vel
        Radius = entity.R
        id = entity.id
        if "agent" in entity.name:
            entity_type = entity_mapping["agent"]
        elif "target" in entity.name:
            entity_type = entity_mapping["target"]
        elif "obstacle" in entity.name:
            entity_type = entity_mapping["obstacle"]
        else:
            raise ValueError(f"{entity.name} not supported")

        return np.hstack([pos, vel, entity_type])  # dim = 5

def dobs_policy(agent, obstacles, dobs):
    action = agent.action
    dt = 0.1
    if agent.t > 20:
        agent.done = True
    if agent.done:
        target_v = np.linalg.norm(agent.state.p_vel)
        if target_v < 1e-3:
            acc = np.array([0,0])
        else:
            acc = -agent.state.p_vel/target_v*agent.max_accel
        a_x, a_y = acc[0], acc[1]
        v_x = agent.state.p_vel[0] + a_x*dt
        v_y = agent.state.p_vel[1] + a_y*dt
        escape_v = np.array([v_x, v_y])
    else:
        max_speed = agent.max_speed
        esp_direction = agent.direction/np.linalg.norm(agent.direction)

        # with obstacles
        d_min = 1.0  # only consider the nearest obstacle, within 1.0m
        for obs in obstacles:
            dist_ = np.linalg.norm(agent.state.p_pos - obs.state.p_pos)
            if dist_ < d_min:
                d_min = dist_
                nearest_obs = obs
        if d_min < 1.0:
            d_vec_ij = agent.state.p_pos - nearest_obs.state.p_pos
            d_vec_ij = 0.5 * d_vec_ij / np.linalg.norm(d_vec_ij) / (np.linalg.norm(d_vec_ij) - nearest_obs.R - agent.R)
            if np.dot(d_vec_ij, esp_direction) < 0:
                d_vec_ij = d_vec_ij - np.dot(d_vec_ij, esp_direction) / np.dot(esp_direction, esp_direction) * esp_direction
        else:
            d_vec_ij = np.array([0, 0])
        esp_direction = esp_direction + d_vec_ij

        for dob in dobs:
            if dob == agent:
                continue
            else:
                dist = np.linalg.norm(agent.state.p_pos - dob.state.p_pos)
                if dist - agent.R - dob.R < 0.5:
                    d_vec_ij = agent.state.p_pos - dob.state.p_pos
                    d_vec_ij = 0.5 * d_vec_ij / np.linalg.norm(d_vec_ij) / (np.linalg.norm(d_vec_ij) - dob.R - agent.R)
                    if np.dot(d_vec_ij, esp_direction) < 0:
                        d_vec_ij = d_vec_ij - np.dot(d_vec_ij, esp_direction) / np.dot(esp_direction, esp_direction) * esp_direction
                    esp_direction = esp_direction + d_vec_ij

        # with dynamic obstacles

        esp_direction = esp_direction/np.linalg.norm(esp_direction)
        a_x, a_y = esp_direction[0]*agent.max_accel, esp_direction[1]*agent.max_accel
        v_x = agent.state.p_vel[0] + a_x*dt
        v_y = agent.state.p_vel[1] + a_y*dt
        # 检查速度是否超过上限
        if abs(v_x) > max_speed:
            v_x = max_speed if agent.state.p_vel[0]>0 else -max_speed
        if abs(v_y) > max_speed:
            v_y = max_speed if agent.state.p_vel[1]>0 else -max_speed
        escape_v = np.array([v_x, v_y])

        # print("exp_direction:", esp_direction)

    action.u = escape_v
    return action