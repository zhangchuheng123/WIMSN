import re
import gym
import numpy as np
import datetime
import sys
from gym import ObservationWrapper, spaces
from gym.spaces import flatdim
from gym.wrappers import TimeLimit as GymTimeLimit
from ..multiagentenv import MultiAgentEnv
from ReplenishmentEnv import make_env
from ReplenishmentEnv.wrapper.observation_wrapper_for_old_code import ObservationWrapper4OldCode
from typing import Tuple
import torch
import numpy as np
import pdb


class OldRLWrapper(ObservationWrapper4OldCode):

    def reset(self, **kwargs):
        self.in_stock_sum = []
        return self.env.reset(**kwargs)

    def step(self, actions: np.array) -> Tuple[np.array, np.array, list, dict]:

        _, rewards, done, infos = self.env.step(actions)
        self.env.pre_step()
        states = self.get_state_v1()
        self.in_stock_sum.append(self.env.agent_states["in_stock"].sum())
        self.env.next_step()

        infos['cur_balance'] = self.env.per_balance.copy()
        if len(self.in_stock_sum) > 40:
            infos['max_in_stock_sum'] = np.max(self.in_stock_sum[20:])
            infos['mean_in_stock_sum'] = np.mean(self.in_stock_sum[20:])
        else:
            infos['max_in_stock_sum'] = np.max(self.in_stock_sum)
            infos['mean_in_stock_sum'] = np.mean(self.in_stock_sum)

        return states, rewards, done, infos


class LambdaRLWrapper(ObservationWrapper4OldCode):

    def reset(self, **kwargs):
        self.in_stock_sum = []
        return self.env.reset(**kwargs)

    def step(self, actions: np.array) -> Tuple[np.array, np.array, list, dict]:

        _, rewards, done, infos = self.env.step(actions)
        self.env.pre_step()
        states = self.get_state_v1()

        self.in_stock_sum.append(self.env.agent_states["in_stock"].sum())
        self.env.next_step()

        infos['cur_balance'] = self.env.per_balance.copy()
        if len(self.in_stock_sum) > 40:
            infos['max_in_stock_sum'] = np.max(self.in_stock_sum[20:])
            infos['mean_in_stock_sum'] = np.mean(self.in_stock_sum[20:])
        else:
            infos['max_in_stock_sum'] = np.max(self.in_stock_sum)
            infos['mean_in_stock_sum'] = np.mean(self.in_stock_sum)

        # additional return in_stocks in reward_info
        rewards = rewards + infos["reward_info"]["excess"] + infos["reward_info"]["holding_cost"]

        # rewards.shape is 100 (n_agents)
        max_lamdba = 10.0
        min_lambda = 0.0
        n_lambda = 51

        rewards_list = []
        for i in range(n_lambda):
            lamdba = min_lambda + i * (max_lamdba - min_lambda) / (n_lambda - 1)
            lambda_instocks = lamdba * infos["reward_info"]["in_stocks"]
            rewards_ = rewards - lambda_instocks
            rewards_list.append(rewards_)

        lamdba_rewards = np.stack(rewards_list, axis=1)

        return states, lamdba_rewards, done, infos


class SingleLambdaRLWrapper(ObservationWrapper4OldCode):

    def reset(self, lbda=None, **kwargs):
        self.in_stock_sum = []
        self.lbda = lbda
        return self.env.reset(**kwargs)

    def step(self, actions: np.array) -> Tuple[np.array, np.array, list, dict]:

        _, rewards, done, infos = self.env.step(actions)
        self.env.pre_step()
        states = self.get_state_v1()

        self.in_stock_sum.append(self.env.agent_states["in_stock"].sum())
        self.env.next_step()

        infos['cur_balance'] = self.env.per_balance.copy()
        if len(self.in_stock_sum) > 40:
            infos['max_in_stock_sum'] = np.max(self.in_stock_sum[20:])
            infos['mean_in_stock_sum'] = np.mean(self.in_stock_sum[20:])
        else:
            infos['max_in_stock_sum'] = np.max(self.in_stock_sum)
            infos['mean_in_stock_sum'] = np.mean(self.in_stock_sum)

        # Get rid of excess cost and holding cost
        rewards = rewards + infos["reward_info"]["excess"] + infos["reward_info"]["holding_cost"]
        # Add lambda cost
        rewards = rewards - self.lbda * infos["reward_info"]["in_stocks"]

        return states, rewards, done, infos


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = len(observation) * [True]
        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class ReplenishmentEnv(MultiAgentEnv):
    def __init__(
        self,
        n_agents = 100,
        task_type = "Standard",
        mode = "train",
        time_limit=1460,
        reward="old",
        **kwargs,
    ):

        env_base = make_env("sku{}.{}".format(n_agents, task_type),
            wrapper_name = "ObservationWrapper4OldCode", mode = mode)
        if reward == "old":
            env_base = OldRLWrapper(env_base)
        elif reward == "lambda":
            env_base = LambdaRLWrapper(env_base)
        elif reward == "single_lambda":
            env_base = SingleLambdaRLWrapper(env_base)

        sampler_seq_len = env_base.config['env']['horizon']
        self.episode_limit = min(time_limit, sampler_seq_len)
        action_space = [0.00, 0.16, 0.33, 0.40, 0.45, 0.50, 0.55, 0.60, 0.66, 0.83, 
                        1.00, 1.16, 1.33, 1.50, 1.66, 1.83, 2.00, 2.16, 2.33, 2.50, 
                        2.66, 2.83, 3.00, 3.16, 3.33, 3.50, 3.66, 3.83, 4.00, 5.00, 
                        6.00, 7.00, 9.00, 12.00]
        update_config = {"action": {"mode": "demand_mean_discrete", "space": action_space}} 
        env_base.reset(update_config=update_config)

        self._env = TimeLimit(env_base, max_episode_steps=sampler_seq_len)
        self._env = FlattenObservation(self._env)

        self.n_agents = self._env.n_agents
        self._obs = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = kwargs["seed"]

    def step(self, actions):
        """Returns reward, terminated, info"""
        actions = [int(a) for a in actions]
        self._obs, reward, done, info = self._env.step(actions)
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]

        if self.n_agents > 1:
            individual_rewards = np.array(reward).astype(np.float32) / 1e4
        else:
            individual_rewards = np.array([reward,]).astype(np.float32) / 1e4


        return (
            sum(reward) / 1e4,
            done, 
            {
                "individual_rewards": individual_rewards,
                "cur_balance": info['cur_balance'],
                "max_in_stock_sum": info['max_in_stock_sum'],
                "mean_in_stock_sum": info['mean_in_stock_sum'],
            },
        )

    def get_storage_capacity(self):
        return self._env.get_storage_capacity()

    def set_storage_capacity(self, storage_capacity):
        self._env.set_storage_capacity(storage_capacity)

    def get_obs(self):
        """Returns all agent observations in a list"""
        assert not np.isnan(self._obs).any()
        return self._obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        raise self._obs[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        raise self._obs[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self, **kwargs):
        """Returns initial observations and states"""
        self._obs = self._env.reset(**kwargs)
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}

    def switch_mode(self, mode):
        self._env.switch_mode(mode)

    def get_profit(self):
        profit = self._env.per_balance.copy()
        return profit

    def set_local_SKU(self, local_SKU):
        self._env.set_local_SKU(local_SKU)
        self.n_agents = self._env.n_agents
        self._obs = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )
    
    def visualize_render(self,visual_outputs_path):
        return self._env.render(visual_outputs_path)
