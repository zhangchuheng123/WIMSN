import numpy as np 
import gym
import scipy.stats as st
from gym import spaces
from typing import Tuple
from ReplenishmentEnv.utility.simulation_tracker import SimulationTracker

"""
    ObservationWrapper4OldCode can generate more information state, which can help the training of RL algorithm. 
    And ObservationWrapper4OldCode can be used for old algo code https://github.com/songCNMS/replenishment-marl-baselines.
"""
class ObservationWrapper4OldCode(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.env = env
        self.n_agents = len(self.env.sku_list)

        self.column_lens = dict()
        self.column_lens["demand_hist"] = self.env.lookback_len
        self.column_lens["hist_order"] = self.env.lookback_len 
        self.column_lens["intransit_hist_sum"] = self.env.lookback_len
        
        self.state_dim = 0
        self.state_info = ["local_info", "global_info"]# "rank_info", "mean_field"]
        if "local_info" in self.state_info:
            self.local_info_dim = (16 + self.column_lens["demand_hist"] + self.column_lens["hist_order"])
            self.state_dim += self.local_info_dim
        if "mean_field" in self.state_info:
            self.state_dim += self.local_info_dim
        if "global_info" in self.state_info:
            self.state_dim += (7 + self.column_lens["intransit_hist_sum"])
        if "rank_info" in self.state_info:
            self.state_dim += 7

        self.state = np.zeros((len(self.env.sku_list), self.state_dim))

        # Modify action_space and observation_space to match old code.
        self.agent_observation_space = spaces.Box(
            low=-5000.00, high=5000.00, shape=(self.state_dim,), dtype=np.float64
        )
        self.observation_space = [self.agent_observation_space] * len(self.env.sku_list)
        self.agent_action_space = spaces.Discrete(34)
        self.action_space = [self.agent_action_space] * len(self.env.sku_list)

        # add mode to match old code.
        self.mode = "train"
        self.max_in_stock_sum = 0

        self.tracker = SimulationTracker(self.n_agents, self.env.picked_start_date, self.env.sku_list)

    """
        Generate the state of all skus, called state_v1.
        Returns:
        state_v1[np.array] : (n_agents, n_dim) The state of all skus, including global information
         and local information.
    """
    def get_state_v1(self) -> np.array:
        demand_mean = np.average(self.env.agent_states["demand", "lookback_with_current"].transpose(), 1)

        state_normalize = demand_mean + 1
        state_normalize_reshape = state_normalize.reshape((-1,1))
        price_normalize = self.env.agent_states['selling_price']

        state_list = []

        if "local_info" in self.state_info:
            # is_out_of_stock
            state_list.append(np.where(self.env.agent_states["in_stock"] <= 0, 1.0, 0.0)[:, np.newaxis])

            # inventory_in_stock
            state_list.append((self.env.agent_states["in_stock"] / state_normalize)[:, np.newaxis])

            # inventory_in_transit
            state_list.append((self.env.agent_states["in_transit"] / state_normalize)[:, np.newaxis])

            # inventory_estimated
            inventory_estimated = self.env.agent_states["in_stock"] + self.env.agent_states["in_transit"]
            state_list.append((inventory_estimated / state_normalize)[:, np.newaxis])

            # inventory_rop
            sale_mean = np.mean(self.env.agent_states["sale", "lookback_with_current"].transpose(), axis=1)
            sale_std = np.std(self.env.agent_states["sale", "lookback_with_current"].transpose(), axis=1)
            inventory_rop = (
                self.env.agent_states["vlt"] * sale_mean
                + np.sqrt(self.env.agent_states["vlt"])
                * sale_std
                * st.norm.ppf(0.95) # service_levels
            )
            state_list.append((inventory_rop / state_normalize)[:, np.newaxis])

            # is_below_rop
            state_list.append(np.where(inventory_estimated <= inventory_rop, 1.0, 0.0)[:, np.newaxis])

            # demand_std
            state_list.append((np.std(self.env.agent_states["demand", "lookback_with_current"].transpose()
                                    , axis=1) / state_normalize)[:, np.newaxis])

            # demand_hist
            state_list.append(self.env.agent_states["demand", "lookback_with_current"].transpose() / \
                            state_normalize_reshape)

            # capacity
            state_list.append((demand_mean / self.env.storage_capacity)[:, np.newaxis])

            # sku_price
            state_list.append((self.env.agent_states['selling_price'] / price_normalize)[:, np.newaxis])

            # sku_cost
            state_list.append((self.env.agent_states['procurement_cost'] / price_normalize)[:, np.newaxis])
            
            # sku_profit
            sku_profit = self.env.agent_states['selling_price'] - self.env.agent_states['procurement_cost']
            state_list.append((sku_profit / price_normalize)[:, np.newaxis])

            # holding_cost
            holding_cost = (self.env.agent_states["storage_cost"] + self.env.agent_states['selling_price']
                        * self.env.agent_states['holding_cost_ratio'])
            state_list.append((holding_cost / price_normalize)[:, np.newaxis])

            # order_cost
            state_list.append((self.env.agent_states['order_cost'] / price_normalize)[:, np.newaxis])

            # vlt
            state_list.append(self.env.agent_states['vlt'][:, np.newaxis])

            # vlt_demand_mean
            state_list.append(((demand_mean * (self.env.agent_states['vlt'] + 1)) / state_normalize)[:, np.newaxis])

            # vlt_day_remain
            state_list.append(((inventory_estimated - demand_mean * 
                                (self.env.agent_states['vlt'] + 1)) / state_normalize)[:, np.newaxis])

            # hist_order
            state_list.append(self.env.agent_states["replenish", "lookback_with_current"].transpose() / \
                            state_normalize_reshape)

        if "global_info" in self.state_info:
            # in_stock_sum
            state_list.append((np.ones(self.n_agents) * self.env.agent_states["in_stock"].sum() / self.env.storage_capacity)[:, np.newaxis])

            # in_stock_profit
            state_list.append((np.ones(self.n_agents) * (self.env.agent_states["in_stock"] * sku_profit).sum() / \
                        ((self.env.agent_states["in_stock"].sum() + 1) * price_normalize))[:, np.newaxis])

            # remain_capacity
            state_list.append((np.ones(self.n_agents) * (self.env.storage_capacity - self.env.agent_states["in_stock"].sum()) /
                                            self.env.storage_capacity)[:, np.newaxis])

            # intransit_sum
            state_list.append((np.ones(self.n_agents) * self.env.agent_states["in_transit"].sum() / self.env.storage_capacity)[:, np.newaxis])

            # intransit_hist_sum
            state_list.append(self.env.agent_states["replenish", "lookback_with_current"].transpose() / \
                            self.env.storage_capacity)

            # intransit_profit
            state_list.append((np.ones(self.n_agents) * (self.env.agent_states["in_transit"] * sku_profit).sum() / \
                        ((self.env.agent_states["in_transit"].sum() + 1) * price_normalize))[:, np.newaxis])

            # instock_intransit_sum
            state_list.append((np.ones(self.n_agents) * inventory_estimated.sum() / self.env.storage_capacity)[:, np.newaxis])

            # instock_intransit_profit
            state_list.append((np.ones(self.n_agents) * (inventory_estimated * sku_profit).sum() / \
                        ((inventory_estimated.sum() + 1) * price_normalize))[:, np.newaxis])

        # Rank infos
        if "rank_info" in self.state_info:
            state_list.append((self.env.agent_states["in_stock"].argsort() / self.n_agents)[:, np.newaxis])
            state_list.append((self.env.agent_states["in_transit"].argsort() / self.n_agents)[:, np.newaxis])
            state_list.append((inventory_estimated.argsort() / self.n_agents)[:, np.newaxis])
            state_list.append((demand_mean.argsort() / self.n_agents)[:, np.newaxis])
            state_list.append((sku_profit.argsort() / self.n_agents)[:, np.newaxis])
            state_list.append((self.env.agent_states["selling_price"].argsort() / self.n_agents)[:, np.newaxis])
            state_list.append((self.env.agent_states["procurement_cost"].argsort() / self.n_agents)[:, np.newaxis])

        state = np.concatenate(state_list, axis = -1)

        if "mean_field" in self.state_info:
            mean_info = state[:, :self.local_info_dim].mean(axis = 0, keepdims = True)
            mean_info = np.tile(mean_info, (self.n_agents, 1))
            state = np.concatenate([state, mean_info], axis = -1)
        # state = np.nan_to_num(state)
        return state

    """
        Step orders: Replenish -> Sell -> Receive arrived skus -> Update balance
        actions: [action_idx/action_quantity] by sku order, defined by action_setting in config
        Returns:
        state_v1[np.array] : (n_agents, n_dim) The state of all skus, including global information
         and local information.
    """
    def step(self, actions: np.array) -> Tuple[np.array, np.array, list, dict]:
        _, rewards, done, infos = self.env.step(actions)
        self.env.pre_step()
        states = self.get_state_v1()
        self.max_in_stock_sum = max(self.max_in_stock_sum, self.env.agent_states["in_stock"].sum())
        self.env.next_step()

        infos['cur_balance'] = self.env.per_balance.copy()
        infos['max_in_stock_sum'] = self.max_in_stock_sum
        return states, rewards, done, infos

    """
        Update the self.config by update_config
        All items except sku data can be updated.
        To avoid the obfuscation, update_config is only needed when reset with update
    """
    def reset(self, update_config: dict = None) -> None:
        self.max_in_stock_sum = 0
        self.env.reset(update_config)
        self.env.pre_step()
        states = self.get_state_v1()
        self.env.next_step()
        return states

    """
        switch mode(train/test) for old code.
    """
    def switch_mode(self, mode: str) -> None:
        self.mode = mode
    
    """
        get profit for old code.
    """
    def get_profit(self) -> int:
        return self.env.balance
    

    """
        use pyecharts to visualize monitors:
        args:
            visual_outputs_path (str) : save path.
    """
    def render(self, visual_outputs_path: str) -> None:
        states4render = self.env.agent_states.states[:, self.env.lookback_len : self.env.lookback_len + 
            self.env.current_step].copy()
        self.tracker.render_sku(states4render, self.env.agent_states.states_items, self.env.sku_monitor, 
            self.env.reward_monitor, visual_outputs_path)
