import numpy as np 
import gym
import scipy.stats as st
from typing import Tuple


"""
    ObservationWrapper can generate more information about the state, which
    can help the training of the RL algorithm.
"""
class ObservationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.env = env
        self.n_agents = len(self.env.sku_list)
        
        self.column_lens = dict()
        self.column_lens["demand_hist"] = self.env.lookback_len
        self.column_lens["hist_order"] = self.env.lookback_len 
        self.column_lens["intransit_hist_sum"] = self.env.lookback_len
    

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
        # hist_order = np.zeros((self.n_agents, self.column_lens["hist_order"]))
        # hist_len_cur = min(self.column_lens["hist_order"], self.env.current_step + 1)
        # hist_order[:, :hist_len_cur] = 
        state_list.append(self.env.agent_states["replenish", "lookback_with_current"].transpose() / \
                        state_normalize_reshape)

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

        state = np.concatenate(state_list, axis = -1)
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
        self.env.next_step()

        return states, rewards, done, infos


    """
        Update the self.config by update_config
        All items except sku data can be updated.
        To avoid the obfuscation, update_config is only needed when reset with update.
        To avoid nan in state, we return yesterday state in reset. It's different with `reset` in base env.
    """
    def reset(self, update_config:dict = None) -> None:
        self.env.reset(update_config)
        self.env.pre_step()
        states = self.get_state_v1()
        self.env.next_step()
        return states