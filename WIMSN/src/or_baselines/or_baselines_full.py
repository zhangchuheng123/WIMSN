from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

import pandas as pd
import numpy as np
import pdb
import sys
import os

from ReplenishmentEnv import make_env


def Ss_policy(env, S, s):
    env.reset()
    done = False
    mean_demand = env.agent_states["demand"]
    sku_count = len(env.get_sku_list())
    total_reward = np.zeros((sku_count))
    re_fre = np.zeros((sku_count))
    reward_info = {
        "profit": np.zeros((sku_count)),
        "excess": np.zeros((sku_count)),
        "order_cost": np.zeros((sku_count)),
        "holding_cost": np.zeros((sku_count)),
        "backlog": np.zeros((sku_count))
    }
    while not done:
        action = (env.get_in_stock() + env.get_in_transit()) / (mean_demand + 0.0001)
        re_fre += np.where(action < s, 1, 0)
        action = np.where(action < s, S - action, 0)
        state, reward, done, info = env.step(action)
        mean_demand = env.get_demand_mean()
        total_reward += reward
        reward_info["profit"] += info["reward_info"]["profit"]
        reward_info["excess"] += info["reward_info"]["excess"]
        reward_info["order_cost"] += info["reward_info"]["order_cost"]
        reward_info["holding_cost"] += info["reward_info"]["holding_cost"]
        reward_info["backlog"] += info["reward_info"]["backlog"]
    GMV = sum(env.get_sale() * env.get_selling_price())
    return total_reward, re_fre, GMV, reward_info


class InventoryProblem(ElementwiseProblem):

    def __init__(self, env, sku_count, search_range, policy='ss'):

        self.my_policy = policy
        if policy == 'ss':
            n_var = 2 * sku_count
        elif policy == 'bs':
            n_var = sku_count
        self.sku_count = sku_count
        self.my_nvar = n_var
        xl = np.ones(n_var) * search_range[0]
        xu = np.ones(n_var) * search_range[1]
        super().__init__(n_var=n_var, n_obj=1, n_ieq_constr=0, xl=xl, xu=xu)
        
        self.env = env

    def _evaluate(self, x, out, *args, **kwargs):
        
        n_var = self.my_nvar
        if self.my_policy == 'ss':
            param_S = x[:n_var//2]
            param_s = x[n_var//2:]
        elif self.my_policy == 'bs':
            param_S = x 
            param_s = x

        assert param_S.shape[0] == self.sku_count
        assert param_s.shape[0] == self.sku_count

        rewards, _, _, _ = Ss_policy(self.env, param_S, param_s)
        out["F"] = - np.sum(rewards)


def search_different_Ss(env, policy='ss', initial_X=None):

    env.reset()
    sku_count = len(env.get_sku_list())

    problem = InventoryProblem(env, sku_count=sku_count, search_range=[0, 12], policy=policy)

    algorithm = GA(pop_size=10, eliminate_duplicates=True, sampling=initial_X)

    # algorithm = CMAES(x0=initial_X)

    # algorithm = ES(n_offsprings=10, rule=1.0 / 7.0, sampling=initial_X)

    # algorithm = BRKGA(
    #     pop_size=200,
    #     sampling=IntegerRandomSampling(),
    #     crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
    #     mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
    #     eliminate_duplicates=True)

    result = minimize(problem, algorithm, ('n_gen', 1000), verbose=True)

    if policy == 'ss':
        best_S = result.X[:sku_count]
        best_s = result.X[sku_count:]
    elif policy == 'bs':
        best_S = result.X 
        best_s = result.X

    rewards, _, _, _ = Ss_policy(env, best_S, best_s)

    info = {
        "reward": np.sum(rewards),
        "S": best_S,
        "s": best_s,
    }

    return info


def search_shared_Ss(env, search_range=np.arange(0.0, 12.1, 0.5)):

    state = env.reset()
    max_rewards = -np.inf
    best_S = 0
    best_s = 0
    sku_count = len(env.get_sku_list())
    for S in search_range:
        for s in np.arange(0, S + 0.1, 0.5):
            rewards, _, _, _, = Ss_policy(env, [S] * sku_count, [s] * sku_count)
            rewards = np.sum(rewards)
            if rewards > max_rewards:
                max_rewards = rewards
                best_S = S
                best_s = s

    info = {
        "reward": max_rewards,
        "S": best_S,
        "s": best_s,
    }
    return info 

def search_shared_basestock(env, search_range=np.arange(0.0, 12.1, 0.5)):

    state = env.reset()
    max_rewards = -np.inf
    best_S = 0
    sku_count = len(env.get_sku_list())
    for S in search_range:
        s = S
        rewards, _, _, _, = Ss_policy(env, [S] * sku_count, [s] * sku_count)
        rewards = np.sum(rewards)
        if rewards > max_rewards:
            max_rewards = rewards
            best_S = S

    info = {
        "reward": max_rewards,
        "basestock": best_S,
    }
    return info 

def get_task_list():
    
    sku_list = ["50", "100", "200", "500", "1000", "2307"]
    challenge_list = [
        "Standard",
        "BacklogRatioHigh", "BacklogRatioLow", "BacklogRatioMiddle", 
        "CapacityHigh", "CapacityLow", "CapacityLowest",
        "OrderCostHigh", "OrderCostHighest", "OrderCostLow",
        "StorageCostHigh", "StorageCostHighest", "StorageCostLow"
    ]

    task_list = []
    for sku in sku_list:
        for challenge in challenge_list:
            task_list.append("sku" + sku + "." + challenge)
    return task_list


if __name__ == "__main__":

    os.makedirs("output", exist_ok=True)
    os.makedirs(os.path.join("output", "or_baselines"), exist_ok=True)
    output_dir = os.path.join("output", "or_baselines")

    task_list = get_task_list()
    mode_list = ["train", "validation", "test"]

    # record = []
    # for task in task_list:
    #     for mode in mode_list:
    #         env = make_env(task, "OracleWrapper", mode)
    #         info = search_shared_Ss(env)
    #         info.update(dict(task=task, mode=mode, policy='ss'))
    #         record.append(info)
    # for task in task_list:
    #     for mode in mode_list:
    #         env = make_env(task, "OracleWrapper", mode)
    #         info = search_shared_basestock(env)
    #         info.update(dict(task=task, mode=mode, policy='bs'))
    #         record.append(info)
    # record = pd.DataFrame(record)
    # record.to_csv(os.path.join(output_dir, 'shared_parameters.csv'))

    initials = pd.read_csv(os.path.join(output_dir, 'shared_parameters.csv'), index_col=0)
    record = []
    for task in task_list:
        for mode in ["validation", "test"]:
            for policy in ['ss', 'bs']:

                env = make_env(task, "OracleWrapper", mode)
                sku_count = len(env.get_sku_list())

                initial_X = initials.loc[(initials['task'] == task) & (initials['mode'] == 'train') & (initials['policy'] == policy)]
                if policy == 'ss':
                    initial_S = [initial_X['S'].values[0]] * sku_count
                    initial_s = [initial_X['s'].values[0]] * sku_count
                elif policy == 'bs':
                    initial_S = [initial_X['basestock'].values[0]] * sku_count
                    initial_s = [initial_X['basestock'].values[0]] * sku_count

                rewards, _, _, _ = Ss_policy(env, initial_S, initial_s)
                info = dict(
                    reward = np.sum(rewards), 
                    S = initial_S,
                    s = initial_s,
                    task = task,
                    mode = mode,
                    policy = policy,
                )

                record.append(info)

                pd.DataFrame(record).to_csv(os.path.join(output_dir, 'tmp_or_baselines.csv'))

    record = pd.DataFrame(record)
    record.to_csv(os.path.join(output_dir, 'or_baselines.csv'))
