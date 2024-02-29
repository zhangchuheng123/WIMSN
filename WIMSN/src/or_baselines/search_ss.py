import os
import numpy as np
import sys
import pandas as pd

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


def search_shared_Ss(env, S_range):

    state = env.reset()
    max_rewards = -np.inf
    best_S = 0
    best_s = 0
    sku_count = len(env.get_sku_list())
    for S in S_range:
        for s in np.arange(0, S + 0.1, 0.5):
            rewards = sum(Ss_policy(env, [S] * sku_count, [s] * sku_count))
            if rewards > max_rewards:
                max_rewards = rewards
                best_S = S
                best_s = s
    return max_rewards, best_S, best_s

def search_independent_Ss(env, search_range=np.arange(0.0, 12.1, 0.5)):

    env.reset()
    sku_count   = len(env.get_sku_list())
    max_reward  = np.ones((sku_count)) * (-np.inf)
    best_S      = np.zeros((sku_count))
    best_s      = np.zeros((sku_count))
    
    for S in search_range:
        for s in np.arange(0, S + 0.1, 0.5):
            reward, _, _, _ = Ss_policy(env, [S] * sku_count, [s] * sku_count)
            best_S          = np.where(reward > max_reward, S, best_S)
            best_s          = np.where(reward > max_reward, s, best_s)
            max_reward      = np.where(reward > max_reward, reward, max_reward)
    return best_S, best_s

def search_independent_basestock(env, search_range=np.arange(0.0, 12.1, 0.5)):

    env.reset()
    sku_count   = len(env.get_sku_list())
    max_reward  = np.ones((sku_count)) * (-np.inf)
    best_S      = np.zeros((sku_count))
    best_s      = np.zeros((sku_count))
    
    for S in search_range:
        s = S
        reward, _, _, _ = Ss_policy(env, [S] * sku_count, [s] * sku_count)
        best_S          = np.where(reward > max_reward, S, best_S)
        best_s          = np.where(reward > max_reward, s, best_s)
        max_reward      = np.where(reward > max_reward, reward, max_reward)
    return best_S, best_s

def analyze_Ss(env, best_S, best_s, output_file):
    env.reset()

    reward, re_fre, GMV, reward_info = Ss_policy(env, best_S, best_s)

    f = open(output_file+"__", "w")
    f.write("SKU,S,s,reward,profit,excess,order_cost,holding_cost,backlog,replenishment_times,GMV,X\n")
    for i in range(len(env.get_sku_list())):
        f.write(env.get_sku_list()[i] + "," \
            + str(best_S[i]) + "," \
            + str(best_s[i]) + "," \
            + str(reward[i]) + "," \
            + str(reward_info["profit"][i]) + "," \
            + str(reward_info["excess"][i]) + "," \
            + str(reward_info["order_cost"][i]) + "," \
            + str(reward_info["holding_cost"][i]) + "," \
            + str(reward_info["backlog"][i]) + "," \
            + str(re_fre[i]) + ","\
            + str(GMV[i]) + ","\
            + str(reward_info["holding_cost"][i] / GMV[i] * 365) + "\n"
        )
    f.close()

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

def summary(input_dir, output_file):
    f = open(output_file, "w")
    
    f.write("Task,Mode,Total_Reward,Total_Profit,Total_Excess,Total_Order_Cost,Total_Holding_Cost,Total_Backlog,X<0.1,X>0.25,Average_X,GMV,Replenishment_frq,s=0,S>=12,Average_S,Average_s\n")
    for name in os.listdir(input_dir):
        file_name = os.path.join(input_dir, name)
        sku_count = int(name.split('.')[0][3:])
        data = []
        df = pd.read_csv(file_name, sep=",").fillna(0)
        data.append('.'.join(name.split('.')[:-2]))
        data.append(name.split('.')[-2])
        data.append(str(np.round(np.sum(df["reward"]) / 1e3, 2)) + "K")
        data.append(str(np.round(np.sum(df["profit"]) / 1e3, 2)) + "K")
        data.append(str(np.round(np.sum(df["excess"]) / 1e3, 2)) + "K")
        data.append(str(np.round(np.sum(df["order_cost"]) / 1e3, 2)) + "K")
        data.append(str(np.round(np.sum(df["holding_cost"]) / 1e3, 2)) + "K")
        data.append(str(np.round(np.sum(df["backlog"]) / 1e3, 2)) + "K")
        data.append(str(len(df[df["X"].astype(float) < 0.1])))
        data.append(str(len(df[df["X"].astype(float) > 0.25])))
        data.append(str(np.round(np.average(df["X"]), 2)))
        data.append(str(np.round(np.sum(df["GMV"]) / 1e3, 2)))
        data.append(str(np.round(sku_count * 100 / np.sum(df["replenishment_times"]), 2)))
        data.append(str(len(df[df["s"] == 0])))
        data.append(str(len(df[df["s"] >= 0])))
        data.append(str(np.round(np.average(df["S"]), 2)))
        data.append(str(np.round(np.average(df["s"]), 2)))
        f.write(",".join(data) + "\n")
    f.close()

if __name__ == "__main__":

    os.makedirs("output", exist_ok=True)
    os.makedirs(os.path.join("output", "different_Ss"), exist_ok=True)
    output_dir = os.path.join("output", "different_Ss")

    task_list = get_task_list()
    mode_list = ["train", "validation", "test"]

    # for task in task_list:
    #     for mode in mode_list:
    #         env = make_env(task, "OracleWrapper", mode)
    #         best_S, best_s = search_independent_Ss(env)
    #         analyze_Ss(env, best_S, best_s, os.path.join(output_dir, task + "." + mode + ".csv"))

    # summary(output_dir, os.path.join("output", "different_Ss.summary.csv"))

    for task in task_list:
        for mode in mode_list:
            env = make_env(task, "OracleWrapper", mode)
            best_S, best_s = search_independent_basestock(env)
            analyze_Ss(env, best_S, best_s, os.path.join(output_dir, task + "." + mode + ".csv"))

    summary(output_dir, os.path.join("output", "different_basestock.summary.csv"))