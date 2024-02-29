import argparse
import os
import sys
import gym
import pandas as pd
import numpy as np
import cvxpy as cp

from ReplenishmentEnv import make_env


def calculate_stock_quantity(
    selling_price: np.array,
    procurement_cost: np.array,
    demand: np.array,
    vlt: int,
    in_stock: int,
    replenishment_before: list,
    unit_storage_cost: int 
) -> np.ndarray:

        # time_hrz_len = history_len + 1 + future_len
        time_hrz_len = len(selling_price)

        # Inventory on hand.
        stocks = cp.Variable(time_hrz_len + 1, integer=True)
        # Inventory on the pipeline.
        transits = cp.Variable(time_hrz_len + 1, integer=True)
        sales = cp.Variable(time_hrz_len, integer=True)
        buy = cp.Variable(time_hrz_len + vlt, integer=True)
        # Requested product quantity from upstream.
        buy_in = cp.Variable(time_hrz_len, integer=True)
        # Expected accepted product quantity.
        buy_arv = cp.Variable(time_hrz_len, integer=True)
        target_stock = cp.Variable(time_hrz_len, integer=True)

        profit = cp.Variable(1)

        max_stock = max(demand) + in_stock + replenishment_before[-vlt]

        # Add constraints.
        constraints = [
            # Variable lower bound.
            stocks >= 0,
            transits >= 0,
            sales >= 0,
            buy >= 0,
            # Initial values.
            stocks[0] == in_stock,
            transits[0] == cp.sum(replenishment_before),
            # Recursion formulas.
            stocks[1 : time_hrz_len + 1] == stocks[0:time_hrz_len] + buy_arv - sales,
            transits[1 : time_hrz_len + 1] == transits[0:time_hrz_len] - buy_arv + buy_in,
            sales <= stocks[0:time_hrz_len],
            sales <= demand,
            buy_in == buy[vlt : time_hrz_len + vlt],
            buy_arv == buy[0:time_hrz_len],
            target_stock == stocks[0:time_hrz_len] + transits[0:time_hrz_len] + buy_in,
            # Objective function.
            profit == cp.sum(
                cp.multiply(selling_price, sales) - cp.multiply(procurement_cost, buy_in) - cp.multiply(unit_storage_cost, stocks[1:]),
            ),
        ]
        # Init the buy before action
        for i in range(vlt):
            constraints.append(buy[i] == replenishment_before[i])
        obj = cp.Maximize(profit)
        prob = cp.Problem(obj, constraints + [stocks[len(replenishment_before):] <= max_stock])
        prob.solve(solver=cp.GLPK_MI, verbose=False)
        if target_stock.value is None:
            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.GLPK_MI, verbose=False)
        return target_stock.value

def oracle_base_stock(env: gym.Wrapper):
    env.reset()
    stock_quantity_list = []
    for sku in env.get_sku_list():
        selling_price        = env.get_selling_price(sku)
        procurement_cost     = env.get_procurement_cost(sku)
        demand               = env.get_demand(sku)
        average_vlt          = env.get_average_vlt(sku)
        in_stock             = env.get_in_stock(sku)
        unit_storage_cost    = env.get_unit_storage_cost(sku)
        replenishment_before = env.get_replenishment_before(sku)

        stock_quantity = calculate_stock_quantity(
            selling_price, 
            procurement_cost, 
            demand, 
            average_vlt, 
            in_stock, 
            replenishment_before, 
            unit_storage_cost
        ).reshape(-1, 1)
        stock_quantity_list.append(stock_quantity)
    total_stock_quantity = np.concatenate(stock_quantity_list, axis=1)

    current_step = 0
    is_done = False
    while not is_done:
        current_stock_quantity = total_stock_quantity[current_step]
        replenish = current_stock_quantity - env.get_in_stock() - env.get_in_transit()
        replenish = np.where(replenish >= 0, replenish, 0) / (env.get_demand_mean() + 0.00001)
        states, reward, is_done, info = env.step(replenish)
        current_step += 1
    return info["balance"]

if __name__ == "__main__":
    env_names = [
        "sku100.Standard",
        "sku100.CapacityHigh",
        "sku100.CapacityLow",
        "sku100.CapacityLowest",
        "sku2307.Standard",
        "sku2307.CapacityHigh",
        "sku2307.CapacityLow",
        "sku2307.CapacityLowest",
    ]
    mode_list = ["train", "validation", "test"]
    os.makedirs("output", exist_ok=True)

    record = []
    for env_name in env_names:
        for mode in mode_list:
            env = make_env(env_name, "OracleWrapper", mode)
            balance = oracle_base_stock(env)
            record.append({
                "task": env_name,
                "mode": mode,
                "balance": balance,
            })
            print(env_name, mode, balance)
    record = pd.DataFrame(record)
    record.to_csv(os.path.join("output", "base_stock.summary.csv"))
