import os
import numpy as np
import pandas as pd
import pyecharts.options as opts
from pyecharts.charts import Line, Timeline, Grid
from datetime import datetime, timedelta

"""
use pyecharts to show each sku's policy and overall visualization.
"""
class SimulationTracker:
    def __init__(self, n_agents: int, start_date: datetime, sku_list: list):
        self.n_agents = n_agents
        self.sku_list = sku_list
        self.start_dt = start_date

    """
    visualize all sku's policy by states and monitors, then save in output_dir
    """
    def render_sku(self, states: np.ndarray, states_items: list, sku_monitor: dict, 
                    reward_monitor: dict, output_dir: str):
        
        # visualization all sku's policy
        for i in range(1, self.n_agents + 1):
            # i-th sku's state
            states_df = pd.DataFrame(states[:, :, i - 1].transpose(), columns = states_items)
            # combine i-th sku's sku_monitor and reward_monitor
            monitor_df = pd.DataFrame.from_dict(
                { 
                    **{
                        k: np.array(v)[:, i - 1]
                        for k, v in sku_monitor.items()
                    },
                    **{
                        k: np.cumsum(np.array(v)[:, i - 1])
                        for k, v in reward_monitor.items()
                    },
                }
            )
            monitor_df = pd.concat([states_df, monitor_df], axis = 1)
            monitor_df['day'] = range(len(monitor_df))
            monitor_df['day'] = monitor_df['day'].map(lambda x : \
                (self.start_dt + timedelta(days=x)).strftime('%Y-%m-%d'))
            if not "discrete_action" in monitor_df:
                monitor_df["discrete_action"] = 0

            # init timeline
            tl = Timeline(init_opts=opts.InitOpts(width="1500px", height="800px"))
            tl.add_schema(pos_bottom="bottom", is_auto_play=False, \
                label_opts = opts.LabelOpts(is_show=True, position="bottom"))
            
            # line1 plots "Sales", "Demand", "Stock", "Excess", "Action", "Transits"
            l1 = (Line()
                    .add_xaxis(xaxis_data = monitor_df['day'].tolist())
                    .add_yaxis(
                        series_name="Sales",
                        y_axis=monitor_df['sale'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='blue'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='blue'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="blue"),
                    )
                    .add_yaxis(
                        series_name="Demand",
                        y_axis=monitor_df['demand'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='orange'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='orange'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="orange"),
                    )
                    .add_yaxis(
                        series_name="Stock",
                        y_axis=monitor_df['in_stock'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='green'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='green'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="green"),
                    )
                    .add_yaxis(
                        series_name="Replenishing Quantity",
                        y_axis=monitor_df['replenish'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='aqua'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='aqua'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="aqua"),
                    )
                    .add_yaxis(
                        series_name="Excess",
                        y_axis=monitor_df['excess'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='yellow'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='yellow'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="yellow"),
                    )
                    .add_yaxis(
                        series_name="Action",
                        y_axis=monitor_df['discrete_action'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='black'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='black'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="black"),
                    )
                    .add_yaxis(
                        series_name="Transits",
                        y_axis=monitor_df['in_transit'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='red'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='red'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="red"),
                    )
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title="Replenishing Decisions", pos_top="top", pos_left='left', pos_right='left'),
                        xaxis_opts=opts.AxisOpts(type_="category", name='Date', boundary_gap=False, axisline_opts=opts.AxisLineOpts(is_on_zero=True)),
                        yaxis_opts=opts.AxisOpts(type_="value", is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True)),
                        legend_opts=opts.LegendOpts(pos_left="center"),
                        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                        datazoom_opts=[
                            opts.DataZoomOpts(is_realtime=True, type_="inside", xaxis_index=[0, 1], range_start=45, range_end=65),
                            opts.DataZoomOpts(is_realtime=True, type_="slider", xaxis_index=[0, 1], range_start=45, range_end=65, pos_bottom='55px'),
                        ],
                    )
            )

            # plot rewards
            l2 = (Line()
                    .add_xaxis(xaxis_data = monitor_df['day'].tolist())
                    .add_yaxis(
                        series_name="Reward1",
                        y_axis=monitor_df['reward1'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='blue'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='blue'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="blue"),
                    )
                    .add_yaxis(
                        series_name="Income",
                        y_axis=monitor_df['income'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='orange'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='orange'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="orange"),
                    )
                    .add_yaxis(
                        series_name="Outcome",
                        y_axis=monitor_df['outcome'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='green'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='green'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="green"),
                    )
                    .add_yaxis(
                        series_name="Order_cost",
                        y_axis=monitor_df['order_cost_reward'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='aqua'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='aqua'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="aqua"),
                    )
                    .add_yaxis(
                        series_name="Backlog",
                        y_axis=monitor_df['backlog'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='yellow'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='yellow'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="yellow"),
                    )
                    .add_yaxis(
                        series_name="Reward2",
                        y_axis=monitor_df['reward2'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='black'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='black'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="black"),
                    )
                    .add_yaxis(
                        series_name="Rent",
                        y_axis=monitor_df['holding_cost'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='red'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='red'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="red"),
                    )
                    .add_yaxis(
                        series_name="Excess",
                        y_axis=monitor_df['excess_reward'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='#808000'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='#808000'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color='#808000'),
                    )
                    .add_yaxis(
                        series_name="Profit",
                        y_axis=monitor_df['profit'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='#808000'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='#808000'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color='#808000'),
                    )
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title="Rewards", pos_top='45%', pos_left='left', pos_right='left'),
                        xaxis_opts=opts.AxisOpts(type_="category", name='Date', boundary_gap=False, axisline_opts=opts.AxisLineOpts(is_on_zero=True)),
                        yaxis_opts=opts.AxisOpts(type_="value", is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True)),
                        legend_opts=opts.LegendOpts(pos_left="center", pos_top='45%'),
                        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                        datazoom_opts=[
                            opts.DataZoomOpts(is_realtime=True, type_="inside", xaxis_index=[0, 1], range_start=45, range_end=65),
                            opts.DataZoomOpts(is_realtime=True, type_="slider", xaxis_index=[0, 1], range_start=45, range_end=65, pos_bottom='55px'),
                        ],
                    )
            )
            grid = (
                    Grid()
                    .add(l1, grid_opts=opts.GridOpts(pos_left=100, pos_right=100, height="30%"))
                    .add(l2, grid_opts=opts.GridOpts(pos_left=100, pos_right=100, pos_top="50%", height="30%"))
                )
            
            tl.add(grid, "{}".format(self.start_dt.strftime("%Y-%m-%d")))
            os.makedirs(output_dir, exist_ok=True)
            tl.render(os.path.join(output_dir,'{}.html'.format(self.sku_list[i - 1])))

        # plot overall skus' policy
        if self.n_agents > 1:
            states_df = pd.DataFrame(states.sum(-1).transpose(), columns = states_items)
            monitor_df = pd.DataFrame.from_dict(
                { 
                    **{
                        k: np.cumsum(np.array(v).sum(1))
                        for k, v in reward_monitor.items()
                    },
                }
            )
            total_df = pd.concat([states_df, monitor_df], axis = 1)
            total_df['day'] = range(len(total_df))
            total_df['day'] = total_df['day'].map(lambda x : \
                (self.start_dt + timedelta(days=x)).strftime('%Y-%m-%d'))
            
            # init timeline
            tl = Timeline(init_opts=opts.InitOpts(width="1500px", height="800px"))
            tl.add_schema(pos_bottom="bottom", is_auto_play=False, \
                label_opts = opts.LabelOpts(is_show=True, position="bottom"))

            # line1 plots "Sales", "Demand", "Stock", "Excess", "Action", "Transits"
            l1 = (Line()
                    .add_xaxis(xaxis_data = total_df['day'].tolist())
                    .add_yaxis(
                        series_name="Sales",
                        y_axis=total_df['sale'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='blue'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='blue'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="blue"),
                    )
                    .add_yaxis(
                        series_name="Demand",
                        y_axis=total_df['demand'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='orange'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='orange'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="orange"),
                    )
                    .add_yaxis(
                        series_name="Stock",
                        y_axis=total_df['in_stock'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='green'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='green'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="green"),
                    )
                    .add_yaxis(
                        series_name="Replenishing Quantity",
                        y_axis=total_df['replenish'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='aqua'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='aqua'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="aqua"),
                    )
                    .add_yaxis(
                        series_name="Transits",
                        y_axis=total_df['in_transit'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='red'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='red'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="red"),
                    )
                    .add_yaxis(
                        series_name="excess_amount",
                        y_axis=total_df['excess'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='#FFC0CB'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='#FFC0CB'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="#FFC0CB"),
                    )
                    .set_global_opts(
                            title_opts=opts.TitleOpts(title="Replenishing Decisions", pos_top="top", pos_left='left', pos_right='left'),
                            xaxis_opts=opts.AxisOpts(type_="category", name='Date', boundary_gap=False, axisline_opts=opts.AxisLineOpts(is_on_zero=True)),
                            yaxis_opts=opts.AxisOpts(type_="value", is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True)),
                            legend_opts=opts.LegendOpts(pos_left="center"),
                            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                            datazoom_opts=[
                                opts.DataZoomOpts(is_realtime=True, type_="inside", xaxis_index=[0, 1], range_start=45, range_end=65),
                                opts.DataZoomOpts(is_realtime=True, type_="slider", xaxis_index=[0, 1], range_start=45, range_end=65, pos_bottom='55px'),
                            ],
                    )
            )

            # plot rewards
            l2 = (Line()
                    .add_xaxis(xaxis_data = total_df['day'].tolist())
                    .add_yaxis(
                        series_name="Reward1",
                        y_axis=total_df['reward1'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='blue'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='blue'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="blue"),
                    )
                    .add_yaxis(
                        series_name="Income",
                        y_axis=total_df['income'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='orange'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='orange'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="orange"),
                    )
                    .add_yaxis(
                        series_name="Outcome",
                        y_axis=total_df['outcome'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='green'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='green'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="green"),
                    )
                    .add_yaxis(
                        series_name="Order_cost",
                        y_axis=total_df['order_cost_reward'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='aqua'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='aqua'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="aqua"),
                    )
                    .add_yaxis(
                        series_name="Backlog",
                        y_axis=total_df['backlog'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='yellow'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='yellow'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="yellow"),
                    )
                    .add_yaxis(
                        series_name="Reward2",
                        y_axis=total_df['reward2'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='black'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='black'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="black"),
                    )
                    .add_yaxis(
                        series_name="Rent",
                        y_axis=total_df['holding_cost'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='red'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='red'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color="red"),
                    )
                    .add_yaxis(
                        series_name="Excess",
                        y_axis=monitor_df['excess_reward'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='#808000'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='#808000'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color='#808000'),
                    )
                    .add_yaxis(
                        series_name="Profit",
                        y_axis=monitor_df['profit'].tolist(),
                        symbol_size=8,
                        is_hover_animation=False,
                        label_opts=opts.LabelOpts(is_show=False, color='#808000'),
                        linestyle_opts=opts.LineStyleOpts(width=1.5, color='#808000'),
                        is_smooth=True,
                        itemstyle_opts=opts.ItemStyleOpts(color='#808000'),
                    )
                    .set_global_opts(
                            title_opts=opts.TitleOpts(title="Rewards", pos_top='45%', pos_left='left', pos_right='left'),
                            xaxis_opts=opts.AxisOpts(type_="category", name='Date', boundary_gap=False, axisline_opts=opts.AxisLineOpts(is_on_zero=True)),
                            yaxis_opts=opts.AxisOpts(type_="value", is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True)),
                            legend_opts=opts.LegendOpts(pos_left="center", pos_top='45%'),
                            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
                            datazoom_opts=[
                                opts.DataZoomOpts(is_realtime=True, type_="inside", xaxis_index=[0, 1], range_start=45, range_end=65),
                                opts.DataZoomOpts(is_realtime=True, type_="slider", xaxis_index=[0, 1], range_start=45, range_end=65, pos_bottom='55px'),
                            ],
                    )
            )
            grid = (
                    Grid()
                    .add(l1, grid_opts=opts.GridOpts(pos_left=100, pos_right=100, height="30%"))
                    .add(l2, grid_opts=opts.GridOpts(pos_left=100, pos_right=100, pos_top="50%", height="30%"))
                )
            
            tl.add(grid, "{}".format(self.start_dt.strftime("%Y-%m-%d")))
            os.makedirs(output_dir, exist_ok=True)
            tl.render(os.path.join(output_dir,'overall.html'))