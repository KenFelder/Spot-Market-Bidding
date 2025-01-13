import numpy as np
import pandas as pd
from config import *


def init_forecasts(self):
    sd_re_gen = [np.linspace(start_sd_re_gen[i], 0, t_max, dtype=np.float64) for i in range(n)]
    self.x_re_cap = [[np.random.normal(
        loc=re_gen_mean[i],
        scale=sd_re_gen[i][t] * re_gen_mean[i]
    ) for t in range(t_max)] for i in range(n)]
    self.x_re_cap = [[max(value, 0) for value in sublist] for sublist in self.x_re_cap]
    # Total Capacity
    self.x_cap = [[x_th_cap[i] + self.x_re_cap[i][t] for i in range(n)] for t in range(t_max)]
    ## Demand
    sd_demand = np.linspace(start_sd_demand, 0, t_max, dtype=np.float64)
    self.x_demand = [np.random.normal(loc=demand_mean, scale=sd_demand[t] * (-demand_mean)) for t in range(t_max)]
    self.x_demand = [min(value, 0) for value in self.x_demand]

    return self.x_re_cap, self.x_cap, self.x_demand

def init_game_data(self):
    data_gd = {
        'market': [],
        'marginal_price': [],
        'social_welfare': [],
        'system_imbalance': [],
        'imbalance_penalty_factor': [],
        'transaction_price': [],
        'top_bid': [],
        'top_ask': [],
        'equilibrium_price_estimate': [],
        'volatility': [],
        'last_event': None,
        'last_price': [],
    }
    self.df_game_data = pd.DataFrame(data_gd).astype({
        'market': 'object',
        'marginal_price': 'float64',
        'social_welfare': 'float64',
        'system_imbalance': 'float64',
        'imbalance_penalty_factor': 'float64',
        'transaction_price': 'float64',
        'top_bid': 'float64',
        'top_ask': 'float64',
        'equilibrium_price_estimate': 'float64',
        'volatility': 'float64',
        'last_event': 'object',
        'last_price': 'float64'
    })
    return self.df_game_data

def init_bid_logs(self):
    data_bid_logs = {
        'bidder': [],
        'price': [],
        'volume': [],
        'transaction_price': [],
        'match_flag': [],
    }
    self.df_bid_logs = pd.DataFrame(data_bid_logs)
    self.df_bid_logs = self.df_bid_logs.astype({
        'bidder': 'int64',
        'price': 'float64',
        'volume': 'float64',
        'transaction_price': 'float64',
        'match_flag': 'object',
    })
    return self.df_bid_logs

def init_order_book(self):
    data_ob = {
        'bid_flag': [],
        'price': [],
        'volume': [],
        'participant': [],
        'timestamp': []
    }
    self.df_order_book = pd.DataFrame(data_ob)
    self.df_order_book = self.df_order_book.astype({
        'bid_flag': 'int64',
        'price': 'float64',
        'volume': 'float64',
        'participant': 'int64',
        'timestamp': 'int64'
    })
    return self.df_order_book

def init_bidders(self):
    data_bidders = {
        'x_demand': [0] * n,
        'x_bought': [0] * n,
        'x_sold': [0] * n,
        'x_da': [0] * n,
        'x_imb': [0] * n,
        'x_re_gen': [0] * n,
        'x_th_gen': [0] * n,
        'x_re_cap': [self.x_re_cap[i][0] for i in range(n)],
        'x_th_cap': x_th_cap,
        'x_prod': [0] * n,
        'x_cap': [self.x_cap[i][0] for i in range(n)],
        'marginal_costs': [0] * n,
        'true_costs': true_costs,
        'production_costs': [0] * n,
        'penalty_imbalance': [0] * n,
        'revenue': [0] * n,
        'expenses': [0] * n,
        'payoff': [0] * n,
        'ask_price': [0] * n,
        'bid_price': [0] * n,
        'limit_buy': [0] * n,
        'limit_sell': [0] * n,
        'target_price_param': start_target_price_param,
        'target_price_param_step_factor': target_price_param_step_factor,
        'aggressiveness_buy': start_aggressiveness_bid,
        'aggressiveness_sell': start_aggressiveness_ask,
        'aggressiveness_step_factor': aggressiveness_step_factor,
        'bid_step_factor': bid_step_factor,

    }
    self.df_bidders = pd.DataFrame(data_bidders)
    self.df_bidders = self.df_bidders.astype({
        'x_demand': 'float64',
        'x_bought': 'float64',
        'x_sold': 'float64',
        'x_da': 'float64',
        'x_imb': 'float64',
        'x_re_gen': 'float64',
        'x_th_gen': 'float64',
        'x_re_cap': 'float64',
        'x_th_cap': 'float64',
        'x_prod': 'float64',
        'x_cap': 'float64',
        'marginal_costs': 'float64',
        'true_costs': 'float64',
        'production_costs': 'float64',
        'penalty_imbalance': 'float64',
        'revenue': 'float64',
        'expenses': 'float64',
        'payoff': 'float64',
        'ask_price': 'float64',
        'bid_price': 'float64',
        'limit_buy': 'float64',
        'limit_sell': 'float64',
        'target_price_param': 'float64',
        'target_price_param_step_factor': 'float64',
        'aggressiveness_buy': 'float64',
        'aggressiveness_sell': 'float64',
        'aggressiveness_step_factor': 'float64',
        'bid_step_factor': 'float64',
    })
    return self.df_bidders
