import numpy as np
import gymnasium as gym
from gymnasium import spaces
from tqdm import tqdm
import pickle
from DA_Auction import optimize_alloc, calc_payoff_DA, aftermarket_evaluation
from id_cont import bid_intra_trustfull, bid_intra_strategic, calc_payoff_int_strategic,update_books
from bidder_classes import Bidder
import pandas as pd


class SpotEnv(gym.Env):
    def __init__(self, t_max=200, n=5, q=1448.4, cap_mean=700):
        self.N = n
        self.bidder = Bidder()
        self.bidder_costs = [self.bidder.costs]  # Needs to be tuple
        self.other_costs = [(0.07, 9), (0.02, 10), (0.03, 12), (0.008, 12)]

        self.Q = q
        self.cap_mean = cap_mean
        self.sd_cap = [np.linspace(0.1, 0, 200),
                       np.linspace(0.15, 0, 200),
                       np.linspace(0.2, 0, 200),
                       np.linspace(0.25, 0, 200),
                       np.linspace(self.bidder.sd_cap_start, 0, 200)]  # Needs to be a float

        data_ob = {
            "bid_flag": [],
            "price": [],
            "volume": [],
            "participant": [],
            "timestamp": []
        }
        self.df_order_book = pd.DataFrame(data_ob)
        self.df_order_book = self.df_order_book.astype({
            'bid_flag': 'int64',
            'price': 'float64',
            'volume': 'float64',
            'participant': 'int64',
            'timestamp': 'int64'
        })

        data_bidders = {
            "x_bought": [0] * self.N,
            "x_sold": [0] * self.N,
            "x_da": [0] * self.N,
            "x_imb": [0] * self.N,
            "x_prod": [0] * self.N,
            "x_cap": [0] * self.N,
            "true_costs": self.other_costs + self.bidder_costs,
            "lambda_hat_int": [0] * self.N,
            "revenue": [0] * self.N
        }

        self.df_bidders = pd.DataFrame(data_bidders)

        # Assign dtype to specific columns
        self.df_bidders = self.df_bidders.astype({
            'x_bought': 'float64',
            'x_sold': 'float64',
            'x_da': 'float64',
            'x_imb': 'float64',
            'x_prod': 'float64',
            'x_cap': 'float64',
            'lambda_hat_int': 'float64',
            'true_costs': 'object',  # Ensure true_costs is treated as object
            'revenue': 'float64',
        })

        self._state = 0
        self._max_steps = t_max
        self._current_step = 0

        self.observation_space = spaces.Dict({
            # Private information
            'x_cap': spaces.Box(low=0, high=np.inf, dtype=np.float32),
            'x_imb': spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32),
            'x_prod': spaces.Box(low=0, high=np.inf, dtype=np.float32),
            'x_da': spaces.Box(low=0, high=np.inf, dtype=np.float32),
            'x_bought': spaces.Box(low=0, high=np.inf, dtype=np.float32),
            'x_sold': spaces.Box(low=0, high=np.inf, dtype=np.float32),
            'revenue': spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32),
            # Public information
            'Best bid (price, volume)': spaces.Box(low=np.array([-np.inf, 0]), high=np.array([np.inf, np.inf])),
            'Best ask (price, volume)': spaces.Box(low=np.array([-np.inf, 0]), high=np.array([np.inf, np.inf])),
            'Last trade (price, volume)': spaces.Box(low=np.array([-np.inf, 0]), high=np.array([np.inf, np.inf])),
            'Volume weighted average prices (bid, ask)' : spaces.Box(low=np.array([-np.inf , -np.inf]), high=np.array([np.inf, np.inf])),
            'Sum volume (bid, ask)': spaces.Box(low=np.array([0, 0]), high=np.array([np.inf, np.inf])),
            'steps left': spaces.Discrete(self._max_steps),
        })
        self.action_space = spaces.Box(low=np.array([-30, 0]), high=np.array([30, self.df_bidders.loc[self.N - 1, 'x_cap']]))

    def reset(self, seed=None):
        self.bidder.restart()

        data_ob = {
            "bid_flag": [],
            "price": [],
            "volume": [],
            "participant": [],
            "timestamp": []
        }
        self.df_order_book = pd.DataFrame(data_ob)
        self.df_order_book = self.df_order_book.astype({
            'bid_flag': 'int64',
            'price': 'float64',
            'volume': 'float64',
            'participant': 'int64',
            'timestamp': 'int64'
        })

        data_bidders = {
            "x_bought": [0] * self.N,
            "x_sold": [0] * self.N,
            "x_da": [0] * self.N,
            "x_imb": [0] * self.N,
            "x_prod": [0] * self.N,
            "x_cap": [0] * self.N,
            "true_costs": self.other_costs + self.bidder_costs,
            "lambda_hat_int": [0] * self.N,
            "revenue": [0] * self.N
        }

        self.df_bidders = pd.DataFrame(data_bidders)

        # Assign dtype to specific columns
        self.df_bidders = self.df_bidders.astype({
            'x_bought': 'float64',
            'x_sold': 'float64',
            'x_da': 'float64',
            'x_imb': 'float64',
            'x_prod': 'float64',
            'x_cap': 'float64',
            'lambda_hat_int': 'float64',
            'true_costs': 'object',  # Ensure true_costs is treated as object
            'revenue': 'float64',
        })

        self._current_step = 0

        obs = {
            # Private information
            'x_cap': self.df_bidders.loc[self.N - 1, 'x_cap'],
            'x_imb': self.df_bidders.loc[self.N - 1, 'x_imb'],
            'x_prod': self.df_bidders.loc[self.N - 1, 'x_prod'],
            'x_da': self.df_bidders.loc[self.N - 1, 'x_da'],
            'x_bought': self.df_bidders.loc[self.N - 1, 'x_bought'],
            'x_sold': self.df_bidders.loc[self.N - 1, 'x_sold'],
            'revenue': self.df_bidders.loc[self.N - 1, 'revenue'],
            # Public information
            'Best bid (price, volume)': None,
            'Best ask (price, volume)': None,
            'Last trade (price, volume)': None,
            'Volume weighted average prices (bid, ask)': None,
            'Sum volume (bid, ask)': None,
            'steps left': self._max_steps,
        }

        return obs, {}

    def step(self, action):  # action is an array (price, volume)
        x_cap = [np.random.normal(
            loc=self.cap_mean,
            scale=self.sd_cap[i][self._current_step] * self.cap_mean
        ) for i in range(self.N)]

        # Day-ahead auction
        if self._current_step == 0:

            x, marginal_price, payments, social_welfare = optimize_alloc(action[0], self.other_costs, self.Q,
                                                                         x_cap[:-1] + [action[1]])

            payoff = calc_payoff_DA(self.N, payments, x, self.other_costs, self.bidder)

            bidder_payoff = payoff[-1]

            for i in range(self.N):
                self.df_bidders.at[i, 'x_da'] = x[i]
                self.df_bidders.at[i, 'x_prod'] = x[i]
                self.df_bidders.at[i, 'revenue'] += payoff[i]
                self.df_bidders.at[i, 'x_cap'] = x_cap[i]

            # TODO: Aftermarket evaluation; how to fill replay buffer?
            regret = aftermarket_evaluation(self.other_costs, self.Q, x_cap, self._current_step, self.bidder)

            # log
            self.bidder.history_action.append(action)
            self.bidder.history_payoff.append(bidder_payoff)
        # Intraday auction
        else:
            player = np.random.randint(0, self.N)
            # TODO: write price guessing function; Placeholder: random lambda_hat_int
            self.df_bidders.at[player, 'lambda_hat_int'] = np.random.randint(10, 30)
            self.df_bidders.at[player, 'x_cap'] = x_cap[player]

            if player != self.N - 1:
                x_prod, x_imb, new_post = bid_intra_trustfull(player, self.df_bidders, self._max_steps,
                                                              self._current_step)
                # log action
                self.bidder.history_action.append(None)
            else:
                x_prod, x_imb, new_post = bid_intra_strategic(action, self.df_bidders, self._current_step)
                # log action
                self.bidder.history_action.append(action)

            rev_before_match = self.df_bidders.at[self.N - 1, 'revenue']

            self.df_order_book, self.df_bidders = update_books(self.df_order_book, self.df_bidders, player, new_post,
                                                               x_prod, x_imb)
            # Calculate Payoff
            bidder_payoff = calc_payoff_int_strategic(self.bidder, self.df_bidders, rev_before_match)

            # log
            self.bidder.history_payoff.append(bidder_payoff)

        # Define state and reward
        imbalance_penalty = (self._current_step / (self._max_steps - self._current_step + 1e-5)
                             * self.df_bidders.at[self.N - 1, 'x_imb'])
        reward = bidder_payoff - imbalance_penalty
        self._state = {
            'Order book': self.df_order_book,
            'Bidders': self.df_bidders,
        }
        self._current_step += 1
        done = False
        if self._current_step >= self._max_steps:
            done = True

        # TODO: calc last trade, volume weighted average prices
        best_bid = self.df_order_book[self.df_order_book['bid_flag'] == 1].iloc[
            len(self.df_order_book[self.df_order_book['bid_flag'] == 1]) - 1]
        best_ask = self.df_order_book[self.df_order_book['bid_flag'] == 0].iloc[0]
        sum_volume_bid = self.df_order_book[self.df_order_book['bid_flag'] == 1]['volume'].sum()
        sum_volume_ask = self.df_order_book[self.df_order_book['bid_flag'] == 0]['volume'].sum()

        obs = {
            # Private information
            'x_cap': self.df_bidders.loc[self.N - 1, 'x_cap'],
            'x_imb': self.df_bidders.loc[self.N - 1, 'x_imb'],
            'x_prod': self.df_bidders.loc[self.N - 1, 'x_prod'],
            'x_da': self.df_bidders.loc[self.N - 1, 'x_da'],
            'x_bought': self.df_bidders.loc[self.N - 1, 'x_bought'],
            'x_sold': self.df_bidders.loc[self.N - 1, 'x_sold'],
            'revenue': self.df_bidders.loc[self.N - 1, 'revenue'],
            # Public information
            'Best bid (price, volume)': np.array([best_bid['price'], best_bid['volume']]),
            'Best ask (price, volume)': np.array([best_ask['price'], best_ask['volume']]),
            'Last trade (price, volume)': None,
            'Volume weighted average prices (bid, ask)': None,
            'Sum volume (bid, ask)': np.array([sum_volume_bid, sum_volume_ask]),
            'steps left': self._max_steps - self._current_step,
        }

        return obs, reward, done, False, {}
