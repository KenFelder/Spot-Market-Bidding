import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import pickle
from DA_Auction import optimize_alloc, calc_payoff_DA
from id_cont import bid_intra_trustful, bid_intra_strategic, update_books
from bidder_classes import Bidder


# TODO: Check Bid/Ask integrity
class SpotEnv(gym.Env):
    def __init__(self, t_max=200, n=5, q=1448.4, cap_mean=700):

        self._current_step = 0
        self.t_int = 0

        self.N = n
        self.bidder = Bidder()
        self.bidder_costs = [self.bidder.costs]  # Needs to be tuple
        self.other_costs = [(0.07, 9), (0.02, 10), (0.03, 12), (0.008, 12)]

        self.t_max = t_max
        self._max_steps = t_max + self.bidder.aftermarket_exploration

        self.Q = q
        self.cap_mean = cap_mean
        self.sd_cap = [np.linspace(0.1, 0, self.t_max, dtype=np.float64),
                       np.linspace(0.15, 0, self.t_max, dtype=np.float64),
                       np.linspace(0.2, 0, self.t_max, dtype=np.float64),
                       np.linspace(0.25, 0, self.t_max, dtype=np.float64),
                       np.linspace(self.bidder.sd_cap_start, 0, self.t_max, dtype=np.float64)]

        self.bidder_DA_cap = [np.random.normal(loc=self.cap_mean, scale=self.sd_cap[-1][self.t_int] * self.cap_mean)]

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

        self.top_bid_prices = []
        self.top_ask_prices = []

        data_bidders = {
            "x_bought": [0] * self.N,
            "x_sold": [0] * self.N,
            "x_da": [0] * self.N,
            "x_imb": [0] * self.N,
            "x_prod": [0] * self.N,
            "x_cap": [0] * (self.N - 1) + self.bidder_DA_cap,
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
            'true_costs': 'object',  # Tuple
            'revenue': 'float64',
        })



        self.observation_space = spaces.Dict({
            # Private information
            'x_cap': spaces.Box(low=0, high=np.inf, dtype=np.float64),
            'x_imb': spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'x_prod': spaces.Box(low=0, high=np.inf, dtype=np.float64),
            'x_da': spaces.Box(low=0, high=np.inf, dtype=np.float64),
            'x_bought': spaces.Box(low=0, high=np.inf, dtype=np.float64),
            'x_sold': spaces.Box(low=0, high=np.inf, dtype=np.float64),
            'revenue': spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            # Public information
            'Best bid (price, volume)': spaces.Box(low=np.array([-np.inf, 0]), high=np.array([np.inf, np.inf]), dtype=np.float64),
            'Best ask (price, volume)': spaces.Box(low=np.array([-np.inf, 0]), high=np.array([np.inf, np.inf]), dtype=np.float64),
            #'Last trade (price, volume)': spaces.Box(low=np.array([-np.inf, 0]), high=np.array([np.inf, np.inf]),
            #                                         dtype=np.float64),
            #'Volume weighted average prices (bid, ask)': spaces.Box(low=np.array([-np.inf, -np.inf]),
            #                                                         high=np.array([np.inf, np.inf]), dtype=np.float64),
            'Sum volume (bid, ask)': spaces.Box(low=np.array([0, 0]), high=np.array([np.inf, np.inf]), dtype=np.float64),
            'steps left': spaces.Box(low=0, high=self.t_max, dtype=np.float64),
        })

        self.action_space = spaces.Box(low=np.array([0, -self.cap_mean]),
                                       high=np.array([30, self.cap_mean]), dtype=np.float64)

    def get_obs(self):
        if self._current_step == 0:
            obs = {
                'x_cap': np.array([self.df_bidders.loc[self.N - 1, 'x_cap']]),
                'x_imb': np.array([self.df_bidders.loc[self.N - 1, 'x_imb']]),
                'x_prod': np.array([self.df_bidders.loc[self.N - 1, 'x_prod']]),
                'x_da': np.array([self.df_bidders.loc[self.N - 1, 'x_da']]),
                'x_bought': np.array([self.df_bidders.loc[self.N - 1, 'x_bought']]),
                'x_sold': np.array([self.df_bidders.loc[self.N - 1, 'x_sold']]),
                'revenue': np.array([self.df_bidders.loc[self.N - 1, 'revenue']]),
                'Best bid (price, volume)': np.array([0, 0]),
                'Best ask (price, volume)': np.array([0, 0]),
                'Sum volume (bid, ask)': np.array([0, 0]),
                'steps left': np.array([self.t_max - self.t_int]),
            }

            # obs = {
            #     # Private information
            #     'x_cap': np.array([self.df_bidders.loc[self.N - 1, 'x_cap']]),
            #     'x_imb': self.df_bidders.loc[self.N - 1, 'x_imb'],
            #     'x_prod': self.df_bidders.loc[self.N - 1, 'x_prod'],
            #     'x_da': self.df_bidders.loc[self.N - 1, 'x_da'],
            #     'x_bought': self.df_bidders.loc[self.N - 1, 'x_bought'],
            #     'x_sold': self.df_bidders.loc[self.N - 1, 'x_sold'],
            #     'revenue': self.df_bidders.loc[self.N - 1, 'revenue'],
            #     # Public information
            #     'Best bid (price, volume)': np.array([0, 0]),
            #     'Best ask (price, volume)': np.array([0, 0]),
            #     #'Last trade (price, volume)': None,
            #     #'Volume weighted average prices (bid, ask)': None,
            #     'Sum volume (bid, ask)': np.array([0, 0]),
            #     'steps left': self._max_steps,
            # }
        else:
            # TODO: calc last trade, volume weighted average prices
            len_bids = len(self.df_order_book[self.df_order_book['bid_flag'] == 1])
            len_asks = len(self.df_order_book[self.df_order_book['bid_flag'] == 0])
            if len_bids != 0:
                best_bid = self.df_order_book[self.df_order_book['bid_flag'] == 1].iloc[0]
                sum_volume_bid = self.df_order_book[self.df_order_book['bid_flag'] == 1]['volume'].sum()
            else:
                best_bid = dict({'price': 0, 'volume': 0})
                sum_volume_bid = 0
            if len_asks != 0:
                best_ask = self.df_order_book[self.df_order_book['bid_flag'] == 0].iloc[
                    len_asks - 1]
                sum_volume_ask = self.df_order_book[self.df_order_book['bid_flag'] == 0]['volume'].sum()
            else:
                best_ask = dict({'price': 0, 'volume': 0})
                sum_volume_ask = 0

            obs = {
                'x_cap': np.array([self.df_bidders.loc[self.N - 1, 'x_cap']]),
                'x_imb': np.array([self.df_bidders.loc[self.N - 1, 'x_imb']]),
                'x_prod': np.array([self.df_bidders.loc[self.N - 1, 'x_prod']]),
                'x_da': np.array([self.df_bidders.loc[self.N - 1, 'x_da']]),
                'x_bought': np.array([self.df_bidders.loc[self.N - 1, 'x_bought']]),
                'x_sold': np.array([self.df_bidders.loc[self.N - 1, 'x_sold']]),
                'revenue': np.array([self.df_bidders.loc[self.N - 1, 'revenue']]),
                'Best bid (price, volume)': np.array([0, 0]),
                'Best ask (price, volume)': np.array([0, 0]),
                'Sum volume (bid, ask)': np.array([0, 0]),
                'steps left': np.array([self.t_max - self.t_int]),
            }

            # obs = {
            #     # Private information
            #     'x_cap': self.df_bidders.loc[self.N - 1, 'x_cap'],
            #     'x_imb': self.df_bidders.loc[self.N - 1, 'x_imb'],
            #     'x_prod': self.df_bidders.loc[self.N - 1, 'x_prod'],
            #     'x_da': self.df_bidders.loc[self.N - 1, 'x_da'],
            #     'x_bought': self.df_bidders.loc[self.N - 1, 'x_bought'],
            #     'x_sold': self.df_bidders.loc[self.N - 1, 'x_sold'],
            #     'revenue': self.df_bidders.loc[self.N - 1, 'revenue'],
            #     # Public information
            #     'Best bid (price, volume)': np.array([best_bid['price'], best_bid['volume']]),
            #     'Best ask (price, volume)': np.array([best_ask['price'], best_ask['volume']]),
            #     #'Last trade (price, volume)': None,
            #     #'Volume weighted average prices (bid, ask)': None,
            #     'Sum volume (bid, ask)': np.array([sum_volume_bid, sum_volume_ask]),
            #     'steps left': self._max_steps - self.t_int,
            # }

        return obs

    def reset(self, seed=None):
        self.bidder.restart()

        self._current_step = 0
        self.t_int = 0

        self.sd_cap = [np.linspace(0.1, 0, self.t_max, dtype=np.float64),
                       np.linspace(0.15, 0, self.t_max, dtype=np.float64),
                       np.linspace(0.2, 0, self.t_max, dtype=np.float64),
                       np.linspace(0.25, 0, self.t_max, dtype=np.float64),
                       np.linspace(self.bidder.sd_cap_start, 0, self.t_max, dtype=np.float64)]

        self.bidder_DA_cap = [np.random.normal(loc=self.cap_mean, scale=self.sd_cap[-1][self.t_int] * self.cap_mean)]

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

        self.top_bid_prices = []
        self.top_ask_prices = []

        data_bidders = {
            "x_bought": [0] * self.N,
            "x_sold": [0] * self.N,
            "x_da": [0] * self.N,
            "x_imb": [0] * self.N,
            "x_prod": [0] * self.N,
            "x_cap": [0] * (self.N - 1) + self.bidder_DA_cap,
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

        obs = self.get_obs()

        return obs, {}

    def step(self, action):  # action is an array (price, volume)
        # if self._current_step == 0:
        #     print(f'\nNew game')
        #print(f'Current step:   {self._current_step} \nAction:         {action}')

        x_cap = [np.random.normal(
            loc=self.cap_mean,
            scale=self.sd_cap[i][self.t_int] * self.cap_mean
        ) for i in range(self.N)]

        done = False
        truncated = False
        # Day-ahead auction
        if self._current_step == 0:

            x, marginal_price, payments, social_welfare = optimize_alloc(action[0], self.other_costs, self.Q,
                                                                         x_cap[:-1] + [max(0, action[1])])

            payoff = calc_payoff_DA(self.N, payments, x, self.other_costs, self.bidder)
            bidder_payoff = payoff[-1]
            bidder_payment = payments[-1]

            for i in range(self.N):
                self.df_bidders.at[i, 'x_da'] = x[i]
                self.df_bidders.at[i, 'x_prod'] = x[i]
                self.df_bidders.at[i, 'revenue'] += payments[i]
                if i < self.N - 1:
                    self.df_bidders.at[i, 'x_cap'] = x_cap[i]

            # log
            self.bidder.history_action.append(action)
            self.bidder.history_payoff.append(bidder_payoff)

        elif self._current_step <= self.bidder.aftermarket_exploration:
            x_tmp, marginal_price_tmp, payments_tmp, sw = optimize_alloc(action[0], self.other_costs, self.Q,
                                                                         x_cap[:-1] + [max(0, action[1])])

            bidder_payment = payments_tmp[-1]

        # Intraday auction
        else:
            bidder_payoff = 0
            bidder_payment = 0
            while True:
                player = np.random.randint(0, self.N)
                # TODO: write price guessing function; Placeholder: random lambda_hat_int
                self.df_bidders.at[player, 'lambda_hat_int'] = np.random.randint(10, 30)
                self.df_bidders.at[player, 'x_cap'] = x_cap[player]

                if player != self.N - 1:
                    x_prod, x_imb, new_post = bid_intra_trustful(player, self.df_bidders, self.t_max,
                                                                 self.t_int)
                    # log action
                    self.bidder.history_action.append(None)
                else:
                    x_prod, x_imb, new_post = bid_intra_strategic(action, player, self.df_bidders, self.t_int)
                    # log action
                    self.bidder.history_action.append(action)

                rev_before_match = self.df_bidders.at[self.N - 1, 'revenue']




                self.df_order_book, self.df_bidders, self.top_bid_prices, self.top_ask_prices = (
                    update_books(self.df_order_book, self.df_bidders, player, new_post, x_prod, x_imb,
                                 self.top_bid_prices, self.top_ask_prices))

                # TODO: Check payoff calculation logic
                bidder_payment += self.df_bidders.at[self.N - 1, 'revenue'] - rev_before_match

                # log
                self.bidder.history_payoff.append(bidder_payoff)
                # Intraday timer
                #print(f'Current t:  {self.t_int}\n')
                self.t_int += 1
                if self.t_int >= self.t_max:
                    truncated = True
                    break
                if player == self.N - 1:
                    break

        # TODO: Make sure imbalance is calculated correctly (e.g. x_prod gets updated to reduce x_imb)

        # Define state and reward
        imbalance_penalty = (self.t_int / (self.t_max - self.t_int + 1e-5)
                             * self.df_bidders.at[self.N - 1, 'x_imb'])
        production_cost = (0.5 * self.bidder_costs[0][1] * self.df_bidders.at[self.N - 1, 'x_prod'] ** 2
                           + self.bidder_costs[0][1] * self.df_bidders.at[self.N - 1, 'x_prod'])
        reward = bidder_payment - imbalance_penalty - production_cost

        self._current_step += 1

        if self._current_step >= self._max_steps:
            done = True

        obs = self.get_obs()

        return obs, reward, done, truncated, {}
