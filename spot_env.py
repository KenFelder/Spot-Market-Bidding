import os
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from init import *
from DA_Auction import *
from id_cont import *
from price_calc import *
from config import *


class SpotEnv(gym.Env):
    def __init__(self):
        #### Steps
        self._current_step = 0
        self.t_int = 0
        self.t_max = t_max
        self._max_steps = self.t_max + aftermarket_expl

        #### Forecasting
        ## Renewable generation
        self.x_re_cap, self.x_cap, self.x_demand = init_forecasts(self)

        #### Intraday Market
        ## Order Book Dataframe
        self.df_order_book = init_order_book(self)

        ## Bidder Dataframe
        self.df_bidders = init_bidders(self)

        ## Log ID
        init_logs(self)

        #### RL definition
        ## Observation space
        self.observation_space = spaces.Dict({
            # Private information
            'x_cap': spaces.Box(low=0, high=np.inf, dtype=np.float64),
            'x_imb': spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            'x_re_gen': spaces.Box(low=0, high=np.inf, dtype=np.float64),
            'x_prod': spaces.Box(low=0, high=np.inf, dtype=np.float64),
            'x_da': spaces.Box(low=0, high=np.inf, dtype=np.float64),
            'x_bought': spaces.Box(low=0, high=np.inf, dtype=np.float64),
            'x_sold': spaces.Box(low=0, high=np.inf, dtype=np.float64),
            'revenue': spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),
            # Public information
            'Best bid (price, volume)': spaces.Box(low=np.array([-np.inf, 0]), high=np.array([np.inf, np.inf]),
                                                   dtype=np.float64),
            'Best ask (price, volume)': spaces.Box(low=np.array([-np.inf, 0]), high=np.array([np.inf, np.inf]),
                                                   dtype=np.float64),
            #'Last trade (price, volume)': spaces.Box(low=np.array([-np.inf, 0]), high=np.array([np.inf, np.inf]),
            #                                         dtype=np.float64),
            #'Volume weighted average prices (bid, ask)': spaces.Box(low=np.array([-np.inf, -np.inf]),
            #                                                         high=np.array([np.inf, np.inf]), dtype=np.float64),
            'Sum volume (bid, ask)': spaces.Box(low=np.array([0, 0]), high=np.array([np.inf, np.inf]), dtype=np.float64),
            'steps left': spaces.Box(low=0, high=self.t_max, dtype=np.float64),
        })

        ## Action space
        self.action_space = spaces.Box(low=np.array([min_price, -max_bid_volume]),
                                       high=np.array([max_price, max_ask_volume]), dtype=np.float64)

    def reset(self, seed=None):
        # log
        self.timestamp = datetime.now().strftime('%Y%m%d_%H-%M-%S')
        os.makedirs(f'./csv/{self.timestamp}/', exist_ok=True)
        os.makedirs(f'./csv/{self.timestamp}/', exist_ok=True)
        os.makedirs(f'./csv/{self.timestamp}/', exist_ok=True)

        #### Steps
        self._current_step = 0
        self.t_int = 0
        self.t_max = t_max
        self._max_steps = self.t_max + aftermarket_expl

        #### Forecasting
        ## Renewable generation
        self.x_re_cap, self.x_cap, self.x_demand = init_forecasts(self)

        #### Intraday Market
        ## Order Book Dataframe
        self.df_order_book = init_order_book(self)

        ## Bidder Dataframe
        self.df_bidders = init_bidders(self)

        ## Log ID
        init_logs(self)

        obs = self.get_obs()

        return obs, {}

    def get_obs(self):
        if self._current_step == 0:
            obs = {
                'x_cap': np.array([self.df_bidders.loc[n - 1, 'x_cap']]),
                'x_imb': np.array([self.df_bidders.loc[n - 1, 'x_imb']]),
                'x_re_gen': np.array([self.df_bidders.loc[n - 1, 'x_re_gen']]),
                'x_prod': np.array([self.df_bidders.loc[n - 1, 'x_prod']]),
                'x_da': np.array([self.df_bidders.loc[n - 1, 'x_da']]),
                'x_bought': np.array([self.df_bidders.loc[n - 1, 'x_bought']]),
                'x_sold': np.array([self.df_bidders.loc[n - 1, 'x_sold']]),
                'revenue': np.array([self.df_bidders.loc[n - 1, 'revenue']]),
                'Best bid (price, volume)': np.array([0, 0]),
                'Best ask (price, volume)': np.array([0, 0]),
                'Sum volume (bid, ask)': np.array([0, 0]),
                'steps left': np.array([self.t_max - self.t_int]),
            }

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
                'x_cap': np.array([self.df_bidders.loc[n - 1, 'x_cap']]),
                'x_imb': np.array([self.df_bidders.loc[n - 1, 'x_imb']]),
                'x_re_gen': np.array([self.df_bidders.loc[n - 1, 'x_re_gen']]),
                'x_prod': np.array([self.df_bidders.loc[n - 1, 'x_prod']]),
                'x_da': np.array([self.df_bidders.loc[n - 1, 'x_da']]),
                'x_bought': np.array([self.df_bidders.loc[n - 1, 'x_bought']]),
                'x_sold': np.array([self.df_bidders.loc[n - 1, 'x_sold']]),
                'revenue': np.array([self.df_bidders.loc[n - 1, 'revenue']]),
                'Best bid (price, volume)': np.array([0, 0]),
                'Best ask (price, volume)': np.array([0, 0]),
                'Sum volume (bid, ask)': np.array([0, 0]),
                'steps left': np.array([self.t_max - self.t_int]),
            }

        return obs

    def step(self, action):  # action is an array (price, volume)
        done = False
        truncated = False

        # Day-ahead auction
        if self._current_step == 0:
            init_new_round(self)
            max_sw(self, action)
            self.t_int += 1

        # Aftermarket exploration TODO: not implemented; Status now would overwrite actual DA data
        #elif self._current_step <= aftermarket_expl:

            #x_tmp, marginal_price_tmp, payments_tmp, sw = blind_auction(prices_da, residual_load, volumes_da)  #TODO: change to new function max_sw


        # Intraday auction
        else:
            while True:
                init_new_round(self)

                player = np.random.randint(0, n)

                # TODO: delete to include rl agent
                if player == n - 1:
                    break

                # TODO: Double-check this poc
                calc_prices(self)

                if player != n - 1:
                    new_post = bid_intra_trustful(self, player)
                else:
                    new_post = bid_intra_strategic(self, action, player)

                update_books(self, player, new_post)

                self.df_x_demand.loc[self.t_int] = self.df_bidders['x_demand'].values
                self.df_x_bought.loc[self.t_int] = self.df_bidders['x_bought'].values
                self.df_x_sold.loc[self.t_int] = self.df_bidders['x_sold'].values
                self.df_x_re_cap.loc[self.t_int] = self.df_bidders['x_re_cap'].values
                self.df_x_re_gen.loc[self.t_int] = self.df_bidders['x_re_gen'].values
                self.df_x_th_gen.loc[self.t_int] = self.df_bidders['x_th_gen'].values

                self.df_ask_prices.loc[self.t_int] = self.df_bidders['ask_price'].values
                self.df_bid_prices.loc[self.t_int] = self.df_bidders['bid_price'].values
                self.df_bid_agg.loc[self.t_int] = self.df_bidders['aggressiveness_buy'].values
                self.df_ask_agg.loc[self.t_int] = self.df_bidders['aggressiveness_sell'].values
                self.df_target_price_param.loc[self.t_int] = self.df_bidders['target_price_param'].values
                self.df_limit_buy.loc[self.t_int] = self.df_bidders['limit_buy'].values
                self.df_limit_sell.loc[self.t_int] = self.df_bidders['limit_sell'].values

                self.df_market_positions.loc[self.t_int] = self.df_bidders['market_position'].values
                self.df_payoffs.loc[self.t_int] = self.df_bidders['payoff'].values
                self.df_revenues.loc[self.t_int] = self.df_bidders['revenue'].values
                self.df_expenses.loc[self.t_int] = self.df_bidders['expenses'].values
                self.df_prod_costs.loc[self.t_int] = self.df_bidders['production_costs'].values
                self.df_penalty_imbalances.loc[self.t_int] = self.df_bidders['penalty_imbalance'].values
                self.df_imbalances.loc[self.t_int] = self.df_bidders['x_imb'].values

                # Intraday timer
                self.t_int += 1
                if self.t_int >= self.t_max:
                    truncated = True
                    break
                if player == n - 1:
                    break

                #TODO: Breakpoint
                if self.t_int == 100:
                    pass

        self.df_game_data.to_csv(f'./csv/{self.timestamp}/game_data.csv', sep=';')
        self.df_bid_logs.to_csv(f'./csv/{self.timestamp}/bid_logs.csv', sep=';')

        self.df_x_demand.to_csv(f'./csv/{self.timestamp}/x_demand.csv', sep=';')
        self.df_x_bought.to_csv(f'./csv/{self.timestamp}/x_bought.csv', sep=';')
        self.df_x_sold.to_csv(f'./csv/{self.timestamp}/x_sold.csv', sep=';')
        self.df_x_re_cap.to_csv(f'./csv/{self.timestamp}/x_re_cap.csv', sep=';')
        self.df_x_re_gen.to_csv(f'./csv/{self.timestamp}/x_re_gen.csv', sep=';')
        self.df_x_th_gen.to_csv(f'./csv/{self.timestamp}/x_th_gen.csv', sep=';')

        self.df_ask_prices.to_csv(f'./csv/{self.timestamp}/ask_prices.csv', sep=';')
        self.df_bid_prices.to_csv(f'./csv/{self.timestamp}/bid_prices.csv', sep=';')
        self.df_bid_agg.to_csv(f'./csv/{self.timestamp}/bid_agg.csv', sep=';')
        self.df_ask_agg.to_csv(f'./csv/{self.timestamp}/ask_agg.csv', sep=';')
        self.df_target_price_param.to_csv(f'./csv/{self.timestamp}/target_price_param.csv', sep=';')
        self.df_limit_buy.to_csv(f'./csv/{self.timestamp}/limit_buy.csv', sep=';')
        self.df_limit_sell.to_csv(f'./csv/{self.timestamp}/limit_sell.csv', sep=';')
        self.df_target_asks.to_csv(f'./csv/{self.timestamp}/target_asks.csv', sep=';')
        self.df_target_bids.to_csv(f'./csv/{self.timestamp}/target_bids.csv', sep=';')

        self.df_market_positions.to_csv(f'./csv/{self.timestamp}/market_positions.csv', sep=';')
        self.df_payoffs.to_csv(f'./csv/{self.timestamp}/payoffs.csv', sep=';')
        self.df_revenues.to_csv(f'./csv/{self.timestamp}/revenues.csv', sep=';')
        self.df_expenses.to_csv(f'./csv/{self.timestamp}/expenses.csv', sep=';')
        self.df_prod_costs.to_csv(f'./csv/{self.timestamp}/prod_costs.csv', sep=';')
        self.df_penalty_imbalances.to_csv(f'./csv/{self.timestamp}/penalty_imbalances.csv', sep=';')
        self.df_imbalances.to_csv(f'./csv/{self.timestamp}/imbalances.csv', sep=';')

        # Define state and reward
        reward = self.df_bidders.at[n - 1, 'payoff']

        self._current_step += 1

        if self._current_step >= self._max_steps:
            done = True

        obs = self.get_obs()

        if self.t_int == self.t_max:
            done = True  # breakpoint to check out graphs

        return obs, reward, done, truncated, {}
