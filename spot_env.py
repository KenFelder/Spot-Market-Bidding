import os
from datetime import datetime
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from init import *
from DA_Auction import *
from id_cont import *
from price_calc import *
from config import *


class SpotEnv(gym.Env):
    def __init__(self, seed=False, log_frequency=1):
        """
        Spot Market Bidding Environment
        
        Args:
            seed: If True, use fixed seed for reproducibility
            log_frequency: How often to write detailed CSV logs (every N episodes)
                          Set to 1 to log every episode, 10 to log every 10th, etc.
        """
        if seed:
            self.rng = np.random.default_rng(seed=42)
        else:
            self.rng = np.random.default_rng()

        self.run = 0
        self.log_frequency = log_frequency

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

        self.old_payoff = 0

        ## Observation space
        self.df_obs = pd.DataFrame()
        self.df_actions = pd.DataFrame()
        self.df_rewards = pd.DataFrame()

        self.observation_space = spaces.Dict({
            # Private information
            'market_id': spaces.Discrete(2),                                                                    #  Feature 0
            'x_imb': spaces.Box(low=-np.inf, high=1e4, dtype=np.float64),                                       #  Feature 1
            'x_re_gen': spaces.Box(low=0, high=1500, dtype=np.float64),                                         #  Feature 2
            'x_th_gen': spaces.Box(low=0, high=300, dtype=np.float64),                                          #  Feature 3
            'x_th_start': spaces.Box(low=0, high=300, dtype=np.float64),                                        #  Feature 4
            'x_da': spaces.Box(low=0, high=np.inf, dtype=np.float64),                                           #  Feature 5
            'x_bought': spaces.Box(low=0, high=np.inf, dtype=np.float64),                                       #  Feature 6
            'x_sold': spaces.Box(low=0, high=np.inf, dtype=np.float64),                                         #  Feature 7
            'revenue': spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64),                                  #  Feature 8
            # Public information
            # Prices OB
            'Bid 0 (price)': spaces.Box(low=np.array(min_price), high=np.array(max_price), dtype=np.float64),   #  Feature 9
            'Bid 1 (price)': spaces.Box(low=np.array(min_price), high=np.array(max_price), dtype=np.float64),   #  Feature 10
            'Bid 2 (price)': spaces.Box(low=np.array(min_price), high=np.array(max_price), dtype=np.float64),   #  Feature 11
            'Bid 3 (price)': spaces.Box(low=np.array(min_price), high=np.array(max_price), dtype=np.float64),   #  Feature 12
            'Bid 4 (price)': spaces.Box(low=np.array(min_price), high=np.array(max_price), dtype=np.float64),   #  Feature 13
            'Bid 5 (price)': spaces.Box(low=np.array(min_price), high=np.array(max_price), dtype=np.float64),   #  Feature 14
            'Ask 0 (price)': spaces.Box(low=np.array(min_price), high=np.array(max_price), dtype=np.float64),   #  Feature 15
            'Ask 1 (price)': spaces.Box(low=np.array(min_price), high=np.array(max_price), dtype=np.float64),   #  Feature 16
            'Ask 2 (price)': spaces.Box(low=np.array(min_price), high=np.array(max_price), dtype=np.float64),   #  Feature 17
            'Ask 3 (price)': spaces.Box(low=np.array(min_price), high=np.array(max_price), dtype=np.float64),   #  Feature 18
            'Ask 4 (price)': spaces.Box(low=np.array(min_price), high=np.array(max_price), dtype=np.float64),   #  Feature 19
            'Ask 5 (price)': spaces.Box(low=np.array(min_price), high=np.array(max_price), dtype=np.float64),   #  Feature 20
            # Volumes OB
            'Bid 0 (volume)': spaces.Box(low=np.array(0), high=np.array(max_bid_volume), dtype=np.float64),     #  Feature 21
            'Bid 1 (volume)': spaces.Box(low=np.array(0), high=np.array(max_bid_volume), dtype=np.float64),     #  Feature 22
            'Bid 2 (volume)': spaces.Box(low=np.array(0), high=np.array(max_bid_volume), dtype=np.float64),     #  Feature 23
            'Bid 3 (volume)': spaces.Box(low=np.array(0), high=np.array(max_bid_volume), dtype=np.float64),     #  Feature 24
            'Bid 4 (volume)': spaces.Box(low=np.array(0), high=np.array(max_bid_volume), dtype=np.float64),     #  Feature 25
            'Bid 5 (volume)': spaces.Box(low=np.array(0), high=np.array(max_bid_volume), dtype=np.float64),     #  Feature 26
            'Ask 0 (volume)': spaces.Box(low=np.array(0), high=np.array(max_bid_volume), dtype=np.float64),     #  Feature 27
            'Ask 1 (volume)': spaces.Box(low=np.array(0), high=np.array(max_bid_volume), dtype=np.float64),     #  Feature 28
            'Ask 2 (volume)': spaces.Box(low=np.array(0), high=np.array(max_bid_volume), dtype=np.float64),     #  Feature 29
            'Ask 3 (volume)': spaces.Box(low=np.array(0), high=np.array(max_bid_volume), dtype=np.float64),     #  Feature 30
            'Ask 4 (volume)': spaces.Box(low=np.array(0), high=np.array(max_bid_volume), dtype=np.float64),     #  Feature 31
            'Ask 5 (volume)': spaces.Box(low=np.array(0), high=np.array(max_bid_volume), dtype=np.float64),     #  Feature 32
            'steps left': spaces.Box(low=0, high=self.t_max, dtype=np.float64),                                 #  Feature 33
        })

        ## Action space
        self.action_space = spaces.Box(low=np.array([min_price, -max_bid_volume]),
                                       high=np.array([max_price, max_ask_volume]), dtype=np.float64)

    def reset(self, seed=None, options=None):
        # log
        self.run += 1
        self.timestamp = datetime.now().strftime('%Y%m%d_%H-%M-%S')

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
        """Get observation - VecNormalize will handle normalization"""
        rl_agent_idx = n - 1
        
        # Get raw values directly (no manual normalization)
        x_imb = self.df_bidders.loc[rl_agent_idx, 'x_imb']
        x_re_gen = self.df_bidders.loc[rl_agent_idx, 'x_re_gen']
        x_th_gen = self.df_bidders.loc[rl_agent_idx, 'x_th_gen']
        x_th_start = self.df_bidders.loc[rl_agent_idx, 'x_th_start']
        x_da = self.df_bidders.loc[rl_agent_idx, 'x_da']
        x_bought = self.df_bidders.loc[rl_agent_idx, 'x_bought']
        x_sold = self.df_bidders.loc[rl_agent_idx, 'x_sold']
        revenue = self.df_bidders.loc[rl_agent_idx, 'revenue']

        if self._current_step == 0:
            obs = {
                'market_id': np.array([0]),  # '0' for day-ahead, '1' for intraday
                'x_imb': np.array([x_imb]),
                'x_re_gen': np.array([x_re_gen]),
                'x_th_gen': np.array([x_th_gen]),
                'x_th_start': np.array([x_th_start]),
                'x_da': np.array([x_da]),
                'x_bought': np.array([x_bought]),
                'x_sold': np.array([x_sold]),
                'revenue': np.array([revenue]),
                'Bid 0 (price)': np.array([min_price]),
                'Bid 1 (price)': np.array([min_price]),
                'Bid 2 (price)': np.array([min_price]),
                'Bid 3 (price)': np.array([min_price]),
                'Bid 4 (price)': np.array([min_price]),
                'Bid 5 (price)': np.array([min_price]),
                'Ask 0 (price)': np.array([max_price]),
                'Ask 1 (price)': np.array([max_price]),
                'Ask 2 (price)': np.array([max_price]),
                'Ask 3 (price)': np.array([max_price]),
                'Ask 4 (price)': np.array([max_price]),
                'Ask 5 (price)': np.array([max_price]),
                'Bid 0 (volume)': np.array([0.]),
                'Bid 1 (volume)': np.array([0.]),
                'Bid 2 (volume)': np.array([0.]),
                'Bid 3 (volume)': np.array([0.]),
                'Bid 4 (volume)': np.array([0.]),
                'Bid 5 (volume)': np.array([0.]),
                'Ask 0 (volume)': np.array([0.]),
                'Ask 1 (volume)': np.array([0.]),
                'Ask 2 (volume)': np.array([0.]),
                'Ask 3 (volume)': np.array([0.]),
                'Ask 4 (volume)': np.array([0.]),
                'Ask 5 (volume)': np.array([0.]),
                'steps left': np.array([self.t_int]),
            }
        else:
            bid_prices = self.df_order_book[self.df_order_book["bid_flag"] == 1]["price"]
            ask_prices = self.df_order_book[self.df_order_book["bid_flag"] == 0]["price"]
            bid_volumes = self.df_order_book[self.df_order_book["bid_flag"] == 1]["volume"]
            ask_volumes = self.df_order_book[self.df_order_book["bid_flag"] == 0]["volume"]

            len_bids = len(bid_prices)
            len_asks = len(ask_prices)

            if len_bids < 6:
                bid_prices = list(bid_prices) + [min_price] * (6 - len_bids)
                bid_volumes = list(bid_volumes) + [0.] * (6 - len_bids)
            else:
                bid_prices = list(bid_prices)[:6]
                bid_volumes = list(bid_volumes)[:6]

            if len_asks < 6:
                ask_prices = list(ask_prices) + [max_price] * (6 - len_asks)
                ask_volumes = list(ask_volumes) + [0.] * (6 - len_asks)
            else:
                ask_prices = list(ask_prices)[:6]
                ask_volumes = list(ask_volumes)[:6]

            obs = {
                'market_id': np.array([1]),
                'x_imb': np.array([x_imb]),
                'x_re_gen': np.array([x_re_gen]),
                'x_th_gen': np.array([x_th_gen]),
                'x_th_start': np.array([x_th_start]),
                'x_da': np.array([x_da]),
                'x_bought': np.array([x_bought]),
                'x_sold': np.array([x_sold]),
                'revenue': np.array([revenue]),
                'Bid 0 (price)': np.array(bid_prices[0]),
                'Bid 1 (price)': np.array(bid_prices[1]),
                'Bid 2 (price)': np.array(bid_prices[2]),
                'Bid 3 (price)': np.array(bid_prices[3]),
                'Bid 4 (price)': np.array(bid_prices[4]),
                'Bid 5 (price)': np.array(bid_prices[5]),
                'Ask 0 (price)': np.array(ask_prices[0]),
                'Ask 1 (price)': np.array(ask_prices[1]),
                'Ask 2 (price)': np.array(ask_prices[2]),
                'Ask 3 (price)': np.array(ask_prices[3]),
                'Ask 4 (price)': np.array(ask_prices[4]),
                'Ask 5 (price)': np.array(ask_prices[5]),
                'Bid 0 (volume)': np.array(bid_volumes[0]),
                'Bid 1 (volume)': np.array(bid_volumes[1]),
                'Bid 2 (volume)': np.array(bid_volumes[2]),
                'Bid 3 (volume)': np.array(bid_volumes[3]),
                'Bid 4 (volume)': np.array(bid_volumes[4]),
                'Bid 5 (volume)': np.array(bid_volumes[5]),
                'Ask 0 (volume)': np.array(ask_volumes[0]),
                'Ask 1 (volume)': np.array(ask_volumes[1]),
                'Ask 2 (volume)': np.array(ask_volumes[2]),
                'Ask 3 (volume)': np.array(ask_volumes[3]),
                'Ask 4 (volume)': np.array(ask_volumes[4]),
                'Ask 5 (volume)': np.array(ask_volumes[5]),
                'steps left': np.array([self.t_int]),
            }

        return obs

    def step(self, action):  # action is an array (price, volume)
        done = False
        truncated = False

        # Day-ahead auction
        if self._current_step == 0:

            init_new_round(self)

            if not include_rl:
                action = np.array([max_price, 0])

            max_sw(self, action)

            self.t_int += 1

        # Aftermarket exploration TODO: not implemented; Status now would overwrite actual DA data
        #elif self._current_step <= aftermarket_expl:

            #x_tmp, marginal_price_tmp, payments_tmp, sw = blind_auction(prices_da, residual_load, volumes_da)  #TODO: change to new function max_sw

            if include_rl:
                self.player = self.rng.integers(0, n)
            else:
                self.player = self.rng.integers(0, n-1)

            init_new_round(self)

        # Intraday auction
        else:
            while True:
                calc_prices(self)

                if self.player != n - 1:
                    new_post = bid_intra_trustful(self, self.player)
                else:
                    new_post = bid_intra_strategic(self, action, self.player)

                update_books(self, self.player, new_post)

                # Update in-memory dataframes (don't write to CSV yet!)
                self.df_x_demand.loc[self.t_int] = self.df_bidders['x_demand'].values
                self.df_x_bought.loc[self.t_int] = self.df_bidders['x_bought'].values
                self.df_x_sold.loc[self.t_int] = self.df_bidders['x_sold'].values
                self.df_x_re_cap.loc[self.t_int] = self.df_bidders['x_re_cap'].values
                self.df_x_re_gen.loc[self.t_int] = self.df_bidders['x_re_gen'].values
                self.df_x_th_gen.loc[self.t_int] = self.df_bidders['x_th_gen'].values
                self.df_x_th_start.loc[self.t_int] = self.df_bidders['x_th_start'].values

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
                    break

                if include_rl:
                    self.player = self.rng.integers(0, n)
                else:
                    self.player = self.rng.integers(0, n - 1)

                init_new_round(self)
                update_production(self, update_x_th_start=False)

                if self.player == n - 1:
                    break

        # Calculate reward
        new_payoff = self.df_bidders.at[n - 1, 'payoff']
        reward = new_payoff - self.old_payoff
        self.old_payoff = new_payoff

        # new rew function
        x_imb = self.df_bidders.at[n - 1, 'x_imb']
        imbalance_penalty_factor = self.df_game_data.iloc[-1]['imbalance_penalty_factor']
        bid_price = action[0] if action[1] < 0 else 0
        ask_price = action[0] if action[1] > 0 else 0
        bid_volume = -action[1] if action[1] < 0 else 0
        ask_volume = action[1] if action[1] > 0 else 0
        #

        reward = -imbalance_penalty_factor * abs(x_imb)

        # Get observation
        obs = self.get_obs()

        # Update logging dataframes (in memory only)
        self.df_obs = pd.concat([self.df_obs, pd.DataFrame(obs)], ignore_index=True)
        self.df_actions.loc[self.t_int, 'price'] = action[0]
        self.df_actions.loc[self.t_int, 'volume'] = action[1]
        self.df_rewards.loc[self.t_int, 'reward'] = reward

        self._current_step += 1

        if self._current_step >= self._max_steps:
            done = True

        if self.t_int == self.t_max:
            done = True

        # ONLY write CSVs when episode is done AND it's time to log
        if (done or truncated) and (self.run % self.log_frequency == 0):
            self._write_all_csvs()

        return obs, reward, done, truncated, {}
    
    def _write_all_csvs(self):
        """Write all CSV logs at episode end - called only when needed"""
        csv_dir = f'./csv/{self.timestamp}/'
        os.makedirs(csv_dir, exist_ok=True)
        
        # Write all CSV files once
        self.df_game_data.to_csv(f'{csv_dir}/game_data.csv', sep=';')
        self.df_bid_logs.to_csv(f'{csv_dir}/bid_logs.csv', sep=';')

        self.df_x_demand.to_csv(f'{csv_dir}/x_demand.csv', sep=';')
        self.df_x_bought.to_csv(f'{csv_dir}/x_bought.csv', sep=';')
        self.df_x_sold.to_csv(f'{csv_dir}/x_sold.csv', sep=';')
        self.df_x_re_cap.to_csv(f'{csv_dir}/x_re_cap.csv', sep=';')
        self.df_x_re_gen.to_csv(f'{csv_dir}/x_re_gen.csv', sep=';')
        self.df_x_th_gen.to_csv(f'{csv_dir}/x_th_gen.csv', sep=';')
        self.df_x_th_start.to_csv(f'{csv_dir}/x_th_start.csv', sep=';')

        self.df_ask_prices.to_csv(f'{csv_dir}/ask_prices.csv', sep=';')
        self.df_bid_prices.to_csv(f'{csv_dir}/bid_prices.csv', sep=';')
        self.df_bid_agg.to_csv(f'{csv_dir}/bid_agg.csv', sep=';')
        self.df_ask_agg.to_csv(f'{csv_dir}/ask_agg.csv', sep=';')
        self.df_target_price_param.to_csv(f'{csv_dir}/target_price_param.csv', sep=';')
        self.df_limit_buy.to_csv(f'{csv_dir}/limit_buy.csv', sep=';')
        self.df_limit_sell.to_csv(f'{csv_dir}/limit_sell.csv', sep=';')
        self.df_target_asks.to_csv(f'{csv_dir}/target_asks.csv', sep=';')
        self.df_target_bids.to_csv(f'{csv_dir}/target_bids.csv', sep=';')

        self.df_market_positions.to_csv(f'{csv_dir}/market_positions.csv', sep=';')
        self.df_payoffs.to_csv(f'{csv_dir}/payoffs.csv', sep=';')
        self.df_revenues.to_csv(f'{csv_dir}/revenues.csv', sep=';')
        self.df_expenses.to_csv(f'{csv_dir}/expenses.csv', sep=';')
        self.df_prod_costs.to_csv(f'{csv_dir}/prod_costs.csv', sep=';')
        self.df_penalty_imbalances.to_csv(f'{csv_dir}/penalty_imbalances.csv', sep=';')
        self.df_imbalances.to_csv(f'{csv_dir}/imbalances.csv', sep=';')

        self.df_bidders.to_csv(f'{csv_dir}/bidders.csv', sep=';')
        self.df_config.to_csv(f'{csv_dir}/config.csv', sep=';')

        self.df_obs.to_csv(f'{csv_dir}/obs.csv', sep=';')
        self.df_actions.to_csv(f'{csv_dir}/actions.csv', sep=';')
        self.df_rewards.to_csv(f'{csv_dir}/rewards.csv', sep=';')
