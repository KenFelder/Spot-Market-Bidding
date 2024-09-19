import numpy as np
from tqdm import tqdm
import pickle
from DA_Auction import optimize_alloc, calc_payoff_DA, aftermarket_evaluation
from id_cont import bid_intra_trustful, update_books
from bidder_classes import D4PG_bidder
import pandas as pd


class auction_data:
    def __init__(self):
        self.allocations = []
        self.payments = []
        self.marginal_prices = []
        self.payoffs = []
        self.Q = []
        self.SW = []
        self.final_dist = None


class player_data:
    def __init__(self):
        self.bids = []
        self.payoffs_each_action = []
        self.history_weights = []
        self.regrets = []
        self.losses = []


def trustful_vs_bidder(bidder, num_runs, T, T_ID, file_name):
    types = ['Trustful vs D4PG']
    game_data_profile = [[]]
    N = 5
    c_cost_bidder = [0.01]  # last player
    d_cost_bidder = [11]

    # Actions of others obtained from diagonalization + their true cost
    other_costs = [(0.07, 9), (0.02, 10), (0.03, 12), (0.008, 12)]

    Q = 1448.4
    cap_mean = 700
    # interpolate from 0.1 to 0 in 200 steps
    sd_cap = [np.linspace(0.1, 0, T_ID),
              np.linspace(0.15, 0, T_ID),
              np.linspace(0.2, 0, T_ID),
              np.linspace(0.25, 0, T_ID),
              np.linspace(0.3, 0, T_ID)]

    player_final_dists = []
    for run in tqdm(range(num_runs)):

        # initialize
        bidder.restart()
        game_data_DA = auction_data()
        game_data_player = player_data()

        data_ob = {
            "bid_flag": [],
            "price": [],
            "volume": [],
            "participant": [],
            "timestamp": []
        }
        df_order_book = pd.DataFrame(data_ob)
        df_order_book = df_order_book.astype({
            'bid_flag': 'int64',
            'price': 'float64',
            'volume': 'float64',
            'participant': 'int64',
            'timestamp': 'int64'
        })

        data_bidders = {
            "x_bought": [0] * N,
            "x_sold": [0] * N,
            "x_da": [0] * N,
            "x_imb": [0] * N,
            "x_prod": [0] * N,
            "x_cap": [0] * N,
            "true_costs": other_costs + [(c_cost_bidder[0], d_cost_bidder[0])],
            "lambda_hat_int": [0] * N,
            "revenue": [0] * N
        }

        df_bidders = pd.DataFrame(data_bidders)

        # Assign dtype to specific columns
        df_bidders = df_bidders.astype({
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

        # Training Loop / Game
        for t in range(T):

            # Day-ahead auction
            cap = [np.random.normal(loc=cap_mean, scale=sd_cap[i][0] * cap_mean) for i in range(N)]
            # action = (price, quantity), random as placeholder
            action = (np.random.randint(10, 30), np.random.randint(10, cap[-1]))
            bidder.played_action = action
            bidder.history_action.append(action)
            bids = other_costs

            x, marginal_price, payments, social_welfare = optimize_alloc(action[0], bids, Q, cap[:-1] + [action[1]])

            payoff = calc_payoff_DA(N, payments, x, other_costs, bidder)
            game_data_DA.payoffs.append(payoff)

            for i in range(N):
                df_bidders.at[i, 'x_da'] = x[i]
                df_bidders.at[i, 'x_prod'] = x[i]
                df_bidders.at[i, 'revenue'] += payments[i]
                df_bidders.at[i, 'x_cap'] = cap[i]

            payoff_bidder = payoff[-1]

            state = [0, 0, 0]
            #state = budget
            action = action
            reward = payoff_bidder
            #state_new = [x[-1], marginal_price, Q]
            state_new = [x[-1], marginal_price, Q]

            transition = (state, action, reward, state_new)

            bidder.replay_buffer.append(transition)

            regret = aftermarket_evaluation(bids, Q, cap, t, bidder)

            # Intraday Continuous
            for t_int in range(T_ID):
                cap = [np.random.normal(loc=cap_mean, scale=sd_cap[i][t_int] * cap_mean) for i in range(N)]

                player = np.random.randint(0, len(df_bidders))
                # Placeholder: set random lambda_hat_int for bidder
                df_bidders.at[player, 'lambda_hat_int'] = np.random.randint(15, 30)
                df_bidders.at[player, 'x_cap'] = cap[player]
                x_prod, x_imb, new_post = bid_intra_trustful(player, df_bidders, T_ID, t_int)

                df_order_book, df_bidders, df_match_book = update_books(df_order_book, df_bidders, player, new_post, x_prod, x_imb)


            print(df_bidders)


            # if t % bidder.target_update_freq == 0:
            #     for i in range(bidder.training_epochs):
            #         bidder.update_actor_weights()
            #     bidder.update_target_actor_model()
            #     bidder.update_target_critic_model()

            #game_data.regrets.append([regret])

            # store data
            game_data_DA.Q.append(Q)
            game_data_DA.SW.append(social_welfare)
            game_data_DA.allocations.append(x)
            game_data_DA.payments.append(payments)
            game_data_DA.marginal_prices.append(marginal_price)

            game_data_player.bids.append(bids)

        player_final_dists.append(bidder.weights)
        game_data_profile[0].append(game_data_DA)

    with open(f'{file_name}.pckl', 'wb') as file:
        pickle.dump(T, file)
        pickle.dump(types, file)
        pickle.dump(game_data_profile, file)
        pickle.dump(player_final_dists, file)

if __name__ == '__main__':
    T = 200
    T_ID = 200
    bidder = D4PG_bidder(T)
    trustful_vs_bidder(bidder, 1, T, T_ID, 'test')
