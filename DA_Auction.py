import numpy as np
import cvxpy as cp
import pandas as pd
from config import *


def max_sw(self, action):
    prices = np.array([max_price] + self.df_bidders['true_costs'][1:-1].tolist() + [action[0]])
    volumes = np.array([self.x_demand[self.t_int]] + self.df_bidders['x_cap'][1:-1].tolist() + [action[1]])

    x_re_caps = np.array([self.df_bidders.at[i, 'x_re_cap'] if volumes[i] > 0 else 0 for i in range(n)])
    x_th_caps = np.array([self.df_bidders.at[i, 'x_th_cap'] if volumes[i] > 0 else 0 for i in range(n)])

    x_re_gen = cp.Variable(n, nonneg=True)
    x_th_gen = cp.Variable(n, nonneg=True)
    x_dem = cp.Variable(n, nonneg=True)

    c_dem = cp.sum(cp.multiply(prices, x_dem))
    c_prod = cp.sum(cp.multiply(prices, x_th_gen))

    sw = c_dem - c_prod

    sw = cp.Problem(cp.Maximize(sw), [
        cp.sum(x_dem) == cp.sum(x_re_gen) + cp.sum(x_th_gen),
        x_re_gen <= x_re_caps,
        x_th_gen <= x_th_caps,
        x_dem <= np.array([-volumes[i] if volumes[i] < 0 else 0 for i in range(n)]),
    ])

    sw.solve(solver=cp.GUROBI)

    # smallest volume that is greater than 0 sets price
    marginal_price = prices[min((i for i, x in enumerate(x_th_gen.value) if x > 0),
                                     key=lambda i: x_th_gen.value[i])]

    marginal_costs = [self.df_bidders['true_costs'][i] if x_th_gen.value[i] > 0 else 0 for i in range(n)]

    self.df_bidders = self.df_bidders.assign(
        x_re_gen=x_re_gen.value,
        x_th_gen=x_th_gen.value,
        x_da=-x_dem.value + x_re_gen.value + x_th_gen.value,
        x_prod=x_re_gen.value + x_th_gen.value,
        x_imb=0.0,
        marginal_costs=marginal_costs,
        production_costs=x_th_gen.value * self.df_bidders['true_costs'],
        revenue=marginal_price * (x_re_gen.value + x_th_gen.value),
        expenses=marginal_price * x_dem.value + x_th_gen.value * self.df_bidders['true_costs'],
        payoff=marginal_price * (x_re_gen.value + x_th_gen.value) -
               marginal_price * x_dem.value -
               x_th_gen.value * self.df_bidders['true_costs'],
        limit_buy=np.maximum([self.imbalance_penalty_factor] * n, marginal_costs),
        limit_sell=np.minimum([self.imbalance_penalty_factor] * n, marginal_costs),
    )

    self.df_game_data.at[self.t_int, 'market'] = 'DA'
    self.df_game_data.at[self.t_int, 'marginal_price'] = marginal_price
    self.df_game_data.at[self.t_int, 'social_welfare'] = sw.value
    self.df_game_data.at[self.t_int, 'system_imbalance'] = self.df_bidders['x_imb'].sum()
    self.df_game_data.at[self.t_int, 'imbalance_penalty_factor'] = self.imbalance_penalty_factor
    self.df_game_data.at[self.t_int, 'last_event'] = 'match'
    self.df_game_data.at[self.t_int, 'last_price'] = marginal_price
    self.df_game_data.at[self.t_int, 'transaction_price'] = marginal_price
    self.df_game_data.at[self.t_int, 'volatility'] = 0
    self.df_game_data.at[self.t_int, 'equilibrium_price_estimate'] = marginal_price

    return

####################################################################
############################## Archiv ##############################
####################################################################
def blind_auction(bids, Q, cap):
    D = np.array(bids)
    n = len(bids)
    A = np.ones(n).T
    I = np.eye(n)

    # non-negativity doesn't strictly hold (small negative allocations might occur)
    x = cp.Variable(n, nonneg=True)
    prob = cp.Problem(cp.Minimize(D.T @ x),
                      [A @ x == Q, I @ x <= cap])
    prob.solve(solver=cp.GUROBI)
    allocs = x.value
    social_cost = prob.value
    # To fix very small values
    for i in range(len(allocs)):
        if allocs[i] < 1e-5:
            allocs[i] = 0

    prices = [bids[i] if allocs[i] > 0 else 0 for i in range(len(allocs))]
    marginal_price = np.max(prices)
    payments = marginal_price * allocs

    return allocs, marginal_price, payments, social_cost

def calc_payoff_DA(n, marginal_price, x_da, costs, x_re_gen):
    payoff = []
    for i in range(n):
        payoff_bidder = (marginal_price - costs[i]) * x_da[i] + marginal_price * x_re_gen[i]
        payoff.append(payoff_bidder)
    return payoff

def optimize_alloc(p_bid, bids, Q, cap):
    print("optimize_alloc (archived)")
    C = np.array([param[0] for param in bids])
    C = np.diag(C)
    D = np.array([param[1] for param in bids])
    n = len(bids) + 1
    A = np.ones(n).T
    I = np.eye(n)

    # non-negativity doesn't strictly hold (small negative allocations might occur)
    x = cp.Variable(n, nonneg=True)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x[:n-1], C) + D.T @ x[:n-1] + p_bid * x[-1]),
                      [A @ x == Q, I @ x <= cap])
    prob.solve(solver=cp.GUROBI)
    allocs = x.value
    social_welfare = prob.value
    # To fix very small values
    for i in range(len(allocs)):
        if allocs[i] < 1e-5:
            allocs[i] = 0


    prices = [bids[i][0] * allocs[i] + bids[i][1] if allocs[i] > 0 else 0 for i in range(len(allocs)-1)] + [p_bid if allocs[-1] > 0 else 0]
    marginal_price = np.max(prices)
    payments = marginal_price * allocs

    return allocs, marginal_price, payments, social_welfare

def aftermarket_evaluation(bids, Q, cap, t, bidder):
    payoffs_each_action = []
    for _ in range(bidder.aftermarket_exploration):
        # action = (price, quantity), random as placeholder
        action_tmp = (np.random.randint(0, 30), np.random.randint(0, cap[-1]))
        tmp_bids = bids.copy()
        x_tmp, marginal_price_tmp, payments_tmp, sw = blind_auction(tmp_bids, Q, cap[:-1] + [action_tmp[1]])
        payoff_tmp = payments_tmp[-1] - bidder.costs * x_tmp[-1]
        payoffs_each_action.append(payoff_tmp)
        state_new = [x_tmp[-1], marginal_price_tmp, Q]
        #bidder.replay_buffer.append((state_tmp, _tmp, payoff_tmp, state_new))
    bidder.history_payoff_profile.append(np.array(payoffs_each_action))
    #regret = (max(bidder.cum_each_action) - sum(bidder.history_payoff)) / (t + 1)
    #return regret