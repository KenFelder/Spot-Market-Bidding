import numpy as np
import cvxpy as cp

def optimize_alloc(p_bid, bids, Q, cap):
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
    prob.solve()
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

def calc_payoff_DA(N, payments, x, other_costs, bidder):
    payoff = []
    for i in range(N):
        if i == N - 1:
            payoff_bidder = payments[-1] - (0.5 * bidder.costs[0] * x[-1] + bidder.costs[1]) * x[-1]
            payoff.append(payoff_bidder)
        else:
            payoff_bidder = payments[i] - (0.5 * other_costs[i][0] * x[i] + other_costs[i][1]) * x[i]
            payoff.append(payoff_bidder)
    return payoff

def aftermarket_evaluation(bids, Q, cap, t, bidder):
    payoffs_each_action = []
    for _ in range(bidder.aftermarket_exploration):
        # action = (price, quantity), random as placeholder
        action_tmp = (np.random.randint(10, 30), np.random.randint(10, cap[-1]))
        tmp_bids = bids.copy()
        x_tmp, marginal_price_tmp, payments_tmp, sw = optimize_alloc(action_tmp[0], tmp_bids, Q, cap[:-1] + [action_tmp[1]])
        payoff_tmp = payments_tmp[-1] - (0.5 * bidder.costs[0] * x_tmp[-1] + bidder.costs[1]) * x_tmp[-1]
        payoffs_each_action.append(payoff_tmp)
        state_new = [x_tmp[-1], marginal_price_tmp, Q]
        #bidder.replay_buffer.append((state_tmp, _tmp, payoff_tmp, state_new))
    bidder.history_payoff_profile.append(np.array(payoffs_each_action))
    #regret = (max(bidder.cum_each_action) - sum(bidder.history_payoff)) / (t + 1)
    #return regret