import numpy as np
from collections import deque


# function to normalize payoffs in [0,1]

def normalize_util(payoffs, min_payoff, max_payoff):
    if min_payoff == max_payoff:
        return payoffs
    payoff_range = max_payoff - min_payoff
    payoffs = np.maximum(payoffs, min_payoff)
    payoffs = np.minimum(payoffs, max_payoff)
    payoffs_scaled = (payoffs - min_payoff) / payoff_range
    return payoffs_scaled


normalize = np.vectorize(normalize_util)


# parent class of bidders

class Bidder:
    def __init__(self, has_seed=False):
        self.costs = (0.01, 11)
        self.sd_cap_start = 0.3
        self.aftermarket_exploration = 10

        # logs
        self.history_payoff_profile = []
        self.history_action = []
        self.history_payoff = []
        self.played_action = None
        # to be able to reproduce exact same behavior
        self.has_seed = has_seed
        if self.has_seed:
            self.seed = np.random.randint(1, 10000)
            self.random_state = np.random.RandomState(seed=self.seed)

    # To clear stored data
    def restart(self):
        self.history_payoff_profile = []
        self.history_action = []
        self.history_payoff = []
        self.played_action = None
        if self.has_seed:
            self.random_state = np.random.RandomState(seed=self.seed)

    # Player using Hedge algorithm (Freund and Schapire. 1997)


class Hedge_bidder(Bidder):
    def __init__(self, c_list, d_list, K, max_payoff, T, c_limit=None, d_limit=None, has_seed=False):
        super().__init__(c_list, d_list, K, c_limit=c_limit, d_limit=d_limit, has_seed=has_seed)
        self.type = 'Hedge'
        self.T = T
        self.learning_rate = np.sqrt(8 * np.log(self.K) / self.T)
        self.max_payoff = max_payoff

    def update_weights(self, payoffs):
        #payoffs = normalize(payoffs, 0, self.max_payoff)
        payoffs = normalize(payoffs, 0, np.max(payoffs))
        losses = np.ones(self.K) - np.array(payoffs)
        self.weights = np.multiply(self.weights, np.exp(np.multiply(self.learning_rate, -losses)))
        self.weights = self.weights / np.sum(self.weights)

        return losses.mean()

        # Player choosing actions uniformly random each time

class D4PG_bidder(Bidder):
    def __init__(self, T, has_seed=False):
        super().__init__(has_seed=has_seed)
        self.type = 'D4PG'
        self.T = T
        self.learning_rate = 0  # to fill
        self.cost = (0.01, 11)
        self.aftermarket_exploration = 10
        self.replay_buffer = deque(maxlen=1000)

