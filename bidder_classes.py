import numpy as np
from collections import deque

# parent class of bidders

class Bidder:
    def __init__(self, has_seed=False):
        self.costs = (0.01, 11)
        self.sd_cap_start = 0.3
        self.aftermarket_exploration = 200

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