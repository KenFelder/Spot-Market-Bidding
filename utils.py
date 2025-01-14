import numpy as np
import cvxpy as cp
from config import *


def init_new_round(self):
    # Update imbalance penalty
    self.imbalance_penalty_factor = np.exp(2 * self.t_int * np.log(10) / t_max)
    #self.imbalance_penalty_factor = 15
    self.df_game_data.at[self.t_int, 'imbalance_penalty_factor'] = self.imbalance_penalty_factor

    # Update forecasts
    for i in range(n):
        self.df_bidders.at[i, 'x_re_cap'] = self.x_re_cap[i][self.t_int]
        self.df_bidders.at[i, 'x_cap'] = self.df_bidders.at[i, 'x_re_cap'] + self.df_bidders.at[i, 'x_th_cap']
    self.df_bidders.at[0, 'x_demand'] = self.x_demand[self.t_int]
    return

def update_production(self):
    for player in range(n):
        x_th_start = self.df_bidders.at[player, 'x_th_gen']

        a = self.df_bidders.at[player, 'true_costs']

        x_demand = self.df_bidders.at[player, 'x_demand']
        x_da = self.df_bidders.at[player, 'x_da']
        x_bought = self.df_bidders.at[player, 'x_bought']
        x_sold = self.df_bidders.at[player, 'x_sold']

        x_re_cap = self.df_bidders.at[player, 'x_re_cap']
        x_th_cap = self.df_bidders.at[player, 'x_th_cap']

        x_re_gen = cp.Variable(nonneg=True)
        x_th_gen = cp.Variable(nonneg=True)
        x_imb = cp.Variable()

        cost_prod = a * x_th_gen
        # 1e-5 to prefer x_re_gen over x_imb at t_int = 0
        imb_penalty = (self.imbalance_penalty_factor + 1e-5) * cp.abs(x_imb)

        objective = cp.Minimize(cost_prod + imb_penalty)

        constraints = [
            x_demand + x_th_gen + x_re_gen + x_bought == x_da + x_sold + x_imb,
            x_th_gen <= x_th_cap,
            cp.abs(x_th_gen - x_th_start) <= max((1 - self.t_int / (t_max * 0.9)) * x_th_cap, 0),
            # TODO: decide if re must be fed-in; if changed also change in bid_intra_trustful
            x_re_gen == x_re_cap,
            #x_re_gen <= x_re_cap,
        ]

        problem = cp.Problem(objective, constraints)

        problem.solve(solver=cp.GUROBI)

        self.df_bidders.loc[player, 'x_re_gen'] = x_re_gen.value
        self.df_bidders.loc[player, 'x_th_gen'] = x_th_gen.value
        self.df_bidders.loc[player, 'x_imb'] = x_imb.value
        self.df_bidders.loc[player, 'x_prod'] = x_re_gen.value + x_th_gen.value
        self.df_bidders.loc[player, 'marginal_costs'] = self.df_bidders.at[player, 'true_costs'] \
            if x_th_gen.value > 0 else 0
        self.df_bidders.loc[player, 'production_costs'] = cost_prod.value
        self.df_bidders.loc[player, 'penalty_imbalance'] = imb_penalty.value
        self.df_bidders.loc[player, 'expenses'] = (x_th_gen.value * self.df_bidders.at[player, 'true_costs']
                                                   + self.imbalance_penalty_factor * abs(x_imb.value)
                                                   )
        self.df_bidders.loc[player, 'payoff'] = (
                self.df_bidders.at[player, 'revenue'] - self.df_bidders.at[player, 'expenses']
        )

    self.df_game_data.at[self.t_int, 'system_imbalance'] = sum(self.df_bidders['x_imb'])
    return
