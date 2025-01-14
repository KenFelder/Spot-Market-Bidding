import numpy as np
import pandas as pd
from config import *


def update_limits(self, player, step_factor=0.1):
    limit_sell = self.df_bidders.at[player, 'limit_sell']
    limit_buy = self.df_bidders.at[player, 'limit_buy']
    imbalance_penalty_price = self.df_game_data.at[self.t_int, 'imbalance_penalty_factor']
    marginal_costs = self.df_bidders.at[player, 'marginal_costs']

    if self.df_bidders.at[player, 'x_imb'] > 0:
        #TODO won't ever rise, because limit_sell doesn't change
        limit_sell = (1 - step_factor) * limit_sell + step_factor * min(imbalance_penalty_price, marginal_costs, limit_sell, max_price)
    elif self.df_bidders.at[player, 'x_imb'] < 0:
        limit_buy = (1 - step_factor) * limit_buy + step_factor * max(imbalance_penalty_price, marginal_costs, limit_buy)

    self.df_bidders.at[player, 'limit_sell'] = limit_sell
    self.df_bidders.at[player, 'limit_buy'] = limit_buy

    return


def comp_price_estimate(transaction_prices, k_max=5):
    k = min(k_max, len(transaction_prices))
    lambda_hat = 1/k * np.sum(transaction_prices[-k:])
    return lambda_hat


def update_aggressiveness(aggressiveness, target_aggressiveness, step_factor):
    aggressiveness = aggressiveness + step_factor * (target_aggressiveness - aggressiveness)
    return aggressiveness

def update_target_aggressiveness_buy(lambda_hat, limit_buy, best_target_price_buy, target_price_param, aggressiveness_buy, delta_r=0.02, delta_a=0.01):
    if lambda_hat > limit_buy:
        #Extra-marginal buyer
        if best_target_price_buy >= limit_buy:
            target_bid_aggressiveness = aggressiveness_buy if aggressiveness_buy > 0 else 0
        else:
            target_bid_aggressiveness = -np.log(1 + (1 - best_target_price_buy / limit_buy) * (np.exp(target_price_param) - 1)) / target_price_param
    else:
        #Intra-marginal buyer
        if best_target_price_buy >= limit_buy:
            target_bid_aggressiveness = aggressiveness_buy if aggressiveness_buy > 0 else 0
        elif best_target_price_buy == lambda_hat:
            target_bid_aggressiveness = 0
        elif best_target_price_buy > lambda_hat:
            target_bid_aggressiveness = np.log((best_target_price_buy - lambda_hat) * (np.exp(target_price_param) - 1) / (limit_buy - lambda_hat) + 1) / target_price_param
        else:
            target_bid_aggressiveness = -np.log((1 - best_target_price_buy / lambda_hat) * (np.exp(target_price_param) - 1) + 1) / target_price_param
    return target_bid_aggressiveness

def update_target_aggressiveness_sell(lambda_hat, limit_sell, best_target_price_sell, target_price_param, max_price, aggressiveness_sell, delta_r=0.02, delta_a=0.01):
    if lambda_hat < limit_sell:
        #Extra-marginal seller
        if best_target_price_sell <= limit_sell:
            target_ask_aggressiveness = aggressiveness_sell if aggressiveness_sell > 0 else 0
        else:
            target_ask_aggressiveness = -np.log((best_target_price_sell - limit_sell) / (max_price - limit_sell) * (np.exp(target_price_param) - 1) + 1) / target_price_param
    else:
        #Intra-marginal seller
        if best_target_price_sell <= limit_sell:
            target_ask_aggressiveness = aggressiveness_sell if aggressiveness_sell > 0 else 0
        elif best_target_price_sell == lambda_hat:
            target_ask_aggressiveness = 0
        elif best_target_price_sell < lambda_hat:
            target_ask_aggressiveness = np.log((1 - (best_target_price_sell - limit_sell) / (lambda_hat - limit_sell)) * (np.exp(target_price_param) - 1) + 1) / target_price_param
        else:
            target_ask_aggressiveness = -np.log((best_target_price_sell - lambda_hat) * (np.exp(target_price_param) - 1) / (max_price - lambda_hat) + 1) / target_price_param
    return target_ask_aggressiveness

def update_target_price_param(target_price_param, target_price_param_scaled, step_factor):
    target_price_param = target_price_param + step_factor * (target_price_param_scaled - target_price_param)
    return target_price_param if target_price_param != 0 else 1e-5


def scale_target_price_param(volatility_scaled, target_price_param_max=2, target_price_param_min=-8, gamma=2):
    target_price_param_scaled = ((target_price_param_max - target_price_param_min) * (1 - volatility_scaled *
                                                                                     np.exp(gamma * (volatility_scaled -
                                                                                                     1))) +
                                 target_price_param_min)
    return target_price_param_scaled


def scale_volatility(volatility, volatility_max, volatility_min):
    if volatility_max == volatility_min:
        return 0
    elif volatility > volatility_max:
        volatility_max = volatility
    return (volatility - volatility_min) / (volatility_max - volatility_min)


def calc_volatility(transaction_prices, lambda_hat, i_max=5):
    i = min(len(transaction_prices), i_max)
    volatility = np.sqrt(1/i * np.sum([(transaction_prices[-j] - lambda_hat) ** 2 for j in range(i)])) / lambda_hat
    return volatility


def calc_target_price(lambda_hat, lambda_max, limit_buy, limit_sell, aggressiveness_buy, aggressiveness_sell,
                      target_price_param):
    # Buy side
    # Intra-marginal buyer
    if limit_buy > lambda_hat:
        if aggressiveness_buy >= 0:
            target_price_buy = (lambda_hat + (limit_buy - lambda_hat) *
                                (np.exp(aggressiveness_buy * target_price_param) - 1)/(np.exp(target_price_param) - 1))
        else:
            target_price_buy = (lambda_hat * (1 - (np.exp(-aggressiveness_buy * target_price_param) - 1) /
                                              (np.exp(target_price_param) - 1)))
    # Extra-marginal buyer
    else:
        if aggressiveness_buy >= 0:
            target_price_buy = limit_buy
        else:
            target_price_buy = limit_buy * (1 - ((np.exp(-aggressiveness_buy * target_price_param)-1) /
                                            (np.exp(target_price_param)-1)))

    # Sell side
    # Intra-marginal seller
    if limit_sell < lambda_hat:
        if aggressiveness_sell >= 0:
            target_price_sell = (limit_sell + (lambda_hat - limit_sell) *
                                 (1 - (np.exp(aggressiveness_sell * target_price_param) - 1) /
                                  (np.exp(target_price_param) - 1)))
        else:
            target_price_sell = (lambda_hat + (lambda_max - lambda_hat) *
                                 (np.exp(-aggressiveness_sell * target_price_param) - 1) /
                                 (np.exp(target_price_param) - 1))
    # Extra-marginal seller
    else:
        if aggressiveness_sell >= 0:
            target_price_sell = limit_sell
        else:
            target_price_sell = (limit_sell + (lambda_max - limit_sell) *
                                 (np.exp(-aggressiveness_sell * target_price_param) - 1) /
                                 (np.exp(target_price_param) - 1))
    return target_price_buy, target_price_sell


def calc_prices(self):
    transaction_prices = self.df_game_data['transaction_price'].dropna().tolist()
    equilibrium_price_estimate = comp_price_estimate(transaction_prices)
    last_price = self.df_game_data['last_price'].dropna().iloc[-1]
    last_event = self.df_game_data['last_event'].dropna().iloc[-1]
    volatilities = self.df_game_data['volatility'].dropna().tolist()

    for player in self.df_bidders.index:
        update_limits(self, player)
        # initialize values
        limit_buy = self.df_bidders.at[player, 'limit_buy']
        limit_sell = self.df_bidders.at[player, 'limit_sell']
        target_price_param = self.df_bidders.at[player, 'target_price_param']
        target_price_param_step_factor = self.df_bidders.at[player, 'target_price_param_step_factor']
        aggressiveness_buy = self.df_bidders.at[player, 'aggressiveness_buy']
        aggressiveness_sell = self.df_bidders.at[player, 'aggressiveness_sell']
        aggressiveness_step_factor = self.df_bidders.at[player, 'aggressiveness_step_factor']
        bid_step_factor = self.df_bidders.at[player, 'bid_step_factor']

        len_bids = len(self.df_order_book[self.df_order_book["bid_flag"] == 1])
        len_asks = len(self.df_order_book[self.df_order_book["bid_flag"] == 0])

        top_ask_price = self.df_order_book[self.df_order_book["bid_flag"] == 0].iloc[-1]["price"] if len_asks > 0 else None
        top_bid_price = self.df_order_book[self.df_order_book["bid_flag"] == 1].iloc[0]["price"] if len_bids > 0 else None


        volatility = calc_volatility(transaction_prices, equilibrium_price_estimate)

        if last_event == 'match' and self.t_int > 1:
            volatility_scaled = scale_volatility(volatility, max(volatilities), min(volatilities))
            target_price_param_scaled = scale_target_price_param(volatility_scaled)
            if len(transaction_prices) >= 5:
                target_price_param = update_target_price_param(target_price_param, target_price_param_scaled, target_price_param_step_factor)
            t_agg_buy = update_target_aggressiveness_buy(equilibrium_price_estimate, limit_buy, last_price, target_price_param, aggressiveness_buy)
            t_agg_sell = update_target_aggressiveness_sell(equilibrium_price_estimate, limit_sell, last_price, target_price_param,
                                                           max_price, aggressiveness_sell)
            aggressiveness_buy = update_aggressiveness(aggressiveness_buy, t_agg_buy, aggressiveness_step_factor)
            aggressiveness_sell = update_aggressiveness(aggressiveness_sell, t_agg_sell, aggressiveness_step_factor)
        elif last_event == 'bid' and len_bids > 0:
            t_agg_buy = update_target_aggressiveness_buy(equilibrium_price_estimate, limit_buy, top_bid_price, target_price_param, aggressiveness_buy)
            t_agg_buy = t_agg_buy if t_agg_buy > aggressiveness_buy else aggressiveness_buy
            aggressiveness_buy = update_aggressiveness(aggressiveness_buy, t_agg_buy, aggressiveness_step_factor)
        elif last_event == 'ask' and len_asks > 0:
            t_agg_sell = update_target_aggressiveness_sell(equilibrium_price_estimate, limit_sell, top_ask_price, target_price_param,
                                                           max_price, aggressiveness_sell)
            t_agg_sell = t_agg_sell if t_agg_sell > aggressiveness_sell else aggressiveness_sell
            aggressiveness_sell = update_aggressiveness(aggressiveness_sell, t_agg_sell, aggressiveness_step_factor)

        target_price_buy, target_price_sell = calc_target_price(equilibrium_price_estimate, max_price, limit_buy, limit_sell,
                                                                aggressiveness_buy, aggressiveness_sell, target_price_param)

        top_ask_price = self.df_order_book[self.df_order_book["bid_flag"] == 0].iloc[-1]["price"] if len_asks > 0 else target_price_sell
        top_bid_price = self.df_order_book[self.df_order_book["bid_flag"] == 1].iloc[0]["price"] if len_bids > 0 else target_price_buy

        ask_price = top_ask_price - (top_ask_price - target_price_sell) / bid_step_factor
        bid_price = top_bid_price + (target_price_buy - top_bid_price) / bid_step_factor

        # save updated values
        self.df_bidders.at[player, 'target_price_param'] = target_price_param
        self.df_bidders.at[player, 'aggressiveness_buy'] = aggressiveness_buy
        self.df_bidders.at[player, 'aggressiveness_sell'] = aggressiveness_sell
        self.df_bidders.at[player, 'bid_price'] = bid_price
        self.df_bidders.at[player, 'ask_price'] = ask_price

        self.df_target_bids.at[self.t_int, f'bidder_{player}'] = target_price_buy
        self.df_target_asks.at[self.t_int, f'bidder_{player}'] = target_price_sell

    # Log game data
    self.df_game_data.at[self.t_int, 'volatility'] = volatility
    self.df_game_data.at[self.t_int, 'equilibrium_price_estimate'] = equilibrium_price_estimate
    self.df_game_data.at[self.t_int, 'market'] = 'ID'

    return
