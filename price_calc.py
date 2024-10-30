import numpy as np


def update_limits(limit_buy, limit_sell, lambda_max, imbalance, imbalance_penalty, step_factor=0.1):
    if imbalance > 0:
        limit_sell = (1 - step_factor) * limit_sell + step_factor * min(imbalance_penalty, limit_sell, lambda_max)
    elif imbalance < 0:
        limit_buy = (1 - step_factor) * limit_buy + step_factor * max(imbalance_penalty, limit_buy)
    return limit_buy, limit_sell


def comp_price_estimate(transaction_prices, k_max=5):
    k = min(k_max, len(transaction_prices))
    lambda_hat = 1/k * np.sum(transaction_prices[-k:])
    return lambda_hat


def update_aggressiveness(aggressiveness, target_aggressiveness, step_factor):
    aggressiveness = aggressiveness + step_factor * (target_aggressiveness - aggressiveness)
    return aggressiveness


def update_target_aggressiveness_buy(lambda_hat, limit_buy, last_target_price_buy, target_price_param, delta_r=0.02, delta_a=0.01):
    # when target_price_param = 0 -> log(0) is undefined
    if lambda_hat == last_target_price_buy <= limit_buy:
        target_bid_aggressiveness = 0
    elif lambda_hat <= last_target_price_buy < limit_buy:
        target_bid_aggressiveness = np.log((last_target_price_buy - lambda_hat) * (np.exp(target_price_param) - 1) /
                                           (limit_buy - lambda_hat) + 1) / target_price_param
    elif 0 <= last_target_price_buy < lambda_hat:
        target_bid_aggressiveness = -np.log((1 - last_target_price_buy / lambda_hat) * (np.exp(target_price_param) - 1) +
                                            1) / target_price_param
    elif 0 <= last_target_price_buy < limit_buy:
        target_bid_aggressiveness = -np.log((1 - last_target_price_buy / limit_buy) * (np.exp(target_price_param) - 1) +
                                            1) / target_price_param
    elif limit_buy <= last_target_price_buy:
        target_bid_aggressiveness = 1
    t_agg_bid = (1 + delta_r) * target_bid_aggressiveness + delta_a
    return t_agg_bid


def update_target_aggressiveness_sell(lambda_hat, limit_sell, last_target_price_sell, target_price_param, max_price, delta_r=0.02, delta_a=0.01):
    if lambda_hat == last_target_price_sell >= limit_sell:
        target_ask_aggressiveness = 0
    elif lambda_hat < last_target_price_sell <= max_price:
        target_ask_aggressiveness = -np.log((last_target_price_sell - lambda_hat) * (np.exp(target_price_param) - 1) /
                                            (max_price - lambda_hat) + 1) / target_price_param
    elif limit_sell < last_target_price_sell < lambda_hat:
        target_ask_aggressiveness = np.log((1 - (last_target_price_sell - limit_sell) / (lambda_hat - limit_sell)) *
                                            (np.exp(target_price_param) - 1) + 1) / target_price_param
    elif limit_sell < last_target_price_sell < max_price:
        target_ask_aggressiveness = -np.log((last_target_price_sell - limit_sell) / (max_price - 1) *
                                            (np.exp(target_price_param) - 1) + 1) / target_price_param
    elif last_target_price_sell <= limit_sell:
        target_ask_aggressiveness = 1
    t_agg_ask = (1 + delta_r) * target_ask_aggressiveness + delta_a
    return t_agg_ask


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


def calc_volatility(transaction_prices, lambda_hat, n_max=5):
    n = min(len(transaction_prices), n_max)
    volatility = np.sqrt(1/n * np.sum([transaction_prices[i] - lambda_hat for i in range(n)]) ** 2) / lambda_hat
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
            target_price_buy = limit_buy * (1 - (np.exp(-aggressiveness_buy * target_price_param)-1) /
                                            (np.exp(target_price_param)-1))

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


def calc_prices(t, t_l, transaction_prices, last_price, df_bidders, df_order_book, volatilities, last_event,
               lambda_max):
    equilibrium_price_estimate = comp_price_estimate(transaction_prices)

    for bidder in df_bidders.index:
        # initialize values
        limit_buy = df_bidders.at[bidder, 'limit_buy']
        limit_sell = df_bidders.at[bidder, 'limit_sell']
        imbalance = df_bidders.at[bidder, 'x_imb']
        target_price_param = df_bidders.at[bidder, 'target_price_param']
        target_price_param_step_factor = df_bidders.at[bidder, 'target_price_param_step_factor']
        aggressiveness_buy = df_bidders.at[bidder, 'aggressiveness_buy']
        aggressiveness_sell = df_bidders.at[bidder, 'aggressiveness_sell']
        aggressiveness_step_factor = df_bidders.at[bidder, 'aggressiveness_step_factor']
        bid_step_factor = df_bidders.at[bidder, 'bid_step_factor']

        imbalance_penalty = t / (t_l - t + 1e-5) * np.abs(imbalance)

        limit_buy, limit_sell = update_limits(limit_buy, limit_sell, lambda_max, imbalance, imbalance_penalty)

        volatility = calc_volatility(transaction_prices, equilibrium_price_estimate)

        if volatility != 0:
            volatility = volatility

        target_price_buy, target_price_sell = calc_target_price(equilibrium_price_estimate, lambda_max, limit_buy, limit_sell,
                                                                aggressiveness_buy, aggressiveness_sell, target_price_param)

        if last_event == 'match':
            volatility_scaled = scale_volatility(volatility, max(volatilities), min(volatilities))
            target_price_param_scaled = scale_target_price_param(volatility_scaled)
            target_price_param = update_target_price_param(target_price_param, target_price_param_scaled,
                                                           target_price_param_step_factor)
            t_agg_buy = update_target_aggressiveness_buy(equilibrium_price_estimate, limit_buy, last_price, target_price_param)
            t_agg_sell = update_target_aggressiveness_sell(equilibrium_price_estimate, limit_sell, last_price, target_price_param,
                                                           lambda_max)
            aggressiveness_buy = update_aggressiveness(aggressiveness_buy, t_agg_buy, aggressiveness_step_factor)
            aggressiveness_sell = update_aggressiveness(aggressiveness_sell, t_agg_sell, aggressiveness_step_factor)
        elif last_event == 'bid' and target_price_buy <= last_price:
            t_agg_buy = update_target_aggressiveness_buy(equilibrium_price_estimate, limit_buy, last_price, target_price_param)
            aggressiveness_buy = update_aggressiveness(aggressiveness_buy, t_agg_buy, aggressiveness_step_factor)
        elif last_event == 'ask' and target_price_sell >= last_price:
            t_agg_sell = update_target_aggressiveness_sell(equilibrium_price_estimate, limit_sell, last_price, target_price_param,
                                                           lambda_max)
            aggressiveness_sell = update_aggressiveness(aggressiveness_sell, t_agg_sell, aggressiveness_step_factor)

        target_price_buy, target_price_sell = calc_target_price(equilibrium_price_estimate, lambda_max, limit_buy, limit_sell,
                                                                aggressiveness_buy, aggressiveness_sell, target_price_param)

        len_bids = len(df_order_book[df_order_book["bid_flag"] == 1])
        len_asks = len(df_order_book[df_order_book["bid_flag"] == 0])
        if len_asks > 0:
            best_ask_price = df_order_book[df_order_book["bid_flag"] == 0].iloc[-1]["price"]
        else:
            best_ask_price = equilibrium_price_estimate  # TODO: Think this through
        if len_bids > 0:
            best_bid_price = df_order_book[df_order_book["bid_flag"] == 1].iloc[0]["price"]
        else:
            best_bid_price = equilibrium_price_estimate  # TODO: Think this through

        ask_price = best_ask_price - (best_ask_price - target_price_sell) / bid_step_factor
        bid_price = best_bid_price + (target_price_buy - best_bid_price) / bid_step_factor

        # save updated values
        df_bidders.at[bidder, 'limit_buy'] = limit_buy
        df_bidders.at[bidder, 'limit_sell'] = limit_sell
        df_bidders.at[bidder, 'target_price_param'] = target_price_param
        df_bidders.at[bidder, 'aggressiveness_buy'] = aggressiveness_buy
        df_bidders.at[bidder, 'aggressiveness_sell'] = aggressiveness_sell
        df_bidders.at[bidder, 'bid_price'] = bid_price
        df_bidders.at[bidder, 'ask_price'] = ask_price
    volatilities.append(volatility)

    return df_bidders, volatilities, equilibrium_price_estimate
