import numpy as np


def update_limits(limit_buy, limit_sell, lambda_max, imbalance, imbalance_penalty, step_factor=0.1):
    if imbalance > 0:
        limit_sell = (1 - step_factor) * limit_sell + step_factor * min(max(imbalance_penalty, limit_sell), lambda_max)
    elif imbalance < 0:
        limit_buy = (1 - step_factor) * limit_buy + step_factor * max(imbalance_penalty, limit_buy)
    return limit_buy, limit_sell


def comp_price_estimate(transaction_prices, k_max=5):
    k = min(k_max, len(transaction_prices))
    lambda_hat = 1/k * np.sum(transaction_prices[-k:])
    return lambda_hat


# TODO: step_factor should be bidder specific
def update_aggressiveness(aggressiveness, target_aggressiveness, step_factor=0.3):
    aggressiveness = aggressiveness + step_factor * (target_aggressiveness - aggressiveness)
    return aggressiveness

def calc_target_aggressiveness_buy(lambda_hat, limit_buy, last_target_price_buy, target_price_param):
    if lambda_hat < last_target_price_buy < limit_buy:
        target_bid_aggressiveness = np.log((last_target_price_buy - lambda_hat) * (np.exp(target_price_param) - 1) /
                                           (limit_buy - lambda_hat) + 1) / target_price_param
    elif 0 < last_target_price_buy < lambda_hat:
        target_bid_aggressiveness = -np.log((1 - last_target_price_buy / lambda_hat) * (np.exp(target_price_param) - 1) +
                                            1) / target_price_param
    elif 0 < last_target_price_buy < limit_buy:
        target_bid_aggressiveness = -np.log((1 - last_target_price_buy / limit_buy) * (np.exp(target_price_param) - 1) +
                                            1) / target_price_param
    elif limit_buy < last_target_price_buy:
        target_bid_aggressiveness = 0
    return target_bid_aggressiveness


def calc_target_aggressiveness_sell(lambda_hat, limit_sell, last_target_price_sell, target_price_param, max_price):
    if lambda_hat < last_target_price_sell < max_price:
        target_ask_aggressiveness = -np.log((last_target_price_sell - lambda_hat) * (np.exp(target_price_param) - 1) /
                                            (max_price - lambda_hat) + 1) / target_price_param
    elif limit_sell < last_target_price_sell < lambda_hat:
        target_ask_aggressiveness = np.log((1 - (last_target_price_sell - limit_sell) / (lambda_hat - limit_sell) *
                                            (np.exp(target_price_param) - 1) + 1)) / target_price_param
    elif limit_sell < last_target_price_sell < max_price:
        target_ask_aggressiveness = -np.log((last_target_price_sell - limit_sell) / (max_price - 1) *
                                            (np.exp(target_price_param) - 1) + 1) / target_price_param
    elif last_target_price_sell < limit_sell:
        target_ask_aggressiveness = 0
    return target_ask_aggressiveness


# TODO: check when delta_r and delta_a should be negative
def calc_target_aggressiveness(t_agg_buy, t_agg_sell, last_event, target_bid_aggressiveness, target_ask_aggressiveness,
                               delta_r=0.02, delta_a=0.01):
    if last_event == 'bid':
        t_agg_buy = (1 + delta_r) * target_bid_aggressiveness + delta_a
    elif last_event == 'ask':
        t_agg_sell = (1 + delta_r) * target_ask_aggressiveness + delta_a
    else:
        t_agg_buy = (1 + delta_r) * target_bid_aggressiveness + delta_a
        t_agg_sell = (1 + delta_r) * target_ask_aggressiveness + delta_a
    return t_agg_buy, t_agg_sell


# TODO: step_factor should be bidder specific
def update_target_price_param(target_price_param, target_price_param_scaled, step_factor=0.3):
    target_price_param = target_price_param + step_factor * (target_price_param_scaled - target_price_param)
    return target_price_param


def scale_target_price_param(volatility_scaled, target_price_param_max=2, target_price_param_min=-8, gamma=2):
    target_price_param_scaled = (target_price_param_max - target_price_param_min) * (1 - volatility_scaled *
                                                                                     np.exp(gamma * (volatility_scaled -
                                                                                                     1)))
    return target_price_param_scaled

def scale_volatility(volatility, volatility_max, volatility_min):
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


if __name__ == "__main__":
    t =200
    t_l = 200
    lambda_max = 30
    imbalance = 40
    imbalance_penalty = t / (t_l - t + 1) * np.abs(imbalance)
    transaction_prices = [19, 17, 10, 9]
    limit_buy_list = [18]
    limit_sell_list = [18]

    aggressiveness_buy_list = [0]
    aggressiveness_sell_list = [0]
    t_agg_buy = 0
    t_agg_sell = 0
    last_event = 'match'
    last_target_price_buy = 9
    last_target_price_sell = 15
    target_price_param_list = [-2]
    volatility_list = [1]

    for i in range(10):
        lambda_hat = comp_price_estimate(transaction_prices)
        limit_buy, limit_sell = update_limits(limit_buy_list[-1], limit_sell_list[-1], lambda_max, imbalance, imbalance_penalty)
        limit_buy_list.append(limit_buy)
        limit_sell_list.append(limit_sell)
        volatility = calc_volatility(transaction_prices, lambda_hat)
        volatility_list.append(volatility)

        if last_event == 'match':
            volatility_scaled = scale_volatility(volatility_list[-1], max(volatility_list), min(volatility_list))
            target_price_param_scaled = scale_target_price_param(volatility_scaled)
            target_price_param = update_target_price_param(target_price_param_list[-1], target_price_param_scaled)
            target_price_param_list.append(target_price_param)
        target_bid_aggressiveness = calc_target_aggressiveness_buy(lambda_hat, limit_buy, last_target_price_buy, target_price_param_list[-1])
        target_ask_aggressiveness = calc_target_aggressiveness_sell(lambda_hat, limit_sell, last_target_price_sell, target_price_param_list[-1], lambda_max)
        t_agg_buy, t_agg_sell = calc_target_aggressiveness(t_agg_buy, t_agg_sell, last_event, target_bid_aggressiveness, target_ask_aggressiveness)
        aggressiveness_buy = update_aggressiveness(aggressiveness_buy_list[-1], t_agg_buy)
        aggressiveness_buy_list.append(aggressiveness_buy)
        aggressiveness_sell = update_aggressiveness(aggressiveness_sell_list[-1], t_agg_sell)
        aggressiveness_sell_list.append(aggressiveness_sell)



        target_price_buy, target_price_sell = calc_target_price(lambda_hat, lambda_max, limit_buy, limit_sell, aggressiveness_buy, aggressiveness_sell, target_price_param)
        print("Iteration: ", i)
        print(f"target_price_buy: {target_price_buy}")
        print(f"target_price_sell: {target_price_sell}")
        print(f"lambda_hat: {lambda_hat}")
        print(f"limit_buy: {limit_buy}")
        print(f"limit_sell: {limit_sell}")
        print(f"imbalance_penalty: {imbalance_penalty}")
        print(f"aggressiveness_buy: {aggressiveness_buy}")
        print(f"aggressiveness_sell: {aggressiveness_sell}")
        print(f"target_price_param: {target_price_param}")
        print(f"volatility: {volatility}\n")
