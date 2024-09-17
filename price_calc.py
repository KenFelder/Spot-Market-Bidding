import numpy as np


def update_limits(limit_buy, limit_sell, imbalance, imbalance_penalty, step_factor=0.1):
    if imbalance > 0:
        limit_sell = (1 - step_factor) * limit_sell + step_factor * np.max(imbalance_penalty, limit_sell)
    elif imbalance < 0:
        limit_buy = (1 - step_factor) * limit_buy + step_factor * np.max(imbalance_penalty, limit_buy)
    return limit_buy, limit_sell


def comp_price_estimate(transaction_prices, k_max=5):
    k = min(k_max, len(transaction_prices))
    lambda_hat = 1/k * np.sum(transaction_prices[-k:])
    return lambda_hat


# TODO: step_factor should be bidder specific
def update_aggressiveness(aggressiveness, target_aggressiveness, step_factor=0.3):
    aggressiveness = aggressiveness + step_factor * (target_aggressiveness - aggressiveness)
    return aggressiveness


# TODO: check these functions (got them from ChatGPT)
def calc_target_aggressiveness_buy(lambda_hat, limit_buy, target_price_buy, target_price_param):
    if target_price_buy > lambda_hat:
        return (1 / target_price_param) * np.log(
            ((target_price_buy - lambda_hat) * (np.exp(target_price_param) - 1)) / (limit_buy - lambda_hat) + 1
        )
    else:
        # Extra-marginal case, aggressiveness is 0 if target_price_buy equals limit_buy
        return 0


# TODO: check these functions (got them from ChatGPT)
def calc_target_aggressiveness_sell(lambda_hat, limit_sell, target_price_sell, target_price_param):
    if target_price_sell < lambda_hat:
        return (1 / target_price_param) * np.log(
            ((1 - (target_price_sell - limit_sell) / (lambda_hat - limit_sell)) * (np.exp(target_price_param) - 1)) + 1
        )
    else:
        # Extra-marginal case, aggressiveness is 0 if target_price_sell equals limit_sell
        return 0


def calc_target_aggressiveness(t_agg_buy, t_agg_sell, last_event, delta_r=0.02, delta_a=0.01):
    if last_event == 'bid':
        t_agg_buy = (1 + delta_r) * bid_aggressiveness + delta_a
    elif last_event == 'ask':
        t_agg_sell = (1 + delta_r) * ask_aggressiveness + delta_a
    else:
        t_agg_buy = (1 + delta_r) * bid_aggressiveness + delta_a
        t_agg_sell = (1 + delta_r) * ask_aggressiveness + delta_a
    return t_agg_buy, t_agg_sell


# TODO: step_factor should be bidder specific
def update_target_price_param(target_price_param, target_price_param_scaled, step_factor=0.3):
    target_price_param = target_price_param + step_factor * (target_price_param_scaled - target_price_param)
    return target_price_param


def scale_target_price_param(volatility_scaled, target_price_param_max=2, target_price_param_min=-8, gamma=2):
    return (target_price_param_max - target_price_param_min) * (1 - volatility_scaled * np.exp(gamma *
                                                                                               (volatility_scaled - 1)))


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
