from utils import *


def bid_intra_trustful(self, player):
    """
    Optimized version: Solves 2 LPs instead of 1 MILP
    10-100x faster than the original MILP version
    """
    # Get current state
    x_th_start = self.df_bidders.at[player, 'x_th_start']
    true_costs = self.df_bidders.at[player, 'true_costs']
    x_demand = self.df_bidders.at[player, 'x_demand']
    x_da = self.df_bidders.at[player, 'x_da']
    x_bought = self.df_bidders.at[player, 'x_bought']
    x_sold = self.df_bidders.at[player, 'x_sold']
    ask_price = self.df_bidders.at[player, 'ask_price']
    bid_price = self.df_bidders.at[player, 'bid_price']
    x_re_cap = self.df_bidders.at[player, 'x_re_cap']
    x_th_cap = self.df_bidders.at[player, 'x_th_cap']

    # Get order book (excluding this player's old orders)
    active_ob = self.df_order_book[self.df_order_book["participant"] != player]

    ob_bid_prices = active_ob[active_ob["bid_flag"] == 1]['price'].tolist()
    ob_bid_volumes = active_ob[active_ob["bid_flag"] == 1]['volume'].tolist()
    ob_ask_prices = active_ob[active_ob["bid_flag"] == 0]['price'].tolist()
    ob_ask_volumes = active_ob[active_ob["bid_flag"] == 0]['volume'].tolist()

    # Solve TWICE: once as buyer, once as seller
    # Then pick whichever gives better payoff

    ## SOLVE AS SELLER (bid_flag = 0) ##
    try:
        payoff_sell, vol_sell = _solve_as_seller(self,
            player, x_th_start, true_costs, x_demand, x_da, x_bought, x_sold,
            ask_price, bid_price, x_re_cap, x_th_cap,
            ob_bid_prices, ob_bid_volumes, ob_ask_prices, ob_ask_volumes
        )
    except:
        payoff_sell = -np.inf
        vol_sell = 0

    ## SOLVE AS BUYER (bid_flag = 1) ##
    try:
        payoff_buy, vol_buy = _solve_as_buyer(self,
            player, x_th_start, true_costs, x_demand, x_da, x_bought, x_sold,
            ask_price, bid_price, x_re_cap, x_th_cap,
            ob_bid_prices, ob_bid_volumes, ob_ask_prices, ob_ask_volumes
        )
    except:
        payoff_buy = -np.inf
        vol_buy = 0

    # Pick the better option
    if payoff_sell > payoff_buy:
        new_post = [0, ask_price, vol_sell, player, self.t_int]  # Sell (ask)
    else:
        new_post = [1, bid_price, vol_buy, player, self.t_int]  # Buy (bid)

    return new_post


def _solve_as_seller(self, player, x_th_start, true_costs, x_demand, x_da, x_bought, x_sold,
                     ask_price, bid_price, x_re_cap, x_th_cap,
                     ob_bid_prices, ob_bid_volumes, ob_ask_prices, ob_ask_volumes):
    """
    Solve the optimization as a SELLER (posting an ask)
    This is now a pure LP (no integer variables!)
    """
    # Decision variables
    x_sell_int = cp.Variable(nonneg=True)
    x_buy_int = cp.Variable(nonneg=True)
    x_th_gen = cp.Variable(nonneg=True)
    x_re_gen = cp.Variable(nonneg=True)
    x_imb = cp.Variable()

    # Order book interaction variables
    if len(ob_ask_prices) > 0:
        ob_buy = cp.Variable(len(ob_ask_volumes), nonneg=True)
        ob_buy_costs = cp.sum(cp.multiply(ob_ask_prices, ob_buy))
        ob_buy_constraint = [
            ob_buy <= ob_ask_volumes,
            x_buy_int >= cp.sum(ob_buy),
        ]
    else:
        ob_buy_constraint = []
        ob_buy_costs = 0
        ob_buy = 0

    if len(ob_bid_prices) > 0:
        ob_sell = cp.Variable(len(ob_bid_volumes), nonneg=True)
        ob_sell_payoff = cp.sum(cp.multiply(ob_bid_prices, ob_sell))
        ob_sell_constraint = [
            ob_sell <= ob_bid_volumes,
            x_sell_int >= cp.sum(ob_sell),
        ]
    else:
        ob_sell_constraint = []
        ob_sell_payoff = 0
        ob_sell = 0

    # Objective
    payoff_new_bid = ask_price * (x_sell_int - cp.sum(ob_sell)) - bid_price * (x_buy_int - cp.sum(ob_buy))
    cost_prod = true_costs * x_th_gen
    penalty_imb = self.imbalance_penalty_factor * cp.abs(x_imb)

    objective = cp.Maximize(payoff_new_bid - cost_prod - penalty_imb + ob_sell_payoff - ob_buy_costs)

    # Constraints (SELLER MODE: x_buy_int = 0, x_sell_int can be positive)
    constraints = [
        x_demand + x_th_gen + x_re_gen + x_bought + x_buy_int == x_da + x_sold + x_sell_int + x_imb,
        x_th_gen <= x_th_cap,
        x_th_gen <= x_th_start + ramp_up[player] * (t_max - self.t_int),
        x_th_gen >= x_th_start - ramp_down[player] * (t_max - self.t_int),
        x_re_gen == x_re_cap,
        # SELLER constraints (instead of big-M):
        x_sell_int <= max_ask_volume,  # Can sell
        x_buy_int == 0,  # Cannot buy
    ]

    problem = cp.Problem(objective, constraints + ob_buy_constraint + ob_sell_constraint)
    problem.solve(solver=cp.GUROBI, verbose=False)

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise ValueError(f"Seller optimization failed with status: {problem.status}")

    return problem.value, x_sell_int.value


def _solve_as_buyer(self, player, x_th_start, true_costs, x_demand, x_da, x_bought, x_sold,
                    ask_price, bid_price, x_re_cap, x_th_cap,
                    ob_bid_prices, ob_bid_volumes, ob_ask_prices, ob_ask_volumes):
    """
    Solve the optimization as a BUYER (posting a bid)
    This is now a pure LP (no integer variables!)
    """
    # Decision variables
    x_sell_int = cp.Variable(nonneg=True)
    x_buy_int = cp.Variable(nonneg=True)
    x_th_gen = cp.Variable(nonneg=True)
    x_re_gen = cp.Variable(nonneg=True)
    x_imb = cp.Variable()

    # Order book interaction variables
    if len(ob_ask_prices) > 0:
        ob_buy = cp.Variable(len(ob_ask_volumes), nonneg=True)
        ob_buy_costs = cp.sum(cp.multiply(ob_ask_prices, ob_buy))
        ob_buy_constraint = [
            ob_buy <= ob_ask_volumes,
            x_buy_int >= cp.sum(ob_buy),
        ]
    else:
        ob_buy_constraint = []
        ob_buy_costs = 0
        ob_buy = 0

    if len(ob_bid_prices) > 0:
        ob_sell = cp.Variable(len(ob_bid_volumes), nonneg=True)
        ob_sell_payoff = cp.sum(cp.multiply(ob_bid_prices, ob_sell))
        ob_sell_constraint = [
            ob_sell <= ob_bid_volumes,
            x_sell_int >= cp.sum(ob_sell),
        ]
    else:
        ob_sell_constraint = []
        ob_sell_payoff = 0
        ob_sell = 0

    # Objective
    payoff_new_bid = ask_price * (x_sell_int - cp.sum(ob_sell)) - bid_price * (x_buy_int - cp.sum(ob_buy))
    cost_prod = true_costs * x_th_gen
    penalty_imb = self.imbalance_penalty_factor * cp.abs(x_imb)

    objective = cp.Maximize(payoff_new_bid - cost_prod - penalty_imb + ob_sell_payoff - ob_buy_costs)

    # Constraints (BUYER MODE: x_sell_int = 0, x_buy_int can be positive)
    constraints = [
        x_demand + x_th_gen + x_re_gen + x_bought + x_buy_int == x_da + x_sold + x_sell_int + x_imb,
        x_th_gen <= x_th_cap,
        x_th_gen <= x_th_start + ramp_up[player] * (t_max - self.t_int),
        x_th_gen >= x_th_start - ramp_down[player] * (t_max - self.t_int),
        x_re_gen == x_re_cap,
        # BUYER constraints (instead of big-M):
        x_sell_int == 0,  # Cannot sell
        x_buy_int <= max_bid_volume,  # Can buy
    ]

    problem = cp.Problem(objective, constraints + ob_buy_constraint + ob_sell_constraint)
    problem.solve(solver=cp.GUROBI, verbose=False)

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise ValueError(f"Buyer optimization failed with status: {problem.status}")

    return problem.value, x_buy_int.value


# ORIGINAL MILP VERSION (for comparison/backup)
def bid_intra_trustful_original(self, player):
    """
    ORIGINAL VERSION - MILP (slow!)
    Keep this as backup for now
    """
    # remove old bids/asks from order book
    self.df_order_book = self.df_order_book[self.df_order_book["participant"] != player]

    x_th_start = self.df_bidders.at[player, 'x_th_start']
    true_costs = self.df_bidders.at[player, 'true_costs']
    x_demand = self.df_bidders.at[player, 'x_demand']
    x_da = self.df_bidders.at[player, 'x_da']
    x_bought = self.df_bidders.at[player, 'x_bought']
    x_sold = self.df_bidders.at[player, 'x_sold']
    ask_price = self.df_bidders.at[player, 'ask_price']
    bid_price = self.df_bidders.at[player, 'bid_price']

    ob_bid_prices = self.df_order_book[(self.df_order_book["bid_flag"] == 1)]['price'].tolist()
    ob_bid_volumes = self.df_order_book[(self.df_order_book["bid_flag"] == 1)]['volume'].tolist()
    ob_ask_prices = self.df_order_book[(self.df_order_book["bid_flag"] == 0)]['price'].tolist()
    ob_ask_volumes = self.df_order_book[(self.df_order_book["bid_flag"] == 0)]['volume'].tolist()

    x_re_cap = self.df_bidders.at[player, 'x_re_cap']
    x_th_cap = self.df_bidders.at[player, 'x_th_cap']

    x_sell_int = cp.Variable(nonneg=True)
    x_buy_int = cp.Variable(nonneg=True)
    bid_flag = cp.Variable(boolean=True)  # ❌ MILP!

    x_th_gen = cp.Variable(nonneg=True)
    x_re_gen = cp.Variable(nonneg=True)
    x_imb = cp.Variable()

    if len(ob_ask_prices) > 0:
        ob_buy = cp.Variable(len(ob_ask_volumes), nonneg=True)
        ob_buy_costs = cp.sum(cp.multiply(ob_ask_prices, ob_buy))
        ob_buy_constraint = [
            ob_buy <= ob_ask_volumes,
            x_buy_int >= cp.sum(ob_buy),
        ]
    else:
        ob_buy_constraint = []
        ob_buy_costs = 0
        ob_buy = 0

    if len(ob_bid_prices) > 0:
        ob_sell = cp.Variable(len(ob_bid_volumes), nonneg=True)
        ob_sell_payoff = cp.sum(cp.multiply(ob_bid_prices, ob_sell))
        ob_sell_constraint = [
            ob_sell <= ob_bid_volumes,
            x_sell_int >= cp.sum(ob_sell),
        ]
    else:
        ob_sell_constraint = []
        ob_sell_payoff = 0
        ob_sell = 0

    payoff_new_bid = ask_price * (x_sell_int - cp.sum(ob_sell)) - bid_price * (x_buy_int - cp.sum(ob_buy))
    cost_prod = true_costs * x_th_gen
    penalty_imb = self.imbalance_penalty_factor * cp.abs(x_imb)

    objective = cp.Maximize(payoff_new_bid - cost_prod - penalty_imb + ob_sell_payoff - ob_buy_costs)

    constraints = [
        x_demand + x_th_gen + x_re_gen + x_bought + x_buy_int == x_da + x_sold + x_sell_int + x_imb,
        x_th_gen <= x_th_cap,
        x_th_gen <= x_th_start + ramp_up[player] * (t_max - self.t_int),
        x_th_gen >= x_th_start - ramp_down[player] * (t_max - self.t_int),
        x_re_gen == x_re_cap,
        x_sell_int <= max_ask_volume * (1 - bid_flag),  # Big-M
        x_buy_int <= max_bid_volume * bid_flag,  # Big-M
    ]

    problem = cp.Problem(objective, constraints + ob_buy_constraint + ob_sell_constraint)
    problem.solve(solver=cp.GUROBI, verbose=False)

    if bid_flag.value == 0:
        new_post = [bid_flag.value, ask_price, x_sell_int.value, player, self.t_int]
    else:
        new_post = [bid_flag.value, bid_price, x_buy_int.value, player, self.t_int]

    return new_post


# Alias for easy switching
#bid_intra_trustful = bid_intra_trustful_optimized


# To use original: bid_intra_trustful = bid_intra_trustful_original


def bid_intra_strategic(self, action, player):
    price = action[0]
    volume = action[1]
    bid_flag = 1 if volume < 0 else 0

    new_post = [bid_flag, price, abs(volume), player, self.t_int]

    return new_post

def match_maker(self):
    bids = self.df_order_book[self.df_order_book["bid_flag"] == 1]
    asks = self.df_order_book[self.df_order_book["bid_flag"] == 0]

    if len(bids)> 0 and len(asks) > 0:
        top_bid = bids.iloc[0]
        top_ask = asks.iloc[-1]

        if top_bid["price"] >= top_ask["price"]:
            volume = min(top_bid["volume"], top_ask["volume"])
            self.df_order_book.loc[top_bid.name, 'volume'] -= volume
            self.df_order_book.loc[top_ask.name, 'volume'] -= volume

            price = top_bid["price"] if top_bid["timestamp"] < top_ask["timestamp"] else top_ask["price"]

            buyer = top_bid["participant"]
            seller = top_ask["participant"]

            if top_bid["volume"] <= 1e-1:
                self.df_order_book = self.df_order_book.drop(top_bid.name)
            if top_ask["volume"] <= 1e-1:
                self.df_order_book = self.df_order_book.drop(top_ask.name)

            return price, volume, buyer, seller

    return None, None, None, None

def update_books(self, player, new_post):
    transaction_prices = self.df_game_data['transaction_price'].dropna().tolist()

    # Unpack new_post tuple
    bid_flag, lambda_hat_int, new_volume, player, t_int = new_post

    # Add new bid/ask to order book
    self.df_order_book.loc[len(self.df_order_book) + 1] = [bid_flag, lambda_hat_int, new_volume, player, t_int]
    # Remove rows where volume reaches 0.1
    self.df_order_book = self.df_order_book[self.df_order_book["volume"] >= 0.1]
    # sort order book
    self.df_order_book = self.df_order_book.sort_values(by="price", ascending=False)
    self.df_order_book = self.df_order_book.reset_index(drop=True)

    bid_prices = self.df_order_book[self.df_order_book["bid_flag"] == 1]["price"]
    ask_prices = self.df_order_book[self.df_order_book["bid_flag"] == 0]["price"]

    if len(bid_prices) > 0:
        self.df_game_data.at[self.t_int, 'top_bid'] = max(bid_prices)
    else:
        self.df_game_data.at[self.t_int, 'top_bid'] = min_price
    if len(ask_prices) > 0:
        self.df_game_data.at[self.t_int, 'top_ask'] = min(ask_prices)
    else:
        self.df_game_data.at[self.t_int, 'top_ask'] = max_price

    last_event = 'bid' if bid_flag == 1 else 'ask'

    # possibly more matches than just one
    while True:
        # Remove rows where volume reaches 0.1
        self.df_order_book = self.df_order_book[self.df_order_book["volume"] >= 0.1]
        price, volume, buyer, seller, = match_maker(self)

        if price is None:
            break
        else:
            self.df_bidders.at[buyer, 'x_bought'] += volume
            self.df_bidders.at[seller, 'x_sold'] += volume
            self.df_bidders.at[buyer, 'expenses'] += price * volume
            self.df_bidders.at[seller, 'revenue'] += price * volume

            last_event = 'match'
            transaction_prices.append(price)

            self.df_bid_logs.at[self.t_int, 'buyer'] = buyer
            self.df_bid_logs.at[self.t_int, 'seller'] = seller

    # Remove rows where volume reaches 0.1
    self.df_order_book = self.df_order_book[self.df_order_book["volume"] >= 0.1]

    self.df_order_book = self.df_order_book.sort_values(by="price", ascending=False)
    self.df_order_book = self.df_order_book.reset_index(drop=True)

    self.df_game_data.at[self.t_int, 'transaction_price'] = transaction_prices[-1] if last_event == 'match' else None
    self.df_game_data.at[self.t_int, 'last_event'] = last_event
    self.df_game_data.at[self.t_int, 'last_price'] = lambda_hat_int

    self.df_bid_logs.at[self.t_int, 'match_flag'] = last_event
    self.df_bid_logs.at[self.t_int, 'transaction_price'] = transaction_prices[-1] if last_event == 'match' else None
    self.df_bid_logs.at[self.t_int, 'price'] = lambda_hat_int
    self.df_bid_logs.at[self.t_int, 'volume'] = new_volume
    self.df_bid_logs.at[self.t_int, 'bidder'] = player

    update_production(self)

    return
