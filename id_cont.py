from utils import *

def bid_intra_trustful(self, player, action):
    # remove old bids/asks from order book
    self.df_order_book = self.df_order_book[self.df_order_book["participant"] != player]

    x_th_start = self.df_bidders.at[player, 'x_th_gen']

    a = self.df_bidders.at[player, 'true_costs']

    x_demand = self.df_bidders.at[player, 'x_demand']
    x_da = self.df_bidders.at[player, 'x_da']
    x_bought = self.df_bidders.at[player, 'x_bought']
    x_sold = self.df_bidders.at[player, 'x_sold']

    ask_price = self.df_bidders.at[player, 'ask_price']
    bid_price = self.df_bidders.at[player, 'bid_price']

    ob_bid_prices = self.df_order_book[(self.df_order_book["bid_flag"] == 1) & (self.df_order_book["price"] >= ask_price)]['price'].tolist()
    ob_bid_volumes = self.df_order_book[(self.df_order_book["bid_flag"] == 1) & (self.df_order_book["price"] >= ask_price)]['volume'].tolist()
    ob_ask_prices = self.df_order_book[(self.df_order_book["bid_flag"] == 0) & (self.df_order_book["price"] <= bid_price)]['price'].tolist()
    ob_ask_volumes = self.df_order_book[(self.df_order_book["bid_flag"] == 0) & (self.df_order_book["price"] <= bid_price)]['volume'].tolist()

    x_re_cap = self.df_bidders.at[player, 'x_re_cap']
    x_th_cap = self.df_bidders.at[player, 'x_th_cap']

    x_sell_int = cp.Variable(nonneg=True)
    x_buy_int = cp.Variable(nonneg=True)
    bid_flag = cp.Variable(boolean=True)

    x_th_gen = cp.Variable(nonneg=True)
    x_re_gen = cp.Variable(nonneg=True)

    #x_imb = x_sell_int - x_buy_int
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
    cost_prod = a * x_th_gen
    penalty_imb = self.imbalance_penalty_factor * cp.abs(x_imb)

    objective = cp.Maximize(payoff_new_bid - cost_prod - penalty_imb + ob_sell_payoff - ob_buy_costs)

    constraints = [
        #x_demand + x_th_gen + x_re_gen + x_bought == x_da + x_sold + x_imb,
        x_demand + x_th_gen + x_re_gen + x_bought + x_buy_int == x_da + x_sold + x_sell_int + x_imb,
        x_th_gen <= x_th_cap,
        cp.abs(x_th_gen - x_th_start) <= (1 - self.t_int / t_max) * x_th_cap,
        # TODO: decide if re must be fed-in; if changed also change in update_production
        x_re_gen == x_re_cap,
        #x_re_gen <= x_re_cap,
        x_sell_int <= max_ask_volume * (1 - bid_flag),
        x_buy_int <= max_bid_volume * bid_flag,
    ]

    problem = cp.Problem(objective, constraints + ob_buy_constraint + ob_sell_constraint)

    problem.solve(solver=cp.GUROBI)

    if bid_flag.value == 0:
        new_post = [bid_flag.value, ask_price, x_sell_int.value, player, self.t_int]
    else:
        new_post = [bid_flag.value, bid_price, x_buy_int.value, player, self.t_int]

    return new_post

def bid_intra_strategic(self, action, player):
    bid_flag = 1 if action[1] < 0 else 0
    new_post = [bid_flag, action[0], abs(action[1]), player, self.t_int]

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

    self.df_game_data.at[self.t_int, 'transaction_price'] = transaction_prices[-1] if last_event == 'match' else None
    self.df_game_data.at[self.t_int, 'last_event'] = last_event
    self.df_game_data.at[self.t_int, 'last_price'] = lambda_hat_int

    self.df_bid_logs.at[self.t_int, 'match_flag'] = last_event
    self.df_bid_logs.at[self.t_int, 'transaction_price'] = new_volume
    self.df_bid_logs.at[self.t_int, 'price'] = lambda_hat_int
    self.df_bid_logs.at[self.t_int, 'volume'] = new_volume
    self.df_bid_logs.at[self.t_int, 'bidder'] = player

    # Remove rows where volume reaches 0.1
    self.df_order_book = self.df_order_book[self.df_order_book["volume"] >= 0.1]

    self.df_order_book = self.df_order_book.sort_values(by="price", ascending=False)
    self.df_order_book = self.df_order_book.reset_index(drop=True)

    update_production(self)

    return
