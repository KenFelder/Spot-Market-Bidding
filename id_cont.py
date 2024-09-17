import cvxpy as cp


def bid_intra_trustful(player, df_bidders, t_l, t_int):
    a = df_bidders.at[player, 'true_costs'][0]
    b = df_bidders.at[player, 'true_costs'][1]

    x_da = df_bidders.at[player, 'x_da']
    x_bought = df_bidders.at[player, 'x_bought']
    x_sold = df_bidders.at[player, 'x_sold']

    lambda_hat_int = df_bidders.at[player, 'lambda_hat_int']

    x_cap = df_bidders.at[player, 'x_cap']

    x_prod = cp.Variable(nonneg=True)
    x_sell_int = cp.Variable(nonneg=True)
    x_buy_int = cp.Variable(nonneg=True)
    bid_flag = cp.Variable(boolean=True)

    x_imb = x_sell_int - x_buy_int

    payoff_new_bid = lambda_hat_int * (x_sell_int - x_buy_int)
    cost_prod = 0.5 * a * x_prod ** 2 + b * x_prod
    penalty_imb = t_int / (t_l - t_int + 1e-5) * cp.abs(x_imb)

    objective = cp.Maximize(payoff_new_bid - cost_prod - penalty_imb)

    constraints = [
        x_prod + x_bought == x_da + x_sold + x_imb,
        x_prod <= x_cap,
        x_sell_int <= 1e5 * (1 - bid_flag),
        x_buy_int <= 1e5 * bid_flag,
    ]

    problem = cp.Problem(objective, constraints)

    problem.solve(solver=cp.GUROBI)

    lambda_prod_prime = a * x_prod + b

#    print(f"Player:                                         {player}")
#    print(f"t_int:                                          {t_int}")
#    if bid_flag.value == 1:
#        print(f"New bid (price, volume):                        {lambda_hat_int, x_sell_int.value}")
#    else:
#        print(f"New ask (price, volume):                        {lambda_hat_int, x_buy_int.value}")
#    print(f"Production cost t+1 (marginal costs, volume):   {lambda_prod_prime.value, x_prod.value}")
#    print(f"Imbalance (volume):                             {x_imb.value}")

    if bid_flag.value == 0:
        new_post = (lambda_hat_int, x_sell_int.value, bid_flag.value, t_int)  # (float, float, int)
    else:
        new_post = (lambda_hat_int, x_buy_int.value, bid_flag.value, t_int)  # (float, float, int)

    return x_prod.value, x_imb.value, new_post

def bid_intra_strategic(action, player, df_bidders,t_int):
    bid_flag = 1 if action[1] < 0 else 0
    new_post = (action[0], abs(action[1]), bid_flag, t_int)

    x_prod = min(df_bidders.at[player, 'x_da'] + df_bidders.at[player, 'x_sold'] - df_bidders.at[player, 'x_bought'],
                 df_bidders.at[player, 'x_cap'])

    x_imb = df_bidders.at[player, 'x_bought'] + x_prod - df_bidders.at[player, 'x_da'] - df_bidders.at[player, 'x_sold']

    return x_prod, x_imb, new_post

def match_maker(df_order_book):
    len_bids = len(df_order_book[df_order_book["bid_flag"] == 1])
    len_asks = len(df_order_book[df_order_book["bid_flag"] == 0])

    if len_bids > 0:
        top_bid = df_order_book[df_order_book["bid_flag"] == 1].iloc[0]
#        print(f"Top bid:                         \n{top_bid}")
#    else:
#        print("No bids available")

    if len_asks > 0:
        top_ask = df_order_book[df_order_book["bid_flag"] == 0].iloc[-1]
#        print(f"Top ask:                         \n{top_ask}")
#    else:
#        print("No asks available")

    if len_bids > 0 and len_asks > 0 and (top_bid["price"] >= top_ask["price"]):
#        print("Match found")
        volume = min(top_bid["volume"], top_ask["volume"])
#        print(top_ask.name, top_bid.name)
        df_order_book.loc[top_bid.name, 'volume'] -= volume
        df_order_book.loc[top_ask.name, 'volume'] -= volume

        price = top_bid["price"] if top_bid["timestamp"] < top_ask["timestamp"] else top_ask["price"]

        buyer = top_bid["participant"]
        seller = top_ask["participant"]

        if top_bid["volume"] <= 1e-1:
            df_order_book = df_order_book.drop(top_bid.name)
        if top_ask["volume"] <= 1e-1:
            df_order_book = df_order_book.drop(top_ask.name)

        return price, volume, buyer, seller, df_order_book

#    print(df_order_book)

    return None, None, None, None, df_order_book

def update_books(df_order_book, df_bidders, bidder, new_post, x_prod, x_imb):
    # remove old bids/asks from order book
    df_order_book = df_order_book.copy()
    df_order_book = df_order_book[df_order_book["participant"] != bidder]

    # Unpack new_post tuple
    lambda_hat_int, new_volume, bid_flag, t_int = new_post
    # Add new bid/ask to order book
    df_order_book.loc[len(df_order_book)] = [bid_flag, lambda_hat_int, new_volume, bidder, t_int]

    # sort order book
    df_order_book = df_order_book.sort_values(by="price", ascending=False)
    df_order_book = df_order_book.reset_index(drop=True)

    while True:
        price, volume, buyer, seller, df_order_book = match_maker(df_order_book)
        if price is None:
            break
        else:
#            print(f"Matched price: {price}, volume: {volume}, buyer: {buyer}, seller: {seller}")
            df_bidders.at[buyer, 'x_bought'] += volume
            df_bidders.at[seller, 'x_sold'] += volume
            df_bidders.at[buyer, 'revenue'] -= price * volume
            df_bidders.at[seller, 'revenue'] += price * volume

    # Update bidder's x_prod and x_imb
    df_bidders.at[bidder, 'x_prod'] = x_prod
    df_bidders.at[bidder, 'x_imb'] = x_imb

    # Remove rows where volume reaches 0.1
    df_order_book = df_order_book[df_order_book["volume"] > 0.09]

    df_order_book = df_order_book.sort_values(by="price", ascending=False)
    df_order_book = df_order_book.reset_index(drop=True)

    # Return the updated DataFrames
    return df_order_book, df_bidders

def calc_payoff_int_strategic(idx, bidder, df_bidders, rev_before_match):
    return df_bidders.at[idx, 'revenue'] - rev_before_match - (0.5 * bidder.costs[0] * df_bidders.at[idx, 'x_prod'] ** 2
                                                              + bidder.costs[1] * df_bidders.at[idx, 'x_prod'])
