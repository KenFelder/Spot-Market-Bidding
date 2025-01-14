import plotly.graph_objects as go
import os
import pandas as pd
from config import *


if __name__ == '__main__':
    # List all folders in the directory
    runs = [f for f in os.listdir("csv") if os.path.isdir(os.path.join("csv", f))]

    # Get the last folder in the list (as sorted by name or directory listing order)
    latest_run = sorted(runs)[-1]

    df_ask_agg = pd.read_csv(f"csv/{latest_run}/ask_agg.csv", delimiter=";")
    df_bid_agg = pd.read_csv(f"csv/{latest_run}/bid_agg.csv", delimiter=";")
    df_ask_prices = pd.read_csv(f"csv/{latest_run}/ask_prices.csv", delimiter=";")
    df_bid_prices = pd.read_csv(f"csv/{latest_run}/bid_prices.csv", delimiter=";")
    df_expenses = pd.read_csv(f"csv/{latest_run}/expenses.csv", delimiter=";")
    df_game_data = pd.read_csv(f"csv/{latest_run}/game_data.csv", delimiter=";")
    df_imbalances = pd.read_csv(f"csv/{latest_run}/imbalances.csv", delimiter=";")
    df_payoffs = pd.read_csv(f"csv/{latest_run}/payoffs.csv", delimiter=";")
    df_penalty_imbalances = pd.read_csv(f"csv/{latest_run}/penalty_imbalances.csv", delimiter=";")
    df_prod_costs = pd.read_csv(f"csv/{latest_run}/prod_costs.csv", delimiter=";")
    df_revenues = pd.read_csv(f"csv/{latest_run}/revenues.csv", delimiter=";")

    # Aggressiveness
    fig_agg = go.Figure()
    for i in range(0, n):
        fig_agg.add_trace(go.Scatter(x=df_ask_agg.index, y=df_ask_agg[f'bidder_{i}'], mode='lines', line=dict(dash='dash'), name=f'ask_bidder_{i}'))
        fig_agg.add_trace(go.Scatter(x=df_bid_agg.index, y=df_bid_agg[f'bidder_{i}'], mode='lines', name=f'bid_bidder_{i}'))
    fig_agg.update_layout(title='Aggressiveness')
    fig_agg.show()

    # Payoffs
    fig_payoffs = go.Figure()
    for i in range(0, n-1):
        fig_payoffs.add_trace(go.Scatter(x=df_payoffs.index, y=df_payoffs[f'bidder_{i}'], mode='lines', name=f'bidder_{i}'))
    fig_payoffs.update_layout(title='Payoffs')
    fig_payoffs.show()

    # Order book
    fig_ob_gap = go.Figure()
    fig_ob_gap.add_trace(go.Scatter(x=df_game_data.index, y=df_game_data['top_bid'], mode='lines', name='top_bid'))
    fig_ob_gap.add_trace(go.Scatter(x=df_game_data.index, y=df_game_data['top_ask'], mode='lines', name='top_ask'))
    fig_ob_gap.add_trace(go.Scatter(x=df_game_data.index, y=df_game_data['equilibrium_price_estimate'], mode='lines', name='equilibrium_price'))
    fig_ob_gap.update_layout(title='Order book gap')
    fig_ob_gap.show()

    # Prices
    fig_prices = go.Figure()
    for i in range(0, n-1):
        fig_prices.add_trace(go.Scatter(x=df_ask_prices.index, y=df_ask_prices[f'bidder_{i}'], line=dict(dash='dash'), mode='lines', name=f'ask_bidder_{i}'))
        fig_prices.add_trace(go.Scatter(x=df_bid_prices.index, y=df_bid_prices[f'bidder_{i}'], mode='lines', name=f'bid_bidder_{i}'))
    fig_prices.update_layout(title='Prices')
    fig_prices.show()
