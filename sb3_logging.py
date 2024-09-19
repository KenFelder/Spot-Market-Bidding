import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure

def plot_bid_ask_spread(top_ask_prices, top_bid_prices):
    """
    Plot the bid-ask spread.
    """
    t = np.arange(len(top_ask_prices))
    plt.plot(t, top_ask_prices, label='Ask prices')
    plt.plot(t, top_bid_prices, label='Bid prices')
    plt.xlabel('Time')
    plt.ylabel('Price [€/MWh]')
    plt.title('Bid-ask spread')
    plt.legend()
    plt.show()

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional values in TensorBoard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Retrieve the rewards and actions
        rewards = self.locals['rewards']  # Rewards from the environment
        actions = self.locals['actions']  # Actions taken by the agent
        dones = self.locals['dones']  # Done status of the environment

        # TODO: Add evaluation metrics
        # Loop through each environment if using multiple environments
        for env_idx in range(self.training_env.num_envs):
            top_ask_prices = self.training_env.envs[env_idx].unwrapped.top_ask_prices
            top_bid_prices = self.training_env.envs[env_idx].unwrapped.top_bid_prices

            print(f"top_ask_prices: {top_ask_prices}")
            print(f"top_bid_prices: {top_bid_prices}")

            # Access specific data from the unwrapped environment
            df_bidders = self.training_env.envs[env_idx].unwrapped.df_bidders
            system_imbalance = df_bidders['x_imb'].sum()
            player_imbalance = df_bidders['x_imb'].iloc[-1]

            # Log rewards
            reward = rewards[env_idx]
            action = actions[env_idx]

            # Log reward and actions to TensorBoard
            self.logger.record(f'environment_{env_idx}/reward', reward)
            self.logger.record(f'environment_{env_idx}/price', action[0])  # Assuming action[0] is price
            self.logger.record(f'environment_{env_idx}/volume', action[1])  # Assuming action[1] is volume
            self.logger.record(f'environment_{env_idx}/system imbalance', system_imbalance)
            self.logger.record(f'environment_{env_idx}/player imbalance', player_imbalance)

            #if dones[env_idx]:

            t = np.arange(len(top_ask_prices))

            # Create the figure and axis
            fig, ax = plt.subplots()

            # Plot ask and bid prices on the same axis
            ax.plot(t, top_ask_prices, label='Ask prices')
            ax.plot(t, top_bid_prices, label='Bid prices')

            # Add labels and title
            ax.set_xlabel('Time')
            ax.set_ylabel('Price [€/MWh]')
            ax.set_title('Bid-ask spread')

            # Add the legend
            ax.legend()

            # Record the figure
            self.logger.record(f'environment_{env_idx}/bid_ask_spread', Figure(fig, close=True),
                               exclude=("stdout", "log", "json", "csv"))

            # Close the plot
            plt.close(fig)

        # Dump logs to TensorBoard
        self.logger.dump(self.num_timesteps)

        return True  # Continue training
