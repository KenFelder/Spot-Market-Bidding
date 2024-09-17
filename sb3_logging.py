import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

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

        # TODO: Add evaluation metrics
        # Loop through each environment if using multiple environments
        for env_idx in range(self.training_env.num_envs):
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

        # Dump logs to TensorBoard
        self.logger.dump(self.num_timesteps)

        return True  # Continue training