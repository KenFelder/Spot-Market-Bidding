from stable_baselines3.common.callbacks import BaseCallback
import torch

class QValueCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(QValueCallback, self).__init__(verbose)
        self.q_values = []

    def _on_step(self) -> bool:
        # Get the last observation from the model
        obs = self.model._last_obs  # Using _last_obs as it's more reliable here

        # Convert to tensor and move to the correct device
        obs_tensor = torch.as_tensor(obs, device=self.model.device)

        with torch.no_grad():
            # Get the action from the policy (actor)
            action = self.model.actor(obs_tensor, deterministic=True)

            # Compute Q-values from critic networks
            q_value1, q_value2 = self.model.critic(obs_tensor, action)  # Unpacking tuple

        # Store mean Q-values
        q1_mean = q_value1.mean().item()
        q2_mean = q_value2.mean().item()

        self.logger.record("train/q1_mean", q1_mean)
        self.logger.record("train/q2_mean", q2_mean)

        return True
