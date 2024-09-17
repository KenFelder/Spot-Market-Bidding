from spot_env import SpotEnv
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import SAC, TD3, DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
from sb3_logging import TensorboardCallback


def TD3_game(env, callback):
    timestamp = np.datetime64('now').astype(str).replace(":", "-")
    log_dir = f"./logs/td3_spot_tensorboard/{timestamp}/"
    model_dir = f"./models/td3_spot_model/{timestamp}/"

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3("MultiInputPolicy", env, action_noise=action_noise, verbose=1,
                tensorboard_log=log_dir)
    model.learn(total_timesteps=10000, log_interval=10, progress_bar=True, callback=callback)
    model.save(model_dir)
    vec_env = model.get_env()

    #del model  # remove to demonstrate saving and loading

    #model = TD3.load(f"./model_dir/")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)

        if dones.any():  # If any of the environments are done
            print(f'Reward: {rewards}')
            obs = vec_env.reset()  # Reset environment after it is done
            break  # You can break the loop or continue based on your use case


def DDPG_game(env, callback):
    timestamp = np.datetime64('now').astype(str).replace(":", "-")
    log_dir = f"./logs/ddpg_spot_tensorboard/{timestamp}/"
    model_dir = f"./models/ddpg_spot_model/{timestamp}/"

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=10000, log_interval=10, progress_bar=True, callback=callback)
    model.save(model_dir)
    vec_env = model.get_env()

    #del model  # remove to demonstrate saving and loading

    #model = DDPG.load(model_dir)

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)

        if dones.any():  # If any of the environments are done
            print(f'Reward: {rewards}')
            obs = vec_env.reset()  # Reset environment after it is done
            break  # You can break the loop or continue based on your use case


def PPO_game(env, callback):
    timestamp = np.datetime64('now').astype(str).replace(":", "-")
    log_dir = f"./logs/ppo_spot_tensorboard/{timestamp}/"
    model_dir = f"./models/ppo_spot_model/{timestamp}/"

    # Parallel environments
    vec_env = make_vec_env(lambda: SpotEnv(t_max=200, n=5, q=1448.4, cap_mean=700), n_envs=3)

    # Train the PPO model
    print(timestamp)
    model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=100, log_interval=10, progress_bar=True, callback=TensorboardCallback())
    model.save(model_dir)
    #del model  # Remove model to demonstrate loading

    # Load the trained model for inference

    #model = PPO.load(f"{model_dir}/")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)

        # 'dones' is a list when using vec_env, so we should check for any True
        if dones.all():  # If any of the environments are done
            print(f'Reward: {rewards}')
            obs = vec_env.reset()  # Reset environment after it is done
            break  # You can break the loop or continue based on your use case


def SAC_game(env, callback):
    timestamp = np.datetime64('now').astype(str).replace(":", "-")
    log_dir = f"./logs/sac_spot_tensorboard/{timestamp}/"
    model_dir = f"./models/sac_spot_model/{timestamp}/"

    model = SAC("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=10000, log_interval=4, progress_bar=True, callback=callback)
    model.save(model_dir)

    #del model  # remove to demonstrate saving and loading

    #model = SAC.load(f"{model_dir}/")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, info = env.reset()


if __name__ == '__main__':
    env = SpotEnv(t_max=100, n=5, q=1448.4, cap_mean=700)
    callback = TensorboardCallback()
    #env = FlattenObservation(env)
    PPO_game(env, callback)  # TODO: Not learning, maybe missing state? Bad selection of obs?
    #                                Too large obs spaces? Flatten obs? Imbalance penalty too harsh?
    #SAC_game(env, callback)  # TODO: Same as PPO
    #TD3_game(env, callback)  # TODO: Same as SAC
    #DDPG_game(env, callback)  # TODO: Test
