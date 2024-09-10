from spot_env import SpotEnv
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import SAC, TD3, DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env

# TODO: Check if "while True" loop is like in docs
def DDPG_game(env, training_steps):
    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=training_steps, log_interval=10)
    model.save("ddpg_spot_model")
    vec_env = model.get_env()

    del model  # remove to demonstrate saving and loading

    model = DDPG.load("ddpg_spot_model")

    obs = vec_env.reset()
    print('Training done')
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

def PPO_game(env, training_steps, train):
    # Parallel environments
    #vec_env = make_vec_env(env, n_envs=4)
    vec_env = make_vec_env(lambda: SpotEnv(t_max=100, n=5, q=1448.4, cap_mean=700), n_envs=1)

    if train:
        model = PPO("MultiInputPolicy", vec_env, verbose=1)
        model = PPO("MultiInputPolicy", env, verbose=1)
        model.learn(total_timesteps=training_steps)
        model.save("ppo_spot_model")

        del model  # remove to demonstrate saving and loading

    else:
        model = PPO.load("ppo_spot_model")

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        if dones:
            print(f'reward: {rewards}')
            break

# TODO: Check if "while True" loop is like in docs
def TD3_game(env, training_steps):
    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3("MultiInputPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=training_steps, log_interval=10)
    model.save("td3_spot_model")


    del model # remove to demonstrate saving and loading

    model = TD3.load("td3_spot_model")

    obs = env.reset()
    print('Training done')

    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        vec_env.render("human")

def SAC_game(env, training_steps):
    model = SAC("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=training_steps, log_interval=4)
    model.save("sac_spot_model")

    del model  # remove to demonstrate saving and loading

    model = SAC.load("sac_spot_model")

    obs, info = env.reset()
    print('Training done')
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, info = env.reset()

if __name__ == '__main__':
    env = SpotEnv(t_max=100, n=5, q=1448.4, cap_mean=700)
    training_steps = 10
    train = False
    #env = FlattenObservation(env)
    PPO_game(env, training_steps, train)  # TODO: Not learning, maybe missing state? Bad selection of obs? Too large obs spaces? Flatten obs?
    #SAC_game(env, training_steps)  # TODO: Same as PPO
    #TD3_game(env, training_steps)  # TODO: Same as SAC
    #DDPG_game(env, training_steps)  # TODO: Test
