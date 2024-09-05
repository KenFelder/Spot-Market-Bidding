from spot_env import SpotEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import SAC

def SAC_game(env):
    model = SAC("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("sac_gridworld")

    del model # remove to demonstrate saving and loading

    model = SAC.load("sac_gridworld")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

if __name__ == '__main__':
    env = SpotEnv()
    #env = FlattenObservation(env)
    print(f'Obs type:   {type(env.action_space)}')
    SAC_game(env)
