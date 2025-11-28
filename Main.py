import multiprocessing
from spot_env import SpotEnv
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

def make_env():
    env = SpotEnv(seed=True)
    env = FlattenObservation(env)
    env = Monitor(env)

    eval_env = SpotEnv(seed=False)
    eval_env = FlattenObservation(eval_env)
    eval_env = Monitor(eval_env)

    return env, eval_env

def SAC_train(learning_rate):
    timestamp = np.datetime64('now').astype(str).replace(":", "-")
    log_dir = f"./logs/"
    model_dir = f"./models/{timestamp}/SAC_LR_{learning_rate}"

    env, eval_env = make_env()

    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True)

    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        #log_path=f"{log_dir}/eval/",
        eval_freq=200, # Timesteps
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    callbacks = CallbackList([eval_callback])

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        verbose=1,
        tensorboard_log=log_dir,
        #batch_size=256,
        buffer_size=100000,
    )

    model.learn(
        total_timesteps=20000000000000000000,
        log_interval=2, # Episodes
        progress_bar=True,
        tb_log_name=f'SAC_LR_{learning_rate}',
        callback=callbacks
    )

def game(model):
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    obs = env.reset()

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated = env.step(action)
        if done or truncated:
            obs, info = env.reset()

            break


if __name__ == '__main__':
    learning_rates = [0.01]#0.0001, 0.0003, 0.001, 0.003]
    trainings = []

    for learning_rate in learning_rates:
        training = multiprocessing.Process(target=SAC_train, args=(learning_rate,))
        training.start()
        trainings.append(training)
    for training in trainings:
        training.join()
