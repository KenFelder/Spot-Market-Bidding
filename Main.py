import multiprocessing
from spot_env import SpotEnv
import numpy as np
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import torch


def make_env(seed=True):
    """Create training and evaluation environments"""
    env = SpotEnv(seed=seed, log_frequency=10)  # Log every 10th episode
    env = FlattenObservation(env)
    env = Monitor(env)

    eval_env = SpotEnv(seed=False, log_frequency=1)  # Log all eval episodes
    eval_env = FlattenObservation(eval_env)
    eval_env = Monitor(eval_env)

    return env, eval_env


def SAC_train(learning_rate):
    """Train SAC model with proper GPU support and error handling"""
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Learning Rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    timestamp = np.datetime64('now').astype(str).replace(":", "-")
    log_dir = f"./logs/"
    model_dir = f"./models/{timestamp}/SAC_LR_{learning_rate}"

    try:
        env, eval_env = make_env(seed=False)

        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True)

        eval_env = DummyVecEnv([lambda: eval_env])
        eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            eval_freq=1000,  # Evaluate every 1000 timesteps
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=1
        )

        callbacks = CallbackList([eval_callback])

        print("Creating SAC model...")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            verbose=1,
            tensorboard_log=log_dir,
            buffer_size=100_000,
            device=device,  # Explicitly set device for GPU training
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
        )

        print(f"\nStarting training for {20_000_000:,} timesteps...")
        print(f"Model will be saved to: {model_dir}")
        print(f"TensorBoard logs: {log_dir}")
        print(f"\nTo monitor training, run:")
        print(f"  tensorboard --logdir {log_dir}\n")
        
        model.learn(
            total_timesteps=20_000_000,  # 20 million timesteps (FIXED!)
            log_interval=2,  # Log every 2 episodes
            progress_bar=True,
            tb_log_name=f'SAC_LR_{learning_rate}',
            callback=callbacks
        )

        print(f"\n{'='*60}")
        print(f"Training completed successfully!")
        print(f"Best model saved to: {model_dir}")
        print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Cleaning up...")
        raise
    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise


def evaluate_model(model_path, n_episodes=10):
    """Evaluate a trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env, _ = make_env(seed=False)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = SAC.load(model_path, env=env, device=device)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}/{n_episodes}: Reward={episode_reward:.2f}, Length={episode_length}")
    
    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    
    return episode_rewards, episode_lengths


if __name__ == '__main__':
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    
    # Training configuration
    learning_rates = [0.0003]  # Can add more: [0.0001, 0.0003, 0.001]
    
    # Sequential training (recommended for debugging)
    # For parallel training, uncomment the section below
    print(f"\n{'='*60}")
    print(f"Starting Sequential Training")
    print(f"Learning Rates: {learning_rates}")
    print(f"{'='*60}\n")
    
    for lr in learning_rates:
        try:
            SAC_train(lr)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            break
        except Exception as e:
            print(f"\nTraining failed for learning rate {lr}: {e}")
            continue
    
    # Parallel training (uncomment to use)
    """
    print(f"\n{'='*60}")
    print(f"Starting Parallel Training")
    print(f"Learning Rates: {learning_rates}")
    print(f"Number of processes: {len(learning_rates)}")
    print(f"{'='*60}\n")
    
    processes = []
    try:
        for lr in learning_rates:
            p = multiprocessing.Process(target=SAC_train, args=(lr,))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
            if p.exitcode != 0:
                print(f"Warning: Process exited with code {p.exitcode}")
                
    except KeyboardInterrupt:
        print("\n\nInterrupted! Cleaning up processes...")
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
    finally:
        for p in processes:
            if p.is_alive():
                p.kill()
    """
    
    print("\n" + "="*60)
    print("All training complete!")
    print("="*60 + "\n")
