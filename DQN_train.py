# Authors: Brittany Carlsson (22752092), Rhianna Hepburn (23340238)  
# Agent 1: Rule-Based Heuristic 
# Unit: CITS3001 Algorithms, Agents and Artificail Intelligence (Semester Two 2023)
# Assignment: Super Mario Project

#References: 
#dvorjackz. (2020, August 25). MarioRL/train.py at master · dvorjackz/mariorl - github. GitHub. https://github.com/dvorjackz/MarioRL/blob/master/train.py
#StableBaselines2. (2018). DQN¶. DQN - Stable Baselines 2.10.3a0 documentation. https://stable-baselines.readthedocs.io/en/master/modules/dqn.html 

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from stable_baselines import DQN
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy, LnCnnPolicy
from stable_baselines.common.atari_wrappers import FrameStack, WarpFrame, MaxAndSkipEnv, EpisodicLifeEnv
from stable_baselines.common.callbacks import CallbackList, EvalCallback, CheckpointCallback
from callbacks import ProgressBarManager
import tensorflow as tf
import cv2
import os
import argparse

import hyperparams as hp

# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#This function was sourced from https://github.com/dvorjackz/MarioRL/blob/master/train.py, with modifications made to suit our training preferences
def run(run_name, existing_model):

    # Create log dir
    log_dir = "./monitor_logs_level1/"
    os.makedirs(log_dir, exist_ok=True)

    print ("Setting up environment...")
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = EpisodicLifeEnv(env)

    # Preprocessing
    env = WarpFrame(env)
    #Stacks together 4 frames at time
    env = FrameStack(env, n_frames=hp.FRAME_STACK)

    # Evaluate every 8th frame and repeat action
    env = MaxAndSkipEnv(env, skip=hp.FRAME_SKIP)

    # Logs will be saved in log_dir/monitor.csv
    env = Monitor(env, log_dir)

    # Save a checkpoint every 1000 steps in the directory models_level1
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models_level1/',
                                            name_prefix=run_name)

    eval_callback = EvalCallback(env,
                                best_model_save_path='./models_level1/',
                                log_path='./models_level1/',
                                eval_freq=10000,
                                deterministic=True,
                                render=False)

    print("Compiling model...")
#Load existing model to train
    if existing_model:
        try:
            model = DQN.load(existing_model, env, tensorboard_log="./mario_tensorboard_level1/")
        except:
            print(f"{existing_model} does not exist!")
            exit(0)
    else:
        #Sourced from Stable Baslines
        model = DQN(LnCnnPolicy,
                    env,
                    batch_size=hp.BATCH_SIZE,
                    verbose=1, 
                    learning_starts=10000,
                    learning_rate=hp.LEARNING_RATE,
                    exploration_fraction=hp.EXPLORATION_FRACT,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.1,
                    prioritized_replay=True, 
                    prioritized_replay_alpha=hp.P_REPLAY_ALPHA,
                    train_freq=hp.TRAINING_FREQ,
                    target_network_update_freq=hp.TARGET_UPDATE_FREQ,
                    tensorboard_log="./mario_tensorboard_level1/"
                )

    print("Training starting...")
    #Commences training and saves the models at 10,000 timestep increments
    with ProgressBarManager(hp.TIME_STEPS) as progress_callback:
        model.learn(total_timesteps=hp.TIME_STEPS,
                    log_interval=1,
                    callback=[progress_callback, checkpoint_callback, eval_callback],
                    tb_log_name=run_name)

    print("Done! Saving model...")
    model.save("models_level1/{}_final".format(run_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-existing", nargs='?', help="Train existing model")
    args = parser.parse_args()

    run("dqn", args.train_existing)
