# Authors: Brittany Carlsson (22752092), Rhianna Hepburn (23340238)  
# Evaluation of a PPO agent
# Unit: CITS3001 Algorithms, Agents and Artificail Intelligence (Semester Two 2023)
# Assignment: Super Mario Project
"""
### References ###
StableBaselines3. (2021). PPO. PPO - Stable Baselines3 2.2.0a7 documentation. https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import time
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.monitor import Monitor
#from stable_baselines3 import PPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
import gym
from stable_baselines3 import PPO
import tensorflow as tf
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#This function was sourced from Stabe Baselines, and modified to include the following variables:
#number of steps, valuable actions, x distance, average number of coins and the number of completed episodes
"""
    The function definition from Stable Baselines:
    Runs policy for `n_eval_episodes` episodes and returns average reward.
    :param model: (BaseRLModel) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a `VecEnv`
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: (bool) If True, a list of reward per episode
        will be returned instead of the mean.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when `return_episode_rewards` is True
"""
def evaluate_policy(model, env, n_eval_episodes, deterministic,
                    render, callback=None, reward_threshold=None,
                    return_episode_rewards=False):

    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths = [], []
    numsteps=0
    completed_ep=0
    valuable_actions=0
    avg_coins=0
    x_distance=0
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            time.sleep(0.05)
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)
            if reward >0: #If the resulted in a positive reward (this excludes just standing still)
                valuable_actions +=1 
            flag = _info[0]['flag_get']
            if(flag):
                done=False
                completed_ep += 1          
            numsteps+=1
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        avg_coins +=_info[0]['coins']
        x_distance+= ((_info[0]['x_pos']/3161)*100) #3161 is the x position of mario reaching the flag, and this calculation gets the percentage of level completed 
        if x_distance == 100:
            completed_ep += 1
        print("episode ",_, " is done")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, 'Mean reward below threshold: '\
                                         '{:.2f} < {:.2f}'.format(mean_reward, reward_threshold)
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    mean_steps=numsteps/n_eval_episodes
    mean_valuable=(valuable_actions/numsteps)*100
    coins = avg_coins/n_eval_episodes
    avg_x_distance=x_distance/n_eval_episodes
    print("X distanc %: " ,avg_x_distance)
    print("Mean rewards is ", mean_reward, "STD reward is : ", std_reward, "Mean steps is: ", mean_steps, "Completed episodes: ", completed_ep)
    print("Percentage of valuable actions: ", round(mean_valuable,2), "Average number of coins per episode: ", coins )

#Environment set up
env = Monitor(gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human"))
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = DummyVecEnv([lambda:env])
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
model = PPO.load("./MLP_POLICY_REINFORCED_PPO_AGENT.zip")

obs = env.reset()
#Runs and evaluates the model over one episode 
evaluate_policy(model, env, n_eval_episodes=1, deterministic=False, render=True, return_episode_rewards=False)

print("Done.")