# Authors: Brittany Carlsson (22752092), Rhianna Hepburn (23340238) 
# Agent 1: Reinforced Learning with Proximal Policy Optimization PPO
# Unit: CITS3001 Algorithms, Agents and Artificail Intelligence (Semester Two 2023)
# Assignment: Super Mario Project
"""
### References ###
#StableBaselines2. (2018). DQN¶. DQN -Evaluation Helper. https://stable-baselines.readthedocs.io/en/master/common/evaluation.html
"""

import numpy as np
import time
from stable_baselines.common.vec_env import VecEnv
from nes_py.wrappers import JoypadSpace
from stable_baselines.deepq.policies import MlpPolicy,CnnPolicy
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.atari_wrappers import FrameStack, WarpFrame, MaxAndSkipEnv, EpisodicLifeEnv
import tensorflow as tf
# Suppress warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import hyperparams as hp

#This function is sourced from Stable Baselines, with the function description, and our modifications, seen below:
"""
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
#This function has been modified to include the following variables for evaluation: number of steps, valuable actions, x distance, average number of coins
#and the number of completed episodes
def evaluate_policy(model, env, n_eval_episodes=10, deterministic=True,
                    render=False, callback=None, reward_threshold=None,
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
            if reward >0: #If the action resulted in a positive reward, then it is a valuable action
                valuable_actions +=1 
            flag = _info['flag_get']
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
        avg_coins +=_info['coins']
        x_distance+= ((_info['x_pos']/3161)*100) #3161 is the x position of mario reaching the flag, and this calculation gets the percentage of level completed 
        if x_distance == 100:
            completed_ep += 1


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


#Environment Setup
env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0') #This can be changed to evaluate the agent on single stages or Mario, or random stages
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = EpisodicLifeEnv(env)
env = WarpFrame(env)
env = FrameStack(env, n_frames=hp.FRAME_STACK)
env = MaxAndSkipEnv(env, skip=hp.FRAME_SKIP)
#This can be changed to the random stages model, currently it loads the agent trained on World 1 Stage 1
model = DQN.load("dqn_500000_steps.zip")

obs = env.reset()
#Run and evaluates the model over 1 episodes
evaluate_policy(model, env, n_eval_episodes=1, deterministic=False, render=True)

print("Done.")