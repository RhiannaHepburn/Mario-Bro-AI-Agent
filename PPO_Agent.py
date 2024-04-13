# Authors: Brittany Carlsson (22752092), Rhianna Hepburn (23340238)  
# Agent 1: Reinforced Learning with Proximal Policy Optimization PPO
# Unit: CITS3001 Algorithms, Agents and Artificail Intelligence (Semester Two 2023)
# Assignment: Super Mario Project
"""
### References ###
# Renotte, N. (2021, December 22). Mario Tutorial.ipynb. GitHub . https://github.com/nicknochnack/MarioRL/blob/main/Mario%20Tutorial.ipynb #
# YouTube. (2021). Build an Mario AI Model with Python | Gaming Reinforcement Learning. YouTube. Retrieved October 15, 2023, from https://www.youtube.com/watch?v=2eeYqJ0uBKE&amp;t=7s.  #
# Stablebaselines3. (2021). Learning Rate Schedule. Examples - Stable Baselines3 2.2.0a7 documentation. https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#learning-rate-schedule 
#StableBaselines3. (2021). PPO. PPO - Stable Baselines3 2.2.0a7 documentation. https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
"""

# List of Imports 
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from matplotlib import pyplot as plt
import os
import gym_super_mario_bros
import gym
import numpy as np 
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.callbacks import BaseCallback

# This class was sourced from (Renotte, Mario Tutorial.ipynb 2021)'s GitHub from their Youtube tutorial. (Renotte, 2021).
# This was used in order to log every 10,000 timesteps while our PPO agent was training
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
    
# Linear learning rate schedule
# Referenced from (Stablebaselines3, Learning Rate Schedule 2021)
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value
    return func    

CHECKPOINT_DIR = './PPO-policy-training'
LOG_DIR_PPO = './PPO-policy-logs'
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
def main():
    # Environment setup:
    # The level played by the AI is the first level within the Super Mario Bros video game. 
    env = Monitor(gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human"))
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = DummyVecEnv([lambda:env])
    """
    The line of code below creates the PPO agent, and was used when training the model. This is not necessary when loading the model, hence, it has been commented out. 
    agent_model = PPO("MlpPolicy", env, verbose=1,n_steps=500, learning_rate=linear_schedule(3e-4), tensorboard_log="LOG_DIR_PPO" )
    """
    #SOLUTION TO PROBLEM OF 'SEED' SOURCED FROM https://stackoverflow.com/questions/76509663/typeerror-joypadspace-reset-got-an-unexpected-keyword-argument-seed-when-i
    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)
    """
    The below two lines of code have been commented out for the sake of loading this model for submission.
    Code for training purposes: this trains and saves the agents every 500,000 timesteps
    agent_model.learn(total_timesteps=500000, callback=callback)
    agent_model.save("./MLP_POLICY_REINFORCED_PPO_AGENT")
    """
    agent_model = PPO.load("./MLP_POLICY_REINFORCED_PPO_AGENT") #This line of code was commented out during training
    obs = env.reset()

    # For the sake of submission, this loop will iterate through 10,000 steps before closing the environment.
    # In this loop the agent will predict the best moves for Mario based on the 500,000 timestep training model on version-0 world1-stage1.
    for step in range(10000):
        action, _states = agent_model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render() 
    env.close()
print(main())