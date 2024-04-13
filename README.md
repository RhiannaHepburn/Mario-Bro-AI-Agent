# CITS3001_Project
CITS3001 Algorithms, Agents and Artifical Intelligence Project Repository. 
Authors: Brittany Carlsson (22752092), Rhianna Hepburn (23340238)  

## Agents That Will Be Run
---------------------------
- Rule-Based Agent *handwritten*.
- Proximal Policy Optimization PPO *implemented using Stablebaselines*.
- DQN Agent *implemented using Stablebaselines*.

## Setup
-------------------------
Below are the steps for setting up the environment:
*Note* Anaconda Python is the environment that has been used for the development and testing of the PPO and Rule-Based agents listed above. Visual Studio Code was the IDE used for the development and testing of the code. 
### Anaconda 
- Step One:
    Set up the environment using the steps provided in the Project Description for Anaconda Python. Ensuring that the "mario" environment is activated and that the python version is 3.8. 
    ```
    conda create -n mario python=3.8
        then activate the environment.
    conda activate mario
    ```
    - excerpt taken from the Project description. 

- Step Two: 
    If using Visual Studio Code as the IDE, open the folder containing the submission documents. 
    
    (IFF Python is installed. If python is not installed, install it before running this command).
    Then type into a Python terminal (if using python3, replace python here with python3):
    ```
    pip install -r PPO_Requirements.txt
    ```
    This will install any requirements that are necessary in to run the agents. 

### Venv
For the DQN agent, the environment uses venv instead of Anaconda. It is important to note that the python version should be *<3.8* to ensure that the imports are compatible 

- Step One:
    Set up the environment using the following steps:
    ```
    python -m venv .env
    ```
    source .env/bin/activate (This will be different for Windows architecture: use .env\Scripts\activate)
    ```
    pip install -r DQNrequirements.txt


## How To Run Each Agent
--------------------------
### Rule-Based Agent
- Step One:
    To run the rule-based agent on the first world and first stage (SuperMarioBros-v0 1-1), enter the following command into the terminal (*Note* use Python3 if that is what is installed on your system):
    
    ```
    python Rulebased.py
    ```
    Mario should capture the flag in this world and stage.

- Step Two (Optional):
    To run the rule-based agent on the overworld stages as discussed in the report, run the following command in the terminal (*Note* use Python3 if that is what is installed on your system):
    ```
    python Rulebased_Random_Stages.py
    ```

Outcome:
A string will be returned in the terminal, displaying the performance matrix data, along with a message describing Mario's state at the end of the execution. 

### PPO Agent
- Step One:
    To run the PPO agent on the first world and first stage (SuperMarioBros-v0 1-1), enter the following command into the terminal (*Note* use Python3 if that is what is installed on your system):
    
    ```
    python PPO_Agent.py
    ```
- Step Two (Optional):
    To run the PPO agent on random stages (trained on random stages too), run the following command in the terminal (*Note* use Python3 if that is what is installed on your system):
    ```
    python PPO_Agent_Random_Stages.py
    ```
Outcome:
The code will run for 10,000 steps before closing the environment. 

### DQN Agent (Optional)
- Step One:
    To run the DQN agent on random stages, using the model trained on world-1 stage-1,  enter the following command into the terminal (*Note* use Python3 if that is what is installed on your system):
    The model loaded and Super Mario Bro version can be changed to view different agents 
    ```
    python DQN_evaluation.py
    ```

Outcome:
This runs the DQN agent through 1 episode. 









