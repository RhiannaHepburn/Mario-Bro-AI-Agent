# List of potential hyperparameters to tune

TIME_STEPS = 500000

FRAME_STACK = 4 
FRAME_SKIP = 8

# Neural network parameters
BATCH_SIZE = 192 
LEARNING_RATE = 1e-4
EXPLORATION_FRACT = 0.1 
P_REPLAY_ALPHA = 0.6
TRAINING_FREQ = 8 
TARGET_UPDATE_FREQ = 100000 