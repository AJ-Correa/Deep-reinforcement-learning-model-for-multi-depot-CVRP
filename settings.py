LEARNING_RATE = 0.00002
BATCH_SIZE = 32
EPSILON_START = 0.9
EPSILON_END = 0.1
EPSILON_DECAY = 180
GAMMA = 0.99
NUM_EPOCHS = 1000
REPLAY_SIZE = 100000
TARGET_UPDATE = 20

USE_RAINBOW = 0

ALPHA = 0.6
BETA = 0.4
PRIOR_EPS = 1e-6

N_STEP = 4

ATOM_SIZE = 51
V_MAX = 10
V_MIN = -10

INSTANCE_DIR = "data/P/P-n16-k8.vrp"

DEVICE = "cpu"

HP_LIST = {"Episodes": NUM_EPOCHS,
           "Target Update": TARGET_UPDATE,
           "Minibatch": BATCH_SIZE,
           "Replay Memory": REPLAY_SIZE,
           "Eps. Start": EPSILON_START,
           "Eps. End": EPSILON_END,
           "Eps. Decay": EPSILON_DECAY,
           "Gamma": GAMMA,
           "Learning Rate": LEARNING_RATE}