import sys
import gym
import zmq
import json

##
# OpenAI Gym State
##
class Environment:
    def __init__(self, environment_name):
        self.sim = gym.make(environment_name)
        self.environment_name = environment_name
        self.reset()
    def step(self, action):
        # [TODO] Check to see if 'action' is valid
        self.state, self.reward, self.done, self.info = self.sim.step(action)
        self.score += self.reward
    def reset(self):
        self.state = self.sim.reset()
        self.reward = 0
        self.score = 0
        self.done = False
        self.info = {}
    def preprocess(self, state):
        raise NotImplementedError
    def get_state(self, preprocess = False):
        state = self.state
        if preprocess:
            state = self.preprocess(state)
        return state


##
# Pong Specific Environment Information
##
import cv2
class PongEnv(Environment):
    def __init__(self):
        super(PongEnv, self).__init__("PongNoFrameskip-v4")
    def preprocess(self, state):
        # Grayscale
        frame = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        # Crop irrelevant parts
        frame = frame[34:194, 15:145] # Crops to shape (160, 130)
        # Downsample
        frame = cv2.resize(frame, (80, 80), interpolation=cv2.INTER_AREA)
        # Normalize
        frame = frame / 255
        return frame

#
## Response Methods
#
def perform_action(msg, env, socket):
    if 'id' not in msg: # [TODO] Include an integer check
        socket.send_string("ERROR: 'id' not in message when type is set to 'action'.")

    action = int(msg['id'])
    env.step(action)

    p = msg['preprocess'] is not None and msg['preprocess']
    content = {}
    content['state'] = env.get_state(preprocess = p)
    content['reward'] = env.reward
    content['done'] = env.done
    content['info'] = env.info
    socket.send_pyobj(content)

def reset_env(msg, env, socket):
    env.reset()
    # Preprocess if part of message is set and equal to True
    p = msg['preprocess'] is not None and msg['preprocess']
    socket.send_pyobj(env.get_state(preprocess = p))

def respond_query(msg, env, socket):
    if 'items' not in msg:
        socket.send_string("ERROR: items not found in query message")
    # [TODO] Have a check here to make sure msg['items'] is a list
    response = {}
    for item in msg['items']:
        if item == "environment_name":
            response[item] = env.environment_name
        elif item == "action_space":
            response[item] = env.sim.action_space
        elif item == "observation_space":
            response[item] = env.sim.observation_space
        elif item == "reward_range":
            response[item] = env.sim.reward_range
        elif item == "metadata":
            response[item] = env.sim.metadata
        elif item == "action_meanings":
            response[item] = env.sim.unwrapped.get_action_meanings()
        elif item == "state":
            p = msg['preprocess'] is not None and msg['preprocess']
            response[item] = env.get_state(preprocess = p)
        elif item == "cumulative_reward":
            response[item] = env.score
        elif item == "reward":
            response[item] = env.reward
        elif item == "done":
            response[item] = env.done
        elif item == "info":
            response[item] = env.info
    socket.send_pyobj(response)

def respond_set(msg, env, socket):
    if 'type' not in msg:
        socket.send_string("ERROR: type not found in JSON message.")
    if msg['type'] == 'reset':
        reset_env(msg, env, socket)
    elif msg['type'] == 'action':
        perform_action(msg, env, socket)
    pass

def respond(msg, env, socket):
    if 'method' not in msg:
        socket.send_string("ERROR: method not found in JSON message.")
    # From here establish what type of message it is....
    # Maybe create a switch statement of some sort that sends it off to each of the specialized functions already created below
    if msg['method'] == 'query':
        respond_query(msg, env, socket)
    elif msg['method'] == "set":
        respond_set(msg, env, socket)
    else:
        socket.send_string("ERROR: method is not 'query' or 'set'.")

#
## Main Routine
#

# [TODO] Make sure port is an int
if len(sys.argv) != 2:
    print("Usage: gymserver_zero.py <port>", file=sys.stderr)
    sys.exit(1)

env = PongEnv()
port = int(sys.argv[1])
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)
while True:
    msg = socket.recv_json()
    respond(msg, env, socket)


