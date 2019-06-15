import zmq
import numpy


# [TODO] Error handling for if server is down
class Environment:
    def __init__(self, address, port):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://%s:%s" % (address, port))
        self.address = address
        self.port = port
        self.observation_space, self.action_space, self.reward_range, self.metadata, self.action_meanings = self.get_initial_metadata()    

    ##
    # Helper Functions
    ##
    def get_environment_name(self):
        self.socket.send_json({'method':'query', 'items':['environment_name']})
        return self.socket.recv_pyobj()['environment_name']
    def get_state(self, preprocess = False):
        self.socket.send_json({'method':'query', 'items':['state'], 'preprocess':preprocess})
        return self.socket.recv_pyobj()['state']
    def get_reward(self):
        self.socket.send_json({'method':'query', 'items':['reward']})
        return self.socket.recv_pyobj()['reward']
    def get_score(self):
        self.socket.send_json({'method':'query', 'items':['cumulative_reward']})
        return self.socket.recv_pyobj()['cumulative_reward']
    def get_done(self):
        self.socket.send_json({'method':'query', 'items':['done']})
        return self.socket.recv_pyobj()['done']
    def get_info(self):
        self.socket.send_json({'method':'query', 'items':['info']})
        return self.socket.recv_pyobj()['info']
    def get_observation_space(self):
        self.socket.send_json({'method':'query', 'items':['observation_space']})
        return self.socket.recv_pyobj()['observation_space']
    def get_action_space(self):
        self.socket.send_json({'method':'query', 'items':['action_space']})
        return self.socket.recv_pyobj()['action_space']
    def get_reward_range(self):
        self.socket.send_json({'method':'query', 'items':['reward_range']})
        return self.socket.recv_pyobj()['reward_range']
    def get_metadata(self):
        self.socket.send_json({'method':'query', 'items':['metadata']})
        return self.socket.recv_pyobj()['metadata']
    def get_action_meanings(self):
        self.socket.send_json({'method':'query', 'items':['action_meanings']})
        return self.socket.recv_pyobj()['action_meanings']
    def get_initial_metadata(self):
        self.socket.send_json({'method':'query', 'items':['observation_space', 'action_space', 'reward_range', 'metadata', 'action_meanings']})
        content = self.socket.recv_pyobj()
        return content['observation_space'], content['action_space'], content['reward_range'], content['metadata'], content['action_meanings']
    
    ##
    # Common API
    ##
    def reset(self, preprocess = False):
        self.socket.send_json({'method':'set', 'type':'reset', 'preprocess':preprocess})
        return self.socket.recv_pyobj()
    def step(self, action_id, preprocess = False):
        self.socket.send_json({'method':'set', 'type':'action', 'id': action_id, 'preprocess':preprocess})
        content = self.socket.recv_pyobj()
        return content['state'], content['reward'], content['done'], content['info']
    
# env = Environment("127.0.0.1", 5000)