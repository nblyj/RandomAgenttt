############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import torch
import random
import collections
from torch.autograd import Variable


class DQN:
    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each
        # gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a
    # transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network
        # parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transition):
        loc_input = transition[0]
        loc_input_tensor = torch.tensor(loc_input)
        for i in range(4):
            if i == transition[1]:
                prediction = self.q_network.forward(loc_input_tensor)[i]
        reward = transition[2]
        label = torch.tensor(reward)
        loss_mse = torch.nn.MSELoss()(prediction, label)
        return loss_mse
        # TODO

    @staticmethod
    def copy_network(aim_network):
        torch.save(aim_network.q_network.state_dict(), './model_state_dict.pt')
        target_network = DQN()
        target_network.q_network.load_state_dict(torch.load('./model_state_dict.pt'))
        return target_network


class ReplayBuffer:
    def __init__(self):
        self.container = collections.deque(maxlen=5000)

    def add_tuple(self, trans):
        self.container.append(trans)



class Network(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension):
        super(Network, self).__init__()
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


class Agent(Network, DQN, ReplayBuffer):

    # Function to initialise the agent
    def __init__(self):
        # Initialisation of the Network, DQN and ReplayBuffer
        Network.__init__(self, 2, 4)
        DQN.__init__(self)
        ReplayBuffer.__init__(self)

        self.dqn = DQN()
        self.dqn_target = self.dqn.copy_network(self.dqn)
        self.buffer = ReplayBuffer()
        self.gamma = 0.9

        # Set the episode length
        self.episode_length = 1000
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Here, the action is random, but you can change this
        #action = np.random.uniform(low=-0.01, high=0.01, size=2).astype(np.float32)
        self.dqn.optimiser.zero_grad()
        state_tensor = torch.tensor(state)
        q_next_torch = self.dqn.q_network.forward(state_tensor)
        q_next = q_next_torch.detach().numpy()
        index = np.argmax(q_next)
        p_basic = np.ones(4)*0.2
        p_basic[index] = 0.4
        action_discrete = self.p_random([0, 1, 2, 3], p_basic)
        action = self._discrete_action_to_continuous(action_discrete)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        reward = 1 - distance_to_goal
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        self.buffer.add_tuple(transition)
        if len(self.buffer.container) < 32:
            sample_num = len(self.buffer.container)
        else:
            sample_num = 32

        self.dqn.optimiser.zero_grad()
        minibatch = random.sample(self.buffer.container, sample_num)
        discrete = None
        in_list = []
        label_list = []
        prediction_list = []
        for tu in minibatch:
            state = tu[0]
            tu_action = tu[1]
            reward = tu[2]
            next_state = tu[3]
            in_list.append(state)

            self.dqn_target.optimiser.zero_grad()
            next_state_torch = torch.tensor(next_state)
            q_next_state_torch = self.dqn_target.q_network.forward(next_state_torch)
            q_next = q_next_state_torch.detach().numpy()
            q_next_max = np.max(q_next)
            out = float(reward+self.gamma*q_next_max)
            label_list.append(out)

            discrete = self._continuous_to_discrete(tu_action)

            prediction_torch = self.dqn.q_network.forward(torch.tensor(state))[discrete]
            prediction = float(prediction_torch.detach().numpy())
            prediction_list.append(prediction)

        minibatch_labels_tensor = torch.tensor(label_list)
        minibatch_labels_tensor = Variable(minibatch_labels_tensor, requires_grad=True)
        minibatch_prediction_tensor = torch.tensor(prediction_list)
        minibatch_prediction_tensor = Variable(minibatch_prediction_tensor, requires_grad=True)

        loss = torch.nn.MSELoss()(minibatch_prediction_tensor, minibatch_labels_tensor)

        loss.backward()
        self.dqn.optimiser.step()
        if self.num_steps_taken % 200 == 0:
            self.dqn_target = self.dqn.copy_network(self.dqn)

        # Now you can do something with this transition ...

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest
        # Q-value
        state_torch = torch.tensor(state)
        q_torch = self.dqn.q_network.forward(state_torch)
        q = q_torch.detach().numpy()
        q_index = np.argmax(q)
        p_greedy = np.ones(4) * 0.1
        p_greedy[q_index] = 0.7
        action_discrete = self.p_random([0, 1, 2, 3], p_greedy)
        action = self._discrete_action_to_continuous(action_discrete)
        return action

    def p_random(self, arr1, arr2):
        #assert len(arr1) == len(arr2)
        #assert sum(arr2) == 1, "Total rate is not 1."

        sup_list = [len(str(i).split(".")[-1]) for i in arr2]
        top = 10 ** max(sup_list)
        new_rate = [int(i * top) for i in arr2]
        rate_arr = []
        for i in range(1, len(new_rate) + 1):
            rate_arr.append(sum(new_rate[:i]))
        rand = random.randint(1, top)
        data = None
        for i in range(len(rate_arr)):
            if rand <= rate_arr[i]:
                data = arr1[i]
                break
        return data

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        elif discrete_action == 1:
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        elif discrete_action == 2:
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        else:
            continuous_action = np.array([-0.02, 0], dtype=np.float32)
        return continuous_action

    def _continuous_to_discrete(self, continuous_action):
        if continuous_action[0] == 0:
            if continuous_action[1] == 0.02:
                transfer_action = 0
            else:
                transfer_action = 2
        else:
            if continuous_action[0] == 0.02:
                transfer_action = 1
            else:
                transfer_action = 3
        return transfer_action
