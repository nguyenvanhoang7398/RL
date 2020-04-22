import gym
import gym_grid_driving
import collections
import numpy as np
import random
import math
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gym_grid_driving.envs.grid_driving import LaneSpec

from utils import *
from parking.env import construct_task2_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_path, 'model.pt')
reward_shaping_path = "reward_shaping_large_43_6.p"

# Hyperparameters --- don't change, RL is very sensitive
learning_rate = 0.001
gamma = 0.98
buffer_limit = 5000
batch_size = 32
max_episodes = 50000
t_max = 600
min_buffer = 1000
target_update = 20  # episode(s)
train_steps = 10
max_epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 500
print_interval = 20

converge_th = 0.001
max_converge_cnt = 20 * print_interval

Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer():
    def __init__(self, buffer_limit=buffer_limit):
        '''
        FILL ME : This function should initialize the replay buffer `self.buffer` with maximum size of `buffer_limit` (`int`).
                  len(self.buffer) should give the current size of the buffer `self.buffer`.
        '''
        self.buffer_limit = buffer_limit
        self.buffer = []
        self.cur = 0

    def push(self, transition):
        '''
        FILL ME : This function should store the transition of type `Transition` to the buffer `self.buffer`.

        Input:
            * `transition` (`Transition`): tuple of a single transition (state, action, reward, next_state, done).
                                           This function might also need to handle the case  when buffer is full.

        Output:
            * None
        '''
        if len(self.buffer) < self.buffer_limit:
            self.buffer.append(None)
        self.buffer[self.cur] = transition
        self.cur = (self.cur + 1) % self.buffer_limit

    def sample(self, batch_size):
        '''
        FILL ME : This function should return a set of transitions of size `batch_size` sampled from `self.buffer`

        Input:
            * `batch_size` (`int`): the size of the sample.

        Output:
            * A 5-tuple (`states`, `actions`, `rewards`, `next_states`, `dones`),
                * `states`      (`torch.tensor` [batch_size, channel, height, width])
                * `actions`     (`torch.tensor` [batch_size, 1])
                * `rewards`     (`torch.tensor` [batch_size, 1])
                * `next_states` (`torch.tensor` [batch_size, channel, height, width])
                * `dones`       (`torch.tensor` [batch_size, 1])
              All `torch.tensor` (except `actions`) should have a datatype `torch.float` and resides in torch device `device`.
        '''
        transitions = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor([t.state for t in transitions]).to(device)
        actions = torch.LongTensor([t.action for t in transitions]).to(device)
        rewards = torch.FloatTensor([t.reward for t in transitions]).to(device)
        next_states = torch.FloatTensor([t.next_state for t in transitions]).to(device)
        dones = torch.BoolTensor([t.done for t in transitions]).to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        '''
        Return the length of the replay buffer.
        '''
        return len(self.buffer)


class Base(nn.Module):
    '''
    Base neural network model that handles dynamic architecture.
    '''

    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.construct()

    def construct(self):
        raise NotImplementedError

    def forward(self, x):
        if hasattr(self, 'features'):
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

    def feature_size(self):
        x = autograd.Variable(torch.zeros(1, *self.input_shape))
        if hasattr(self, 'features'):
            x = self.features(x)
        return x.view(1, -1).size(1)


class BaseAgent(Base):
    def act(self, state, epsilon=0.0):
        if not isinstance(state, torch.FloatTensor):
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        '''
        FILL ME : This function should return epsilon-greedy action.

        Input:
            * `state` (`torch.tensor` [batch_size, channel, height, width])
            * `epsilon` (`float`): the probability for epsilon-greedy

        Output: action (`Action` or `int`): representing the action to be taken.
                if action is of type `int`, it should be less than `self.num_actions`
        '''

        # random for exploration - exploitation
        if random.random() < epsilon:
            # choose exploration
            return random.randrange(self.num_actions)
        else:
            # choose exploitation
            with torch.no_grad():
                # do not compute gradient for this forward
                logits = self(state)
                action = int(torch.argmax(logits).detach().cpu())
                return action


class DQN(BaseAgent):
    def construct(self):
        self.layers = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )


class ConvDQN(DQN):
    def construct(self):
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
        )
        super().construct()


def compute_loss(model, target, states, actions, rewards, next_states, dones):
    '''
    FILL ME : This function should compute the DQN loss function for a batch of experiences.

    Input:
        * `model`       : model network to optimize
        * `target`      : target network
        * `states`      (`torch.tensor` [batch_size, channel, height, width])
        * `actions`     (`torch.tensor` [batch_size, 1])
        * `rewards`     (`torch.tensor` [batch_size, 1])
        * `next_states` (`torch.tensor` [batch_size, channel, height, width])
        * `dones`       (`torch.tensor` [batch_size, 1])

    Output: scalar representing the loss.

    References:
        * MSE Loss  : https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss
        * Huber Loss: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss
    '''
    state_action_values = model(states).gather(1, actions)
    non_final_mask = dones.squeeze(-1) == False
    non_final_next_states = next_states[non_final_mask]
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) + rewards.squeeze(-1)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    return loss


def optimize(model, target, memory, optimizer):
    '''
    Optimize the model for a sampled batch with a length of `batch_size`
    '''
    batch = memory.sample(batch_size)
    loss = compute_loss(model, target, *batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def compute_epsilon(episode):
    '''
    Compute epsilon used for epsilon-greedy exploration
    '''
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)
    return epsilon


def reward_shape_coord(x, y, next_x, next_y, reward, reward_shaping_mtx):
    alpha = 1
    f_reward = reward_shaping_mtx[next_x][next_y] - reward_shaping_mtx[x][y]
    shaped_reward = reward + alpha * f_reward
    return shaped_reward


def reward_shape(state, next_state, reward, done, reward_shaping_mtx):
    pos_next_state = np.where(next_state[1] == 1)
    next_x, next_y = int(pos_next_state[1]), int(pos_next_state[0])
    pos_state = np.where(state[1] == 1)
    x, y = int(pos_state[1]), int(pos_state[0])

    goal_state = np.where(state[2] == 1)
    goal_x, goal_y = int(goal_state[1]), int(goal_state[0])

    if goal_x != next_x and goal_y != next_y and not done:
        # if the next state is done but not goal, decrease the reward of this state
        reward += -10
    return reward_shape_coord(x, y, next_x, next_y, reward, reward_shaping_mtx)


def train(model_class, env, pretrained=None, reward_shaping_p=reward_shaping_path, input_t_max=None):
    '''
    Train a model of instance `model_class` on environment `env` (`GridDrivingEnv`).

    It runs the model for `max_episodes` times to collect experiences (`Transition`)
    and store it in the `ReplayBuffer`. It collects an experience by selecting an action
    using the `model.act` function and apply it to the environment, through `env.step`.
    After every episode, it will train the model for `train_steps` times using the
    `optimize` function.

    Output: `model`: the trained model.
    '''
    if input_t_max is not None:
        t_max = input_t_max
    # Initialize model and target network
    if pretrained is None:
        model = model_class(env.observation_space.shape, env.action_space.n).to(device)
    else:
        model = pretrained
    target = model_class(env.observation_space.shape, env.action_space.n).to(device)
    target.load_state_dict(model.state_dict())
    target.eval()

    if os.path.exists(reward_shaping_p):
        reward_shaping_mtx = load_from_pickle(reward_shaping_p)
        use_reward_shaping = True
    else:
        n_lanes, n_width = len(env.lanes), env.width
        reward_shaping_mtx = np.zeros(n_width, n_lanes)
        use_reward_shaping = False

    print("Reward shaping mtx")
    print(reward_shaping_mtx.T)

    # Initialize replay buffer
    memory = ReplayBuffer()

    print(model)

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    converge_cnt = 0

    for episode in range(max_episodes):
        epsilon = compute_epsilon(episode)
        state = env.reset()
        episode_rewards = 0.0

        for t in range(t_max):
            # Model takes action
            action = model.act(state, epsilon)

            # Apply the action to the environment
            next_state, reward, done, info = env.step(action)

            reward = reward_shape(state, next_state, reward, done, reward_shaping_mtx)

            # Save transition to replay buffer
            memory.push(Transition(state, [action], [reward], next_state, [done]))

            state = next_state
            episode_rewards += reward
            if done:
                break
        rewards.append(episode_rewards)

        # Train the model if memory is sufficient
        norm_factor = 1 if input_t_max is None else input_t_max
        if len(memory) > min_buffer:
            if (np.mean(rewards[print_interval:]) < 0.1 and not use_reward_shaping) or \
                    (np.mean(rewards[print_interval:]) / norm_factor < -5.0 and use_reward_shaping):
                print('Bad initialization. Please restart the training.')
                return None
            for i in range(train_steps):
                loss = optimize(model, target, memory, optimizer)
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())

        if episode % print_interval == 0 and episode > 0:
            print(
                "[Episode {}]\tavg rewards : {:.3f},\tavg loss: : {:.6f},\tbuffer size : {},\tepsilon : {:.1f}%".format(
                    episode, np.mean(rewards[print_interval:]), np.mean(losses[print_interval * 10:]), len(memory),
                    epsilon * 100))

        # convergence check
        if len(rewards) >= 2 and np.abs(rewards[-1] - rewards[-2]) < converge_th:
            converge_cnt += 1
            if converge_cnt == max_converge_cnt:
                print("Training converges")
                break
        else:
            # reset converge count
            converge_cnt = 0
    return model


def test(model, env, max_episodes=600):
    '''
    Test the `model` on the environment `env` (GridDrivingEnv) for `max_episodes` (`int`) times.

    Output: `avg_rewards` (`float`): the average rewards
    '''
    rewards = []
    for episode in range(max_episodes):
        state = env.reset()
        episode_rewards = 0.0
        for t in range(t_max):
            action = model.act(state)
            state, reward, done, info = env.step(action)
            episode_rewards += reward
            if done:
                break
        rewards.append(episode_rewards)
    avg_rewards = np.mean(rewards)
    print("{} episodes avg rewards : {:.1f}".format(max_episodes, avg_rewards))
    return avg_rewards


def get_model():
    '''
    Load `model` from disk. Location is specified in `model_path`.
    '''
    model_class, model_state_dict, input_shape, num_actions = torch.load(model_path)
    model = eval(model_class)(input_shape, num_actions).to(device)
    model.load_state_dict(model_state_dict)
    return model


def save_model(model, model_path=model_path):
    '''
    Save `model` to disk. Location is specified in `model_path`.
    '''
    data = (model.__class__.__name__, model.state_dict(), model.input_shape, model.num_actions)
    torch.save(data, model_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train and test DQN agent.')
    parser.add_argument('--train', dest='train', action='store_true', help='train the agent')
    args = parser.parse_args()

    default_env = construct_task2_env()

    print("Training on {} x {} environment".format(len(default_env.lanes), default_env.width))

    if args.train:
        model = None
        while model is None:
            model = train(ConvDQN, default_env)
        save_model(model)
    else:
        model = get_model()
    test(model, default_env, max_episodes=600)