from parking.mcts import MonteCarloTreeSearch, GridWorldState
from parking.env import construct_task2_env
from parking.models import ConvDQN
import torch
from dqn import reward_shape, reward_shaping_path
from utils import *
import numpy as np
from copy import deepcopy
from gym.utils import seeding
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from dqn import save_model
from tqdm import tqdm

random_seed = 42
n_episodes = 2000
n_epochs = 50
trajectory_len = 5
numiters = 10
learning_rate = 0.001
converge_max = 5
converge_margin = 1e-6
batch_size = 32

env = construct_task2_env(tensor_state=False)
tensor_env = construct_task2_env()
n_lanes, n_width = len(env.lanes), env.width

random, seed = seeding.np_random(random_seed)

if os.path.exists(reward_shaping_path):
    print("Use reward shaping")
    reward_shaping_mtx = load_from_pickle(reward_shaping_path)
    use_reward_shaping = True
else:
    print("Do not use reward shaping")
    reward_shaping_mtx = np.zeros(n_width, n_lanes)
    use_reward_shaping = False
print("Shape: {}".format(reward_shaping_mtx.shape))


def print_state_tensor(state_tensor):
    cars = state_tensor[0]
    agent = state_tensor[1]
    goal = state_tensor[2]
    all_state = cars * 1 + goal * 2 + agent * 3

    def symbol(x):
        if x == 0:
            return "-"
        if x == 1:
            return "1"
        elif x == 2:
            return "F"
        else:
            return "<"

    for row in all_state:
        print("  ".join([symbol(x) for x in row]))


def randomPolicy(state, env):
    '''
    Policy followed in MCTS simulation for playout
    '''
    global random
    reward = 0.
    while not state.isDone():
        action = random.choice(env.actions)
        curr_state = env.world.as_tensor()
        state = state.simulateStep(env=env, action=action)
        next_state = env.world.as_tensor()
        reward = state.getReward()
        reward = reward_shape(curr_state, next_state, reward, False, reward_shaping_mtx)
        reward += state.getReward()
    return reward


class DAGGER(object):
    """
    DAGGER Imitation learning with MCTS
    """
    def __init__(self, n_lanes, n_width, policy_net, env, device, debug=False):
        self.n_episodes = n_episodes
        self.trajectory_len = trajectory_len
        self.n_epochs = n_epochs
        self.n_lanes = n_lanes
        self.n_width = n_width
        self.learning_rate = learning_rate
        self.policy_net = policy_net
        self.env = env
        self.device = device
        self.mcts = self.init_mcts()
        self.action_map = self.init_action_map()
        self.debug = debug

    def init_mcts(self, explorationParam=1., random_seed=random_seed):
        # by default use random policy for playout policy
        # not so sure if this is optimal, why can't we use the policy that is being trained as the playout policy?
        return MonteCarloTreeSearch(env=self.env, numiters=numiters, explorationParam=explorationParam,
                                    playoutPolicy=randomPolicy, random_seed=random_seed)

    def init_action_map(self):
        action_map = {}
        for i, action in enumerate(self.env.actions):
            action_map[str(action)] = i
        return action_map

    def run(self):
        best_f1_val = 0.

        all_train_x, all_test_x, all_train_y, all_test_y = [], [], [], []

        # iterate through epochs
        for episode in tqdm(range(self.n_episodes), desc="Iterating episodes"):
            self.policy_net.eval()
            self.env.reset()
            trajectory_tensor_states = [self.env.world.as_tensor()]
            trajectory_normal_states = [deepcopy(self.env.state)]

            agent_actions = []
            expert_actions = []

            # sample T -step trajectory
            for _ in tqdm(range(self.trajectory_len), desc="Iterating trajectory"):
                prev_states = trajectory_tensor_states[-1]
                prev_state_tensor = torch.FloatTensor([prev_states]).to(self.device)
                logits = self.policy_net(prev_state_tensor).squeeze(0)
                action = int(torch.argmax(logits).detach().cpu())
                agent_actions.append(action)
                _, _, done, info = self.env.step(action)
                if done:
                    break
                trajectory_tensor_states.append(self.env.world.as_tensor())
                trajectory_normal_states.append(deepcopy(self.env.state))

            print("Trajectory length: {}".format(len(trajectory_tensor_states)))

            data_x, data_y = [], []

            # start building expert data for imitation learning
            for i, state in enumerate(tqdm(trajectory_normal_states, desc="Run MCTS")):
                grid_word_state = GridWorldState(state)
                action = self.mcts.buildTreeAndReturnBestAction(initialState=grid_word_state)
                expert_actions.append(action)
                data_x.append(trajectory_tensor_states[i])
                data_y.append(self.action_map[str(action)])

            if self.debug:
                for i, state in enumerate(trajectory_tensor_states):
                    print("State")
                    print_state_tensor(state)
                    if i < len(agent_actions):
                        print("Agent action: {}".format(self.env.actions[agent_actions[i]]))
                        print("Expert action: {}".format(expert_actions[i]))

            if len(data_x) > 1:
                train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.1)
            else:
                train_x, test_x, train_y, test_y = data_x, [], data_y, []

            all_train_x.extend(train_x)
            all_train_y.extend(train_y)
            all_test_x.extend(test_x)
            all_test_y.extend(test_y)

            dataset_size = len(all_train_x)

            # start training policy net to imitate the expert
            self.policy_net.train()
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

            converge_cnt, last_loss = 0, None

            # training imitation learning
            print("Train {} samples, test {} samples".format(len(all_train_x), len(all_test_x)))
            for epoch in tqdm(range(self.n_epochs), desc="Training"):
                batch_idx, epoch_loss = 0, 0.
                while batch_idx < dataset_size:
                    batch_train_x = all_train_x[batch_idx: batch_idx + batch_size]
                    batch_train_y = all_train_y[batch_idx: batch_idx + batch_size]
                    batch_test_x = all_test_x[batch_idx: batch_idx + batch_idx]

                    train_x_tensor = torch.FloatTensor(batch_train_x).to(self.device)
                    train_y_tensor = torch.LongTensor(batch_train_y).to(self.device)

                    logits = self.policy_net(train_x_tensor)
                    loss = loss_fn(logits, train_y_tensor)
                    loss_val = loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss_val
                    batch_idx += batch_size

                epoch_loss /= dataset_size

                if last_loss is not None:
                    if np.abs(epoch_loss-last_loss) < converge_margin:
                        converge_cnt += 1
                    if converge_cnt == converge_max:
                        print("Training converges")
                        break
                last_loss = epoch_loss

                print("Episode: {} | Epoch: {} | Loss {}".format(episode, epoch, epoch_loss))

            self.policy_net.eval()
            # evaluate training
            train_x_tensor = torch.FloatTensor(all_train_x).to(self.device)
            logits = self.policy_net(train_x_tensor)
            train_predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
            f1_val = f1_score(all_train_y, train_predictions, average="macro")
            print("Episode: {} | F1: {} | Confusion matrix".format(episode, f1_val))
            print(confusion_matrix(all_train_y, train_predictions))

            # evaluate testing imitation learning
            test_x_tensor = torch.FloatTensor(all_test_x).to(self.device)
            logits = self.policy_net(test_x_tensor)
            predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
            f1_val = f1_score(all_test_y, predictions, average="macro")
            print("Episode: {} | F1: {} | Confusion matrix".format(episode, f1_val))
            print(confusion_matrix(all_test_y, predictions))
            if f1_val > best_f1_val:
                print("Update f1 score")
                best_f1_val = f1_val
                save_model(self.policy_net, "best_il_policy.pt")
        return self.policy_net


def run_dagger():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = ConvDQN(tensor_env.observation_space.shape, tensor_env.action_space.n).to(device)

    dagger = DAGGER(n_lanes, n_width, policy_net, env, device, debug=True)
    dagger.run()