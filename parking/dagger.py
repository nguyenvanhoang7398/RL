from parking.mcts import MonteCarloTreeSearch, GridWorldState
from parking.env import construct_task2_env
from parking.models import ConvDQN
import torch
from dqn import reward_shape_coord, reward_shaping_path
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
from scipy.special import softmax
from parking.utils import *
from parking import test_single, ExampleAgent

np.set_printoptions(linewidth=400, precision=2)


random_seed = 42
n_episodes = 2000
n_epochs = 50
trajectory_len = 6
numiters = 10
learning_rate = 0.001
converge_max = 5
converge_margin = 1e-12
batch_size = 32
explorationParam = 0
save_eval_per_episodes = 5
n_eval_runs = 5

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
    reward_shaping_mtx = np.zeros(shape=(n_width, n_lanes))
    use_reward_shaping = False

# print("Shape: {}".format(reward_shaping_mtx.shape))
#
# print(reward_shaping_mtx.T)


def randomPolicy(state, env):
    '''
    Policy followed in MCTS simulation for playout
    '''
    global random
    reward = 0.
    while not state.isDone():
        action = random.choice(env.actions)
        next_state = state.simulateStep(env=env, action=action)
        reward = state.getReward()

        # update this new node reward with reward shaping
        cur_agent_pos = state.state.agent.position
        cur_x, cur_y = cur_agent_pos.x, cur_agent_pos.y
        next_agent_pos = next_state.state.agent.position
        next_x, next_y = next_agent_pos.x, next_agent_pos.y

        goal_pos = state.state.finish_position
        goal_x, goal_y = goal_pos.x, goal_pos.y
        done = next_state.is_done

        if goal_x != next_x and goal_y != next_y and done:
            # if the next state is done but not goal, decrease the reward of this state
            reward += -10

        reward = reward_shape_coord(cur_x, cur_y, next_x, next_y, reward,
                                    reward_shaping_mtx)
        # reward += state.getReward()
        state = next_state
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
        self.exp_name = get_exp_name("dagger", "dqn")

    def init_mcts(self, explorationParam=explorationParam, random_seed=random_seed):
        # by default use random policy for playout policy
        # not so sure if this is optimal, why can't we use the policy that is being trained as the playout policy?
        return MonteCarloTreeSearch(env=self.env, numiters=numiters, explorationParam=explorationParam,
                                    playoutPolicy=randomPolicy, random_seed=random_seed,
                                    reward_shaping_mtx=reward_shaping_mtx)

    def init_action_map(self):
        action_map = {}
        for i, action in enumerate(self.env.actions):
            action_map[str(action)] = i
        return action_map

    def run(self):
        all_train_x, all_test_x, all_train_y, all_test_y = [], [], [], []
        all_train_reward_margins, all_test_reward_margins = [], []

        # iterate through epochs
        for episode in tqdm(range(self.n_episodes), desc="Iterating episodes"):
            self.policy_net.eval()
            self.env.reset()

            trajectory_tensor_states = [self.env.world.as_tensor()]
            trajectory_normal_states = [deepcopy(self.env.state)]

            agent_actions = []
            expert_actions = []
            reward_margins = []

            # sample T -step trajectory
            for i in tqdm(range(self.trajectory_len), desc="Iterating trajectory"):
                prev_states = trajectory_tensor_states[-1]
                prev_state_tensor = torch.FloatTensor([prev_states]).to(self.device)
                logits = self.policy_net(prev_state_tensor).squeeze(0)
                action = int(torch.argmax(logits).detach().cpu())
                agent_actions.append(action)
                _, _, done, info = self.env.step(action)
                if done or i == self.trajectory_len - 1:
                    break
                trajectory_tensor_states.append(self.env.world.as_tensor())
                trajectory_normal_states.append(deepcopy(self.env.state))

            print("Trajectory length: {}".format(len(trajectory_tensor_states)))

            data_x, data_y = [], []

            # start building expert data for imitation learning
            for i, state in enumerate(tqdm(trajectory_normal_states, desc="Run MCTS")):
                grid_word_state = GridWorldState(state)
                agent_action = self.env.actions[agent_actions[i]]
                expert_action, q_map = self.mcts.buildTreeAndReturnBestAction(initialState=grid_word_state)
                expert_actions.append(expert_action)

                reward_margin = q_map[str(expert_action)] - q_map[str(agent_action)]

                reward_margins.append(reward_margin)
                data_x.append(trajectory_tensor_states[i])
                data_y.append(self.action_map[str(expert_action)])

            if self.debug:
                for i, state in enumerate(trajectory_tensor_states):
                    print("State")
                    print_state_tensor(state)
                    if i < len(agent_actions):
                        print("Agent action: {}".format(self.env.actions[agent_actions[i]]))
                        print("Expert action: {}, margin: {}".format(expert_actions[i], reward_margins[i]))

            if len(data_x) > 1:
                train_idx, test_idx = train_test_split(range(len(data_x)), test_size=0.1)
                train_x = [data_x[i] for i in train_idx]
                test_x = [data_x[i] for i in test_idx]
                train_y = [data_y[i] for i in train_idx]
                test_y = [data_y[i] for i in test_idx]
                train_reward_margin = [reward_margins[i] for i in train_idx]
            else:
                train_x, test_x, train_y, test_y = data_x, [], data_y, []
                train_reward_margin = reward_margins

            train_reward_margin = softmax(train_reward_margin)

            # add some smoothing coeff to make sure all examples are concerned
            train_reward_margin += 0.1

            all_train_reward_margins.extend(train_reward_margin)
            all_train_x.extend(train_x)
            all_train_y.extend(train_y)
            all_test_x.extend(test_x)
            all_test_y.extend(test_y)

            dataset_size = len(all_train_x)

            # start training policy net to imitate the expert
            self.policy_net.train()
            loss_fn = nn.CrossEntropyLoss(reduction="none")
            optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

            converge_cnt, last_loss = 0, None

            # training imitation learning
            print("Train {} samples, test {} samples".format(len(all_train_x), len(all_test_x)))
            for epoch in tqdm(range(self.n_epochs), desc="Training"):
                batch_idx, epoch_loss = 0, 0.
                while batch_idx < dataset_size:
                    batch_train_x = all_train_x[batch_idx: batch_idx + batch_size]
                    batch_train_y = all_train_y[batch_idx: batch_idx + batch_size]
                    batch_reward_margin = all_train_reward_margins[batch_idx: batch_idx + batch_size]

                    train_x_tensor = torch.FloatTensor(batch_train_x).to(self.device)
                    train_y_tensor = torch.LongTensor(batch_train_y).to(self.device)
                    batch_reward_margin_tensor = torch.FloatTensor(batch_reward_margin).to(self.device)

                    logits = self.policy_net(train_x_tensor)
                    loss = loss_fn(logits, train_y_tensor)
                    weighted_loss = loss * batch_reward_margin_tensor
                    loss_val = torch.sum(weighted_loss).detach().cpu().numpy()
                    optimizer.zero_grad()
                    weighted_loss.sum().backward()
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

                # print("Episode: {} | Epoch: {} | Loss {}".format(episode, epoch, epoch_loss))

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

            if (episode + 1) % save_eval_per_episodes == 0:
                print("Test and save model")
                model_path = os.path.join("models", self.exp_name, "{}.pt".format(episode))
                ensure_path(model_path)
                save_model(self.policy_net, model_path)

                eval_rewards = []
                agent = ExampleAgent(test_case_id=0, model_path=model_path)
                for _ in range(n_eval_runs):
                    eval_env = construct_task2_env()
                    reward = test_single(agent, eval_env, t_max=trajectory_len, render_step=False)
                    eval_rewards.append(reward)
                print("Average reward: {}".format(np.mean(eval_rewards)))

        return self.policy_net


def run_dagger():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = ConvDQN(tensor_env.observation_space.shape, tensor_env.action_space.n).to(device)

    dagger = DAGGER(n_lanes, n_width, policy_net, env, device, debug=True)
    dagger.run()