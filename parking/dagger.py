from parking.mcts import MonteCarloTreeSearch, GridWorldState
from dqn import ConvDQN, AtariDQN
from copy import deepcopy
from gym.utils import seeding
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from dqn import save_model, get_model
from tqdm import tqdm
from scipy.special import softmax
from parking.utils import *
from parking import test_single, ExampleAgent

np.set_printoptions(linewidth=400, precision=2)

n_episodes = 5000
n_epochs = 20
trajectory_len = 50
learning_rate = 0.001
converge_max = 5
converge_margin = 1e-12
batch_size = 32
save_eval_per_episodes = 5
train_per_episodes = 100
n_eval_runs = 10
min_positive_examples = 0
random, seed = seeding.np_random(random_seed)

# print("Shape: {}".format(reward_shaping_mtx.shape))

# print(reward_shaping_mtx.T)


def randomPolicy(state, env, _):
    '''
    Policy followed in MCTS simulation for playout
    '''
    global random
    reward = 0.
    cnt = 0
    while not state.isDone() and cnt < max_playout_step:
        actions = [a for a in env.actions if str(a) != "down"]
        action = np.random.choice(actions, replace=False, p=[0.85, 0.05, 0.05, 0.05])
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
            reward += -4

        reward = reward_shape_coord(cur_x, cur_y, next_x, next_y, reward,
                                    reward_shaping_mtx)
        reward += state.getReward()
        state = next_state
        cnt += 1
    return reward


class DAGGER(object):
    """
    DAGGER Imitation learning with MCTS
    """
    def __init__(self, n_lanes, n_width, policy_net, env, debug=False):
        self.n_episodes = n_episodes
        self.trajectory_len = trajectory_len
        self.n_epochs = n_epochs
        self.n_lanes = n_lanes
        self.n_width = n_width
        self.learning_rate = learning_rate
        self.policy_net = policy_net
        self.env = env
        self.mcts = self.init_mcts()
        self.action_map = self.init_action_map()
        self.debug = debug
        self.exp_name = get_exp_name("dagger", "dqn")

    def init_mcts(self, explorationParam=explorationParam, random_seed=random_seed):
        # by default use random policy for playout policy
        # not so sure if this is optimal, why can't we use the policy that is being trained as the playout policy?
        return MonteCarloTreeSearch(env=self.env, numiters=numiters, explorationParam=explorationParam,
                                    playoutPolicy=agentPolicy, random_seed=random_seed,
                                    reward_shaping_mtx=reward_shaping_mtx, policy_net=self.policy_net)

    def init_action_map(self):
        action_map = {}
        for i, action in enumerate(self.env.actions):
            action_map[str(action)] = i
        return action_map

    def run(self):
        all_train_x, all_test_x, all_train_y, all_test_y = [], [], [], []
        all_train_reward_margins, all_test_reward_margins = [], []

        print("Start testing first before running")
        eval_rewards = []
        agent = ExampleAgent(test_case_id=0, model=self.policy_net)
        for _ in range(n_eval_runs):
            if curr_id is not None:
                eval_env = construct_curriculum_env(curr_id)
            else:
                eval_env = construct_task2_env()
            reward = test_single(agent, eval_env, t_max=trajectory_len, render_step=False)
            eval_rewards.append(reward)
        print("Average reward: {}".format(np.mean(eval_rewards)))
        train_ep = 0

        # iterate through epochs
        for episode in tqdm(range(self.n_episodes), desc="Iterating episodes"):
            self.env.reset()
            # self.env.render()
            self.policy_net.eval()

            trajectory_tensor_states = [self.env.world.as_tensor()]
            trajectory_normal_states = [deepcopy(self.env.state)]

            agent_actions = []
            agent_logits = []
            expert_actions = []
            reward_margins = []

            successful = False

            # sample T -step trajectory
            for i in tqdm(range(self.trajectory_len), desc="Iterating trajectory"):
                prev_states = trajectory_tensor_states[-1]
                prev_state_tensor = torch.FloatTensor([prev_states]).to(device)
                logits = self.policy_net(prev_state_tensor).squeeze(0)
                logit_cpu = softmax(logits.detach().cpu().numpy())
                action = int(np.argmax(logit_cpu))
                # action = int(torch.argmax(logits).detach().cpu())
                agent_actions.append(action)
                agent_logits.append(logit_cpu)
                _, reward, done, info = self.env.step(action)
                if done or i == self.trajectory_len - 1:
                    if reward == 10:    # successful run
                        successful = True
                    break
                trajectory_tensor_states.append(self.env.world.as_tensor())
                trajectory_normal_states.append(deepcopy(self.env.state))

            if successful:
                print("Successful run, no need to run imitation learning")
                # for i, state in enumerate(trajectory_tensor_states):
                #     all_train_x.append(trajectory_tensor_states[i])
                #     all_train_y.append(agent_actions[i])
                #     all_train_reward_margins.append(1.0/len(trajectory_tensor_states))
                continue
            else:
                traj_len = len(trajectory_tensor_states)
                print("Run fails with trajectory length: {}".format(traj_len))
                if len(all_train_x) < min_positive_examples:
                    print("Skip MCTS due to only {} positive examples".format(len(all_train_x)))
                    continue
                if traj_len == 1:
                    print("Skip due to short trajectory length")
                    continue

            data_x, data_y = [], []

            # start building expert data for imitation learning
            # only cares about the failed step
            for i, state in enumerate(tqdm(trajectory_normal_states, desc="Run MCTS")):
                if i < len(trajectory_normal_states)-1:
                    continue
                grid_word_state = GridWorldState(state)
                agent_action = self.env.actions[agent_actions[i]]
                expert_action, q_map = self.mcts.buildTreeAndReturnBestAction(initialState=grid_word_state)
                expert_actions.append(expert_action)

                if str(agent_action) == "down":
                    reward_margin = 10
                else:
                    reward_margin = q_map[str(expert_action)] - q_map[str(agent_action)]
                if str(expert_action) == "up" and str(agent_action) != "up":
                    # prioritizing up action
                    reward_margin += ((n_width - i) / 10)
                reward_margins.append(reward_margin)
                data_x.append(trajectory_tensor_states[i])
                data_y.append(self.action_map[str(expert_action)])

            # if self.debug:
            #     for i, state in enumerate(trajectory_tensor_states):
            #         print("State")
            #         print_state_tensor(state)
            #         if i < len(agent_actions):
            #             print("Agent action: {}".format(self.env.actions[agent_actions[i]]))
            #             print("Expert action: {}, margin: {}".format(expert_actions[i], reward_margins[i]))

            print_state_tensor(trajectory_tensor_states[-1])
            print("Agent action: {}".format(self.env.actions[agent_actions[-1]]))
            print("Expert action: {}, margin: {}".format(expert_actions[-1], reward_margins[0]))

            # if len(data_x) > 1:
            if False:
                train_idx, test_idx = train_test_split(range(len(data_x)), test_size=0.1)
                train_x = [data_x[i] for i in train_idx]
                test_x = [data_x[i] for i in test_idx]
                train_y = [data_y[i] for i in train_idx]
                test_y = [data_y[i] for i in test_idx]
                train_reward_margin = [reward_margins[i] for i in train_idx]
            else:
                train_x, test_x, train_y, test_y = [], [], [], []
                train_reward_margin = []
                for i, (margin, x, y) in enumerate(zip(reward_margins, data_x, data_y)):
                    if margin > 0.:
                        train_reward_margin.append(min(margin, 6)+1e-6)     # smoothing
                        train_x.append(x)
                        train_y.append(y)
            if len(train_reward_margin) == 0:
                continue
            train_reward_margin = softmax(train_reward_margin)

            # only train the significant mistakes
            print(train_reward_margin)

            # add some smoothing coeff to make sure all examples are concerned
            # train_reward_margin += 1e-9

            all_train_reward_margins.extend(train_reward_margin)
            all_train_x.extend(train_x)
            all_train_y.extend(train_y)
            all_test_x.extend(test_x)
            all_test_y.extend(test_y)

            dataset_size = len(all_train_x)

            if episode % train_per_episodes == 0:

                # start training policy net to imitate the expert
                self.policy_net.train()
                loss_fn = nn.CrossEntropyLoss(reduction="none")
                optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

                converge_cnt, last_loss = 0, None

                # training imitation learning
                for epoch in tqdm(range(self.n_epochs), desc="Training"):
                    batch_idx, epoch_loss = 0, 0.
                    while batch_idx < dataset_size:
                        batch_train_x = all_train_x[batch_idx: batch_idx + batch_size]
                        batch_train_y = all_train_y[batch_idx: batch_idx + batch_size]
                        batch_reward_margin = all_train_reward_margins[batch_idx: batch_idx + batch_size]

                        train_x_tensor = torch.FloatTensor(batch_train_x).to(device)
                        train_y_tensor = torch.LongTensor(batch_train_y).to(device)
                        batch_reward_margin_tensor = torch.FloatTensor(batch_reward_margin).to(device)

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

                # self.policy_net.eval()
                # evaluate training
                train_x_tensor = torch.FloatTensor(all_train_x).to(device)
                logits = self.policy_net(train_x_tensor)
                train_predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
                f1_val = f1_score(all_train_y, train_predictions, average="macro")
                print("Episode: {} | Loss: {} | F1: {} | Confusion matrix".format(episode, epoch_loss, f1_val))
                print(confusion_matrix(all_train_y, train_predictions))
                train_ep += 1
                #
                # # evaluate testing imitation learning
                # test_x_tensor = torch.FloatTensor(all_test_x).to(device)
                # logits = self.policy_net(test_x_tensor)
                # predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
                # f1_val = f1_score(all_test_y, predictions, average="macro")
                # print("Episode: {} | F1: {} | Confusion matrix".format(episode, f1_val))
                # print(confusion_matrix(all_test_y, predictions))

            if (train_ep + 1) % save_eval_per_episodes == 0:
                self.policy_net.eval()
                print("Test and save model")
                model_path = os.path.join("models", self.exp_name, "{}.pt".format(episode))
                train_x_path = os.path.join("models", self.exp_name, "train_x.p")
                train_y_path = os.path.join("models", self.exp_name, "train_y.p")

                ensure_path(model_path)
                save_model(self.policy_net, model_path)
                save_to_pickle(all_train_x, train_x_path)
                save_to_pickle(all_train_y, train_y_path)

                eval_rewards = []
                agent = ExampleAgent(test_case_id=0, model_path=model_path)
                for _ in range(n_eval_runs):
                    if curr_id is not None:
                        eval_env = construct_curriculum_env(curr_id)
                    else:
                        eval_env = construct_task2_env()
                    reward = test_single(agent, eval_env, t_max=trajectory_len, render_step=False)
                    eval_rewards.append(reward)
                print("Average reward: {}".format(np.mean(eval_rewards)))

        return self.policy_net


def run_dagger():
    # policy_net = ConvDQN(tensor_env.observation_space.shape, tensor_env.action_space.n).to(device)
    # policy_net = AtariDQN(tensor_env.observation_space.shape, tensor_env.action_space.n).to(device)
    policy_net = get_model("model_tuned_recent.pt")
    # policy_net = get_model("models/one_eight/824.pt")

    dagger = DAGGER(n_lanes, n_width, policy_net, env, debug=True)
    dagger.run()


def test_dagger(model_path, test_runs=50):
    eval_rewards = []
    agent = ExampleAgent(test_case_id=0, model_path=model_path)
    for _ in range(test_runs):
        if curr_id is not None:
            eval_env = construct_curriculum_env(curr_id)
        else:
            eval_env = construct_task2_env()
        reward = test_single(agent, eval_env, t_max=trajectory_len, render_step=False)
        eval_rewards.append(reward)
    print("Average reward: {}".format(np.mean(eval_rewards)))
