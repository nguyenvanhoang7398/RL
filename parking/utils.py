from datetime import datetime
import torch
from parking.env import *
from utils import *
import numpy as np
from dqn import reward_shaping_path, reward_shape_coord

curr_id = 3
numiters = 16
max_playout_step = 2
explorationParam = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_seed = 42


if curr_id is not None:
    env = construct_curriculum_env(curr_id, tensor_state=False)
    tensor_env = construct_curriculum_env(curr_id)

    # use curriculum reward shaping
    # reward_shaping_path = "reward_shaping_large.p"
    reward_shaping_path = "rs_large3.p"
else:
    # use default reward shaping from dqn
    env = construct_task2_env(tensor_state=False)
    tensor_env = construct_task2_env()
n_lanes, n_width = len(env.lanes), env.width

if os.path.exists(reward_shaping_path):
    print("Use reward shaping at {}".format(reward_shaping_path))
    reward_shaping_mtx = load_from_pickle(reward_shaping_path)
    reward_shaping_mtx[0][0] -= 10
    use_reward_shaping = True
else:
    print("Do not use reward shaping")
    reward_shaping_mtx = np.zeros(shape=(n_width, n_lanes))
    use_reward_shaping = False


def agentPolicy(state, env, policy_net):
    global random
    # print(state.state[3])
    reward = 0.
    cnt = 0
    while not state.isDone() and cnt < max_playout_step:
        state_np = env.world.as_tensor()
        state_tensor = torch.FloatTensor([state_np]).to(device)
        logits = policy_net(state_tensor).squeeze(0)
        action_idx = int(torch.argmax(logits).detach().cpu())
        action = env.actions[action_idx]

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
        else:
            reward = reward_shape_coord(cur_x, cur_y, next_x, next_y, reward,
                                        reward_shaping_mtx)
        reward += state.getReward()
        state = next_state
        cnt += 1
    return reward


def get_exp_name(task_name, model_name):
    return "{}-{}-{}".format(task_name, model_name, datetime.now().strftime("%D-%H-%M-%S").replace("/", "_"))


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