from parking.dagger import *
from copy import deepcopy


def debug():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = ConvDQN(tensor_env.observation_space.shape, tensor_env.action_space.n).to(device)

    dagger = DAGGER(n_lanes, n_width, policy_net, env, device)

    c = 0
    max_actions = 3
    env.reset()

    start_env = deepcopy(env)
    print("Start")
    env.render()
    print_state_tensor(env.world.as_tensor())
    while c < max_actions and not env.done:
        grid_word_state = GridWorldState(env.state)
        print("Before Env")
        env.render()
        action = dagger.mcts.buildTreeAndReturnBestAction(initialState=grid_word_state)
        print("After Env")
        env.render()
        done = env.step(state=deepcopy(grid_word_state.state), action=action)[2]
        print(action)
        env.render()
        c += 1

    print("Start")
    start_env.render()
