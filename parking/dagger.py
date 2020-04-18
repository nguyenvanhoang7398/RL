from parking.mcts import MonteCarloTreeSearch
import torch


class DAGGER(object):
    """
    DAGGER Imitation learning with MCTS
    """
    def __init__(self, N, T, epsilon, policy_net, env, device):
        self.N = N
        self.T = T
        self.policy_net = policy_net
        self.env = env
        self.device = device
        self.mcts = self.init_mcts()

    def init_mcts(self, numiters=100, explorationParam=1., random_seed=42):
        return MonteCarloTreeSearch(env=self.env, numiters=numiters, explorationParam=explorationParam,
                                    random_seed=random_seed)

    def run(self):
        self.policy_net.train()

        # iterate through epochs
        for _ in range(self.N):
            trajectory_states = [self.env.reset()]

            # sample T -step trajectory
            for _ in range(self.T):
                prev_states = trajectory_states[-1]
                logits = self.policy_net(prev_states)
                action = int(torch.argmax(logits).detach().cpu())