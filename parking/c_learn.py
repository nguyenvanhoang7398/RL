from parking import run_test
#from parking.reward_shaping import heuristic_reward
from parking.dagger import run_dagger
from parking.env import construct_task2_env
import argparse

env_quarter = construct_task2_env(conf=1)
env_half = construct_task2_env(conf=2)
env_threequarter = construct_task2_env(conf=3)
env_full = construct_task2_env(conf=0)

def run_c_learn():
    policy_net = run_dagger(0, env_quarter, "reward_shaping_quarter.p")
    policy_net = run_dagger(policy_net, env_half, "reward_shaping_half.p")
    policy_net = run_dagger(policy_net, env_threequarter, "reward_shaping_threequarter.p")
    policy_net = run_dagger(policy_net, env_full, "reward_shaping_full.p")
    return policy_net
