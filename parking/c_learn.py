from dqn import train, save_model, test, ConvDQN, get_model
from parking.env import construct_curriculum_env
from parking.utils import get_exp_name
from utils import *
import os
import torch


def run_c_learn():
    pretrained_max_epsilon = 0.5
    exp_name = get_exp_name("curriculum", "dqn")

    one_eight_path = "reward_shaping_large_43_8.p"
    one_eight_env = construct_curriculum_env(0)
    one_eight_model_path = os.path.join("models", exp_name, "one_eight.pt")

    quarter_path = "reward_shaping_large_quarter.p"
    quarter_env = construct_curriculum_env(1)
    quarter_model_path = os.path.join("models", exp_name, "quarter.pt")

    half_path = "reward_shaping_large_half.p"
    half_env = construct_curriculum_env(2)
    half_model_path = os.path.join("models", exp_name, "half.pt")

    full_path = "reward_shaping_large.p"
    full_env = construct_curriculum_env(3)
    full_model_path = os.path.join("models", exp_name, "full.pt")

    print("=" * 20)
    print("Start octive curriculum learning")
    print("=" * 20)
    one_eight_model = get_model(os.path.join("models", "best_one_eight", "one_eight.pt"))
    # one_eight_model = None
    if one_eight_model is None:
        while one_eight_model is None:
            one_eight_model = train(ConvDQN, one_eight_env, pretrained=None, reward_shaping_p=one_eight_path, input_t_max=6)
        ensure_path(one_eight_model_path)
        save_model(one_eight_model, one_eight_model_path)
    print("Test one eight curriculum learning")
    test(one_eight_model, one_eight_env, input_tmax=6, max_episodes=100)

    print("=" * 20)
    print("Start quarter curriculum learning")
    print("=" * 20)
    quarter_model = get_model(os.path.join("models", "best_quarter", "quarter.pt"))
    # quarter_model = None
    if quarter_model is None:
        while quarter_model is None:
            quarter_model = train(ConvDQN, quarter_env, pretrained=one_eight_model, reward_shaping_p=quarter_path,
                                  input_t_max=13, max_epsilon=pretrained_max_epsilon)
        ensure_path(quarter_model_path)
        save_model(quarter_model, quarter_model_path)
    print("Test quarter curriculum learning")
    test(quarter_model, quarter_env, input_tmax=13, max_episodes=100)

    print("=" * 20)
    print("Start half curriculum learning")
    print("=" * 20)
    half_model = None
    if half_model is None:
        while half_model is None:
            half_model = train(ConvDQN, half_env, pretrained=quarter_model, reward_shaping_p=half_path,
                               max_epsilon=pretrained_max_epsilon, input_t_max=25)
        ensure_path(half_model_path)
        save_model(half_model, half_model_path)
    print("Test half curriculum learning")
    test(half_model, half_env, input_tmax=25, max_episodes=100)

    print("=" * 20)
    print("Start full curriculum learning")
    print("=" * 20)
    full_model = None
    if full_model is None:
        while full_model is None:
            full_model = train(ConvDQN, full_env, pretrained=half_model, reward_shaping_p=full_path,
                               max_epsilon=pretrained_max_epsilon, input_t_max=50)
        ensure_path(full_model_path)
        save_model(full_model, full_model_path)
    print("Test full curriculum learning")
    test(full_model, half_env, input_tmax=50, max_episodes=100)

