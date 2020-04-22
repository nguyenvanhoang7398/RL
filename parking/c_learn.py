from dqn import train, save_model, test, ConvDQN
from parking.env import construct_curriculum_env
from parking.utils import get_exp_name
from utils import *
import os


def run_c_learn():
    exp_name = get_exp_name("curriculum", "dqn")

    quarter_path = "reward_shaping_large_quarter.p"
    quarter_env = construct_curriculum_env(0)
    quarter_model_path = os.path.join("models", exp_name, "quarter.pt")

    half_path = "reward_shaping_large_half.p"
    half_env = construct_curriculum_env(1)
    half_model_path = os.path.join("models", exp_name, "half.pt")

    full_path = "reward_shaping.p"
    full_env = construct_curriculum_env(2)
    full_model_path = os.path.join("models", exp_name, "full.pt")

    print("=" * 20)
    print("Start quarter curriculum learning")
    print("=" * 20)
    quarter_model = None
    while quarter_model is None:
        quarter_model = train(ConvDQN, quarter_env, pretrained=None, reward_shaping_p=quarter_path, input_t_max=13)
    ensure_path(quarter_model_path)
    save_model(quarter_model, quarter_model_path)
    print("Test quarter curriculum learning")
    test(quarter_model, quarter_env)

    print("=" * 20)
    print("Start half curriculum learning")
    print("=" * 20)
    half_model = None
    while half_model is None:
        half_model = train(ConvDQN, half_env, pretrained=quarter_model, reward_shaping_p=half_path, input_t_max=25)
    ensure_path(half_model_path)
    save_model(half_model, half_model_path)
    print("Test half curriculum learning")
    test(half_model, half_env)

    print("=" * 20)
    print("Start full curriculum learning")
    print("=" * 20)
    full_model = None
    while full_model is None:
        full_model = train(ConvDQN, full_env, pretrained=half_model, reward_shaping_p=full_path, input_t_max=50)
    ensure_path(full_model_path)
    save_model(full_model, full_model_path)
    print("Test full curriculum learning")
    test(full_model, half_env)

