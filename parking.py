from parking import run_test
from parking.reward_shaping import heuristic_reward
from parking.dagger import run_dagger
from parking.c_learn import run_c_learn
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Graph Learning')
    parser.add_argument('--rs', action="store_true", help="Running reward shaping")
    parser.add_argument('--dagger', action="store_true", help="Running DAGGER")
    parser.add_argument('--test-many', action="store_true", help="Test on many examples")
    parser.add_argument('--test-single', action="store_true", help="Test on a single example")
    parser.add_argument('--c-learn', action="store_true", help="Running Curriculum Learning")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.rs:
        print("Run reward shaping")
        heuristic_reward()
    if args.dagger:
        print("Run DAGGER")
        run_dagger()
    if args.test_many:
        print("Test many")
        run_test(mode="many")
    if args.test_single:
        print("Test single")
        run_test(mode="single")
    if args.c_learn:
        print("Curriculum Learning")
        run_c_learn()
