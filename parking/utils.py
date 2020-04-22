from datetime import datetime


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