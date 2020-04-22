import gym
from gym_grid_driving.envs.grid_driving import LaneSpec, Point

default_lanes = [LaneSpec(cars=7, speed_range=[-2, -1]),
                 LaneSpec(cars=8, speed_range=[-2, -1]),
                 LaneSpec(cars=6, speed_range=[-1, -1]),
                 LaneSpec(cars=6, speed_range=[-3, -1]),
                 LaneSpec(cars=7, speed_range=[-2, -1]),
                 LaneSpec(cars=8, speed_range=[-2, -1]),
                 LaneSpec(cars=6, speed_range=[-3, -2]),
                 LaneSpec(cars=7, speed_range=[-1, -1]),
                 LaneSpec(cars=6, speed_range=[-2, -1]),
                 LaneSpec(cars=8, speed_range=[-2, -2])]
default_speed_range = [-3, -1]
default_width = 50


def construct_curriculum_env(curr_id, tensor_state=True):
    quarter_config = {'agent_speed_range': default_speed_range, 'width': default_width,
                      'lanes': default_lanes,
                      'finish_position': Point(36, 6)
                      }
    half_config = {'agent_speed_range': default_speed_range, 'width': default_width,
                   'lanes': default_lanes,
                   'finish_position': Point(24, 4)
                  }
    full_config = {'agent_speed_range': default_speed_range, 'width': default_width,
                   'lanes': default_lanes,
                   'finish_position': Point(0, 0)
                   }
    if curr_id == 0:
        config = quarter_config
    elif curr_id == 1:
        config = half_config
    elif curr_id == 2:
        config = full_config
    else:
        raise ValueError("No curriculum of ID: {}".format(curr_id))
    if tensor_state:
        config['observation_type'] = 'tensor'
    return gym.make('GridDriving-v0', **config)


def construct_task2_env(tensor_state=True):
    large_config = {'agent_speed_range': [-3, -1], 'width': 50,
                    'lanes': default_lanes
                    }
    small_config = {'observation_type': 'tensor', 'agent_speed_range': [-2, -1], 'stochasticity': 0.0, 'width': 10,
              'lanes': [
                  LaneSpec(cars=3, speed_range=[-2, -1]),
                  LaneSpec(cars=4, speed_range=[-2, -1]),
                  LaneSpec(cars=2, speed_range=[-1, -1]),
                  LaneSpec(cars=2, speed_range=[-3, -1])
              ]}
    medium_config = {'observation_type': 'tensor', 'agent_speed_range': [-3, -1], 'width': 15,
              'lanes': [
                  LaneSpec(cars=3, speed_range=[-2, -1]),
                  LaneSpec(cars=4, speed_range=[-2, -1]),
                  LaneSpec(cars=2, speed_range=[-1, -1]),
                  LaneSpec(cars=2, speed_range=[-3, -1]),
                  LaneSpec(cars=3, speed_range=[-2, -1]),
                  LaneSpec(cars=4, speed_range=[-2, -1])
              ]}
    medium_large_config = {'agent_speed_range': [-3, -1], 'width': 40,
                    'lanes': [LaneSpec(cars=6, speed_range=[-2, -1]),
                              LaneSpec(cars=7, speed_range=[-2, -1]),
                              LaneSpec(cars=5, speed_range=[-1, -1]),
                              LaneSpec(cars=5, speed_range=[-3, -1]),
                              LaneSpec(cars=6, speed_range=[-2, -1]),
                              LaneSpec(cars=7, speed_range=[-2, -1]),
                              ]
                    }
    curri_large_config = {'agent_speed_range': [-3, -1], 'width': 50,
                    'lanes': default_lanes,
                    'finish_position': Point(43, 6)
                    }
    config = curri_large_config
    if tensor_state:
        config['observation_type'] = 'tensor'
    return gym.make('GridDriving-v0', **config)