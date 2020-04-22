import gym
from gym_grid_driving.envs.grid_driving import LaneSpec, Point


def construct_task2_env(tensor_state=True, conf=0):
    large_config = {'agent_speed_range': [-3, -1], 'width': 50,
              'lanes': [LaneSpec(cars=7, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -1]),
                        LaneSpec(cars=6, speed_range=[-1, -1]),
                        LaneSpec(cars=6, speed_range=[-3, -1]),
                        LaneSpec(cars=7, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -1]),
                        LaneSpec(cars=6, speed_range=[-3, -2]),
                        LaneSpec(cars=7, speed_range=[-1, -1]),
                        LaneSpec(cars=6, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -2])]
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
    medium_large_config_full = {'agent_speed_range': [-3, -1], 'width': 40,
                    'lanes': [LaneSpec(cars=6, speed_range=[-2, -1]),
                              LaneSpec(cars=7, speed_range=[-2, -1]),
                              LaneSpec(cars=5, speed_range=[-1, -1]),
                              LaneSpec(cars=5, speed_range=[-3, -1]),
                              LaneSpec(cars=6, speed_range=[-2, -1]),
                              LaneSpec(cars=7, speed_range=[-2, -1]),
                              ]
                    }
    medium_large_config_quarter = {'agent_speed_range': [-3, -1], 'width': 40, 'finish_position': Point(30,4),
                    'lanes': [LaneSpec(cars=6, speed_range=[-2, -1]),
                              LaneSpec(cars=7, speed_range=[-2, -1]),
                              LaneSpec(cars=5, speed_range=[-1, -1]),
                              LaneSpec(cars=5, speed_range=[-3, -1]),
                              LaneSpec(cars=6, speed_range=[-2, -1]),
                              LaneSpec(cars=7, speed_range=[-2, -1]),
                              ]
                    }
    medium_large_config_half = {'agent_speed_range': [-3, -1], 'width': 40,'finish_position': Point(20,2),
                    'lanes': [LaneSpec(cars=6, speed_range=[-2, -1]),
                              LaneSpec(cars=7, speed_range=[-2, -1]),
                              LaneSpec(cars=5, speed_range=[-1, -1]),
                              LaneSpec(cars=5, speed_range=[-3, -1]),
                              LaneSpec(cars=6, speed_range=[-2, -1]),
                              LaneSpec(cars=7, speed_range=[-2, -1]),
                              ]
                    }
    medium_large_config_threequarter = {'agent_speed_range': [-3, -1], 'width': 40, 'finish_position': Point(10,1),
                    'lanes': [LaneSpec(cars=6, speed_range=[-2, -1]),
                              LaneSpec(cars=7, speed_range=[-2, -1]),
                              LaneSpec(cars=5, speed_range=[-1, -1]),
                              LaneSpec(cars=5, speed_range=[-3, -1]),
                              LaneSpec(cars=6, speed_range=[-2, -1]),
                              LaneSpec(cars=7, speed_range=[-2, -1]),
                              ]
                    }
    large_config_half = {'agent_speed_range': [-3, -1], 'width': 50, 'finish_position': Point(24,4),
              'lanes': [LaneSpec(cars=7, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -1]),
                        LaneSpec(cars=6, speed_range=[-1, -1]),
                        LaneSpec(cars=6, speed_range=[-3, -1]),
                        LaneSpec(cars=7, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -1]),
                        LaneSpec(cars=6, speed_range=[-3, -2]),
                        LaneSpec(cars=7, speed_range=[-1, -1]),
                        LaneSpec(cars=6, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -2])]
            }
    large_config_quarter = {'agent_speed_range': [-3, -1], 'width': 50, 'finish_position': Point(36,6),
              'lanes': [LaneSpec(cars=7, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -1]),
                        LaneSpec(cars=6, speed_range=[-1, -1]),
                        LaneSpec(cars=6, speed_range=[-3, -1]),
                        LaneSpec(cars=7, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -1]),
                        LaneSpec(cars=6, speed_range=[-3, -2]),
                        LaneSpec(cars=7, speed_range=[-1, -1]),
                        LaneSpec(cars=6, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -2])]
            }
    large_config_43_6 = {'agent_speed_range': [-3, -1], 'width': 50, 'finish_position': Point(43,6),
              'lanes': [LaneSpec(cars=7, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -1]),
                        LaneSpec(cars=6, speed_range=[-1, -1]),
                        LaneSpec(cars=6, speed_range=[-3, -1]),
                        LaneSpec(cars=7, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -1]),
                        LaneSpec(cars=6, speed_range=[-3, -2]),
                        LaneSpec(cars=7, speed_range=[-1, -1]),
                        LaneSpec(cars=6, speed_range=[-2, -1]),
                        LaneSpec(cars=8, speed_range=[-2, -2])]
            }
    if conf == 0:
        config = medium_large_config_full
    elif conf == 1:
        config = medium_large_config_quarter
    elif conf == 2:
        config = medium_large_config_half
    elif conf == 3:
        config = medium_large_config_threequarter
    else:
        config = medium_large_config_threequarter
    if tensor_state:
        config['observation_type'] = 'tensor'
    return gym.make('GridDriving-v0', **config)
