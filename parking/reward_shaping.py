from parking.env import construct_task2_env
import gym
import gym_grid_driving
from gym_grid_driving.envs.grid_driving import LaneSpec, Point
import os
import re
from utils import *
import numpy as np

FAST_DOWNWARD_DIRECTORY_ABSOLUTE_PATH = "/fast_downward/"
PDDL_FILE_ABSOLUTE_PATH = ""


class GeneratePDDL_Stationary:
    '''
    Class to generate the PDDL files given the environment description.
    '''

    def __init__(self, env, num_lanes, width, file_name):
        self.state = env.reset()
        self.num_lanes = num_lanes
        self.width = width
        self.file_name = file_name
        self.problem_file_name = self.file_name + 'problem.pddl'
        self.domain_file_name = self.file_name + 'domain.pddl'
        self.domain_string = ""
        self.type_string = ""
        self.predicate_strings = self.addHeader("predicates")
        self.action_strings = ""
        self.problem_string = ""
        self.object_strings = self.addHeader("objects")

    def addDomainHeader(self, name='default_header'):
        '''
        Adds the header in the domain file.

        Parameters :
        name (string): domain name.
        '''
        self.domain_header = "(define (domain " + name + " ) \n" + "(:requirements :strips :typing) \n"

    def addTypes(self, types={}):
        '''
        Adds the object types to the PDDL domain file.

        Parameters :
        types (dict): contains a dictionary of (k,v) pairs, where k is the object type, and v is the supertype. If k has no supertype, v is None.
        '''
        type_string = "(:types "

        for _type, _supertype in types.items():
            if _supertype is None:
                type_string += _type + "\n"
            else:
                type_string += _type + " - " + _supertype + "\n"
        type_string += ") \n"
        self.type_string = type_string

    def addPredicate(self, name='default_predicate', parameters=(), isLastPredicate=False):
        '''
        Adds predicates to the PDDL domain file

        Parameters :
        name (string) : name of the predicate.
        parameters (tuple or list): contains a list of (var_name, var_type) pairs, where var_name is an instance of object type var_type.
        isLastPredicate (bool) : True for the last predicate added.
        '''
        predicate_string = "(" + name
        for var_name, var_type in parameters:
            predicate_string += " ?" + var_name + " - " + var_type
        predicate_string += ") \n"
        self.predicate_strings += predicate_string

        if isLastPredicate:
            self.predicate_strings += self.addFooter()

    def addAction(self, name='default_action', parameters=(), precondition_string="", effect_string=""):
        '''
        Adds actions to the PDDL domain file

        Parameters :
        name (string) : name of the action.
        parameters (tuple or list): contains a list of (var_name, var_type) pairs, where var_name is an instance of object type var_type.
        precondition_string (string) : The precondition for the action.
        effect_string (string) : The effect of the action.
        '''
        action_string = name + "\n"
        parameter_string = ":parameters ("
        for var_name, var_type in parameters:
            parameter_string += " ?" + var_name + " - " + var_type
        parameter_string += ") \n"

        precondition_string = ":precondition " + precondition_string + "\n"
        effect_string = ":effect " + effect_string + "\n"
        action_string += parameter_string + precondition_string + effect_string
        action_string = self.addHeader("action") + action_string + self.addFooter()
        self.action_strings += action_string

    def generateDomainPDDL(self):
        '''
        Generates the PDDL domain file after all actions, predicates and types are added
        '''
        domain_file = open(PDDL_FILE_ABSOLUTE_PATH + self.domain_file_name, "w")
        PDDL_String = self.domain_header + self.type_string + self.predicate_strings + self.action_strings + self.addFooter()
        domain_file.write(PDDL_String)
        domain_file.close()

    def addProblemHeader(self, problem_name='default_problem_name', domain_name='default_domain_name'):
        '''
        Adds the header in the problem file.

        Parameters :
        problem_name (string): problem name.
        domain_name (string): domain name.
        '''
        self.problem_header = "(define (problem " + problem_name + ") \n (:domain " + domain_name + ") \n"

    def addObjects(self, obj_type, obj_list=[], isLastObject=False):
        '''
        Adds object instances of the same type to the problem file

        Parameters :
        obj_type (string) : the object type of the instances that are being added
        obj_list (list(str)) : a list of object instances to be added
        isLastObject (bool) : True for the last set of objects added.
        '''
        obj_string = ""
        for obj in obj_list:
            obj_string += obj + " "
        obj_string += " - " + obj_type
        self.object_strings += obj_string + "\n "
        if isLastObject:
            self.object_strings += self.addFooter()

    def addInitState(self):
        '''
        Generates the complete init state
        '''
        initString = self.generateInitString()
        self.initString = self.addHeader("init") + initString + self.addFooter()

    def addGoalState(self):
        '''
        Generates the complete goal state
        '''
        goalString = self.generateGoalString()
        self.goalString = self.addHeader("goal") + goalString + self.addFooter()

    def generateGridCells(self):
        '''
        Generates the grid cell objects.

        For a |X| x |Y| sized grid, |X| x |Y| objects to represent each grid cell are created.
        pt0pt0, pt1pt0, .... ptxpt0
        pt0pt1, pt1pt1, .... ptxpt1
        ..       ..            ..
        ..       ..            ..
        pt0pty, pt1pty, .... ptxpty


        '''
        self.grid_cell_list = []
        for w in range(self.width):
            for lane in range(self.num_lanes):
                self.grid_cell_list.append("pt{}pt{}".format(w, lane))

    def _generate_move_next(self, cars, time_step):
        blocked_positions = []
        for car in cars:
            car_x, car_y, car_id, car_speed = car["x"], car["y"], car["id"], car["speed"]
            for step in range(min(car_speed, 1), max(car_speed + 1, 0)):
                next_car_x, next_car_y = car_x + step, car_y
                next_car_x = (next_car_x + self.width) if next_car_x < 0 else next_car_x
                blocked_positions.append("pt{}pt{}ts{}".format(next_car_x, next_car_y, time_step + 1))

        cell_re = "^pt(\\d+)pt(\\d+)$"
        width, n_lanes = self.width, self.num_lanes
        move_next_preds = []

        for cell in self.grid_cell_list:
            pos = re.search(cell_re, cell)
            x_pos, y_pos = int(pos.group(1)), int(pos.group(2))
            current_pos = "pt{}pt{}ts{}".format(x_pos, y_pos, time_step)
            if x_pos > 0:
                # generate up_next predicate based on negative speed assumption
                # simply move 1 step forward, do not go up
                if y_pos == 0:
                    up_next_x_pos, up_next_y_pos = x_pos - 1, y_pos
                # move forward and go up
                else:
                    up_next_x_pos, up_next_y_pos = x_pos - 1, y_pos - 1
                up_next_pos = "pt{}pt{}ts{}".format(up_next_x_pos, up_next_y_pos, time_step + 1)
                if up_next_pos not in blocked_positions:
                    up_next_pred = ["up_next", current_pos, up_next_pos]
                    move_next_preds.append(up_next_pred)

                # generate down_next predicate
                # simply move 1 step forward, do not go down
                if y_pos == n_lanes - 1:
                    down_next_x_pos, down_next_y_pos = x_pos - 1, y_pos
                # move forward and go down
                else:
                    down_next_x_pos, down_next_y_pos = x_pos - 1, y_pos + 1
                down_next_pos = "pt{}pt{}ts{}".format(down_next_x_pos, down_next_y_pos, time_step + 1)
                if down_next_pos not in blocked_positions:
                    down_next_pred = ["down_next", current_pos, down_next_pos]
                    move_next_preds.append(down_next_pred)

                # generate forward_next predicate
                forward_next_preds = []
                for speed in range(self.state.agent.speed_range[0], self.state.agent.speed_range[1] + 1, 1):
                    is_blocked = False
                    # assume that speed is negative, check if all the steps in speed are not blocked
                    for step in range(speed, 0):
                        step_next_x_pos, step_next_y_pos = x_pos + step, y_pos
                        step_next_pos = "pt{}pt{}ts{}".format(step_next_x_pos, y_pos, time_step + 1)
                        is_blocked = step_next_pos in blocked_positions
                        if is_blocked:
                            break
                    if is_blocked:
                        continue

                    forward_next_x_pos, forward_next_y_pos = x_pos + speed, y_pos
                    if forward_next_x_pos >= 0:
                        forward_next_pred = ["forward_next", current_pos,
                                             "pt{}pt{}ts{}".format(forward_next_x_pos, forward_next_y_pos,
                                                                   time_step + 1)]
                        forward_next_preds.append(forward_next_pred)

                move_next_preds.extend(forward_next_preds)
        move_next_strings = ["(" + " ".join(x) + ")" for x in move_next_preds]
        return " ".join(move_next_strings)

    def _generate_agent_preds(self):
        agent_x, agent_y = self.state.agent.position.x, self.state.agent.position.y
        agent_name = "agent1"
        agent_at_strings = "(" + " ".join(["at", "pt{}pt{}ts0".format(agent_x, agent_y), agent_name]) + ")"
        return agent_at_strings

    def generateInitString(self):
        '''
        FILL ME : Should return the init string in the problem PDDL file.
        Hint : Use the defined grid cell objects from genearateGridCells and predicates to construct the init string.

        Information that might be useful here :

        1. Initial State of the environment : self.state
        2. Agent's x position : self.state.agent.position.x
        3. Agent's y position : self.state.agent.position.y
        4. The object of type agent is called "agent1" (see generateProblemPDDLFile() ).
        5. Set of cars in the grid: self.state.cars
        6. For a car in self.state.cars, it's x position: car.position.x
        7. For a car in self.state.cars, it's y position: car.position.y
        8. List of grid cell objects : self.grid_cell_list
        9. Width of the grid: self.width
        10. Number of lanes in the grid : self.num_lanes

        Play with environment (https://github.com/cs4246/gym-grid-driving) to see the type of values above objects return

        Example: The following statement adds the initial condition string from https://github.com/pellierd/pddl4j/blob/master/pddl/logistics/p01.pddl

        return "(at apn1 apt2) (at tru1 pos1) (at obj11 pos1) (at obj12 pos1) (at obj13 pos1) (at tru2 pos2) (at obj21 pos2) (at obj22 pos2)
                (at obj23 pos2) (in-city pos1 cit1) (in-city apt1 cit1) (in-city pos2 cit2) (in-city apt2 cit2)"
        '''
        current_cars = []
        for car in self.state.cars:
            car_x, car_y, car_id, speed_min, speed_max = car.position.x, car.position.y, car.id, \
                                                         car.speed_range[0], car.speed_range[1]
            assert speed_min == speed_max, "Car speed is expected to be a constant"
            current_cars.append({"x": car_x, "y": car_y, "speed": speed_max, "id": car_id})
        all_move_next_strings = []
        for time_step in range(self.width - 1):
            time_step_move_next_string = self._generate_move_next(current_cars, time_step)
            all_move_next_strings.append(time_step_move_next_string)
            current_cars = [{"x": ((car["x"] + car["speed"]) % self.width), "y": car["y"], "speed": car["speed"],
                             "id": car["id"]}
                            for car in current_cars]
        final_move_next_string = " ".join(all_move_next_strings)
        agent_at_string = self._generate_agent_preds()
        init_string = " ".join([final_move_next_string, agent_at_string])
        return init_string

    def generateGoalString(self):
        '''
        FILL ME : Should return the goal string in the problem PDDL file
        Hint : Use the defined grid cell objects from genearateGridCells and predicates to construct the goal string.

        Information that might be useful here :
        1. Goal x Position : self.state.finish_position.x
        2. Goal y Position : self.state.finish_position.y
        3. The object of type agent is called "agent1" (see generateProblemPDDLFile() ).
        Play with environment (https://github.com/cs4246/gym-grid-driving) to see the type of values above objects return

        Example: The following statement adds goal string from https://github.com/pellierd/pddl4j/blob/master/pddl/logistics/p01.pddl

        return "(and (at obj11 apt1) (at obj23 pos1) (at obj13 apt1) (at obj21 pos1)))"
        '''
        finish_x, finish_y = self.state.finish_position.x, self.state.finish_position.y
        agent_name = "agent1"
        finish_positions = ["pt{}pt{}ts{}".format(finish_x, finish_y, ts) for ts in range(self.width)]
        conjuctions = " ".join(["(" + " ".join(["at", x, agent_name]) + ")" for x in finish_positions])
        goal_string = "(or " + conjuctions + ")"
        return goal_string

    def generateProblemPDDL(self):
        '''
        Generates the PDDL problem file after the object instances, init state and goal state are added
        '''
        problem_file = open(PDDL_FILE_ABSOLUTE_PATH + self.problem_file_name, "w")
        PDDL_String = self.problem_header + self.object_strings + self.initString + self.goalString + self.addFooter()
        problem_file.write(PDDL_String)
        problem_file.close()

    '''
    Helper Functions
    '''

    def addHeader(self, name):
        return "(:" + name + " "

    def addFooter(self):
        return ") \n"


def initializeSystem(env):
    gen = GeneratePDDL_Stationary(env, len(env.lanes), width=env.width, file_name='HW1')
    return gen


def generateDomainPDDLFile(gen):
    '''
    Function that specifies the domain and generates the PDDL Domain File.
    As a part of the assignemnt, you will need to add the actions here.
    '''
    gen.addDomainHeader("grid_world")
    gen.addTypes(types={"car": None, "agent": "car", "gridcell": None})
    '''
    Predicate Definitions :
    at(pt, car) : car is at gridcell pt.
    up_next(pt1, pt2) : pt2 is the next location of the car when it takes the UP action from pt1
    down_next(pt1, pt2) : pt2 is the next location of the car when it takes the DOWN action from pt1
    forward_next(pt1, pt2) : pt2 is the next location of the car when it takes the FORWARD action from pt1
    blocked(pt) : The gridcell pt is occupied by a car and is "blocked".
    '''
    gen.addPredicate(name="at", parameters=(("pt1", "gridcell"), ("car", "car")))
    gen.addPredicate(name="up_next", parameters=(("pt1", "gridcell"), ("pt2", "gridcell")))
    gen.addPredicate(name="down_next", parameters=(("pt1", "gridcell"), ("pt2", "gridcell")))
    gen.addPredicate(name="forward_next", parameters=(("pt1", "gridcell"), ("pt2", "gridcell")))
    gen.addPredicate(name="blocked", parameters=[("pt1", "gridcell")], isLastPredicate=True)

    # generate UP action
    up_parameters = (("pt1", "gridcell"), ("pt2", "gridcell"))
    up_precondition_string = "(and (at ?pt1 agent1) (up_next ?pt1 ?pt2))"
    up_effect_string = "(and (not (at ?pt1 agent1)) (at ?pt2 agent1))"

    gen.addAction(name="UP",
                  parameters=up_parameters,
                  precondition_string=up_precondition_string,
                  effect_string=up_effect_string)

    # generate DOWN action
    down_parameters = (("pt1", "gridcell"), ("pt2", "gridcell"))
    down_precondition_string = "(and (at ?pt1 agent1) (down_next ?pt1 ?pt2))"
    down_effect_string = "(and (not (at ?pt1 agent1)) (at ?pt2 agent1))"

    gen.addAction(name="DOWN",
                  parameters=down_parameters,
                  precondition_string=down_precondition_string,
                  effect_string=down_effect_string)

    # generate FORWARD action
    forward_parameters = (("pt1", "gridcell"), ("pt2", "gridcell"))
    forward_precondition_string = "(and (at ?pt1 agent1) (forward_next ?pt1 ?pt2))"
    forward_effect_string = "(and (not (at ?pt1 agent1)) (at ?pt2 agent1))"

    gen.addAction(name="FORWARD",
                  parameters=forward_parameters,
                  precondition_string=forward_precondition_string,
                  effect_string=forward_effect_string)
    gen.generateDomainPDDL()

    '''
    FILL ME : Add the actions UP, DOWN, FORWARD with the help of gen.addAction() as follows :

        gen.addAction(name="UP", parameters = (...), precondition_string = "...", effect_string="...")
        gen.addAction(name="DOWN", parameters = (...), precondition_string = "...", effect_string="...")
        gen.addAction(name="FORWARD", parameters = (...), precondition_string = "...", effect_string="...")

        You have to fill up the ... in each of gen.addAction() above.

    Example :

    The following statement adds the LOAD-TRUCK action from https://tinyurl.com/y3jocxdu [The domain file referenced in the assignment] to the domain file
    gen.addAction(name="LOAD-TRUCK",
                  parameters=(("pkg", "package"), ("truck" , "truck"), ("loc", "place")),
                  precondition_string="(and (at ?truck ?loc) (at ?pkg ?loc))",
                  effect_string= "(and (not (at ?pkg ?loc)) (in ?pkg ?truck))")
    '''
    pass


def generateProblemPDDLFile(gen):
    '''
    Function that specifies the domain and generates the PDDL Domain File.
    Objects defined here should be used to construct the init and goal strings
    '''
    gen.addProblemHeader("parking", "grid_world")
    gen.addObjects("agent", ["agent1"])
    gen.generateGridCells()
    grid_cell_objects = []
    for ts in range(gen.width):
        grid_cell_objects.extend([grid_cell + "ts{}".format(ts) for grid_cell in gen.grid_cell_list])
    gen.addObjects("gridcell", grid_cell_objects, isLastObject=True)
    gen.addInitState()
    gen.addGoalState()
    gen.generateProblemPDDL()
    pass


def runPDDLSolver(gen):
    '''
    Runs the fast downward solver to get the optimal plan
    '''
    os.system(
        FAST_DOWNWARD_DIRECTORY_ABSOLUTE_PATH + 'fast-downward.py ' + PDDL_FILE_ABSOLUTE_PATH + gen.domain_file_name + ' ' + PDDL_FILE_ABSOLUTE_PATH + gen.problem_file_name + ' --search  \"lazy_greedy([ff()], preferred=[ff()])\"' + ' > temp ')


def delete_files(gen):
    '''
    Deletes PDDL and plan files created.
    '''
    os.remove(PDDL_FILE_ABSOLUTE_PATH + gen.domain_file_name)
    os.remove(PDDL_FILE_ABSOLUTE_PATH + gen.problem_file_name)
    os.remove('sas_plan')


def parse_sas_plan(pos_memo, start_x, start_y):
    if not os.path.exists("sas_plan"):
        print("Cannot find plan for start x: {} start y: {}".format(start_x, start_y))
        pos_memo[start_x][start_y] = -1
        return
    sas_plan = load_text_as_list("sas_plan")
    cost_line = sas_plan[-1]
    cost_re = "; cost = (\\d+) \\(unit cost\\)"
    matched_cost = re.search(cost_re, cost_line)
    cost = int(matched_cost.group(1))

    action_re = "pt(\\d+)pt(\\d+)ts(\\d+)$"
    for i, action_line in enumerate(sas_plan[:-1]):
        tokens = action_line.split()
        pos_from = re.search(action_re, tokens[1])
        x_pos_from, y_pos_from = int(pos_from.group(1)), int(pos_from.group(2))
        pos_memo[x_pos_from][y_pos_from] = cost-i


def heuristic_reward():
    # parse_sas_plan(pos_memo)
    task2_env = construct_task2_env(conf=5)
    n_lanes, n_width, agent_speed_range = len(task2_env.lanes), task2_env.width, task2_env.agent_speed_range
    pos_memo = [[0 if (x == 10 and y == 1) else None for y in range(n_lanes)] for x in range(n_width)]

    lanes = [LaneSpec(0, [0, 0])] * n_lanes

    for start_x in list(range(n_width))[::-1]:
        for start_y in list(range(n_lanes))[::-1]:
            if pos_memo[start_x][start_y] is None:
                print("Start x: {} start y: {}".format(start_x, start_y))
                if start_x > 10:
                    env = gym.make('GridDriving-v0', lanes=lanes, width=n_width,
                                   random_seed=42, agent_speed_range=(-3, -1), finish_position=Point(10,1), agent_pos_init=Point(x=start_x, y=start_y))
                    gen = initializeSystem(env)
                    generateDomainPDDLFile(gen)
                    generateProblemPDDLFile(gen)
                    runPDDLSolver(gen)
                    parse_sas_plan(pos_memo, start_x, start_y)
                else:
                    pos_memo[start_x][start_y] = -1
                print("pos_memo:")
                print(pos_memo)

    pos_inf_reward, neg_inf_reward = 10, -10

    for x in range(n_width):
        for y in range(n_lanes):
            if pos_memo[x][y] == 0:
                pos_memo[x][y] = pos_inf_reward
            elif pos_memo[x][y] == -1:
                pos_memo[x][y] = neg_inf_reward
            else:
                pos_memo[x][y] = 1. / pos_memo[x][y]
    print("Final reward matrix")
    pos_memo = np.array(pos_memo)
    print(pos_memo)
    save_to_pickle(pos_memo, "reward_shaping_threequarter.p")
