import simpy
import math
import random
import numpy as np
import copy
import settings as s
from instance_gen import gen_uniform_instance, read_cvrp_benchmark_instance
import torch

class Solution:
    def __init__(self):
        self.results_pool = []
        self.map_pool = []

    def add_pool(self, agent, solution):
        self.map_pool.append(solution)
        total_distance = 0

        # for route in range(len(solution)):
        #     for node in range(len(solution[route]) - 1):
        #         if solution[route][node] == agent.num_customers and solution[route][node + 1] != agent.num_customers:
        #             total_distance += math.dist(agent.depot_location, agent.customers_coordinates[solution[route][node + 1]])
        #         elif solution[route][node] != agent.num_customers and solution[route][node + 1] != agent.num_customers:
        #             total_distance += math.dist(agent.customers_coordinates[solution[route][node]], agent.customers_coordinates[solution[route][node + 1]])
        #         elif solution[route][node] == agent.num_customers and solution[route][node + 1] == agent.num_customers:
        #             pass
        #         else:
        #             total_distance += math.dist(agent.depot_location, agent.customers_coordinates[solution[route][node]])

        self.results_pool.append(agent.total_distance)

class MDCVRP:
    def __init__(self, mode):
        # self.customers_coordinates, self.customers_demands, self.depots_coordinates = gen_uniform_instance(
        #     self.num_vehicles, self.num_customers, self.capacity, self.map_size)
        self.customers_coordinates, self.customers_demands, self.depots_coordinates, self.capacity, self.num_vehicles, self.num_customers = read_cvrp_benchmark_instance(s.INSTANCE_DIR)

        self.env = None
        self.solution = Solution()
        self.loss_log = None
        self.mode = mode

    def reset(self, agents):
        # self.customers_coordinates, self.customers_demands, self.depots_coordinates = gen_uniform_instance(
        #     self.num_vehicles, self.num_customers, self.capacity, self.map_size)
        self.customers_coordinates, self.customers_demands, self.depots_coordinates, self.capacity, self.num_vehicles, self.num_customers = read_cvrp_benchmark_instance(s.INSTANCE_DIR)

        self.env = simpy.Environment()

        self.agent = Agent(self, self.env, agents)

    def execute(self, eps):
        if self.mode == "train":
            print(f'*************************************************************************************')
            print(f'Episode: {eps}')

        self.env.run()
        self.solution.add_pool(self.agent, self.agent.solution)

class Agent:
    def __init__(self, mdcvrp, env, agents):
        self.num_vehicles = mdcvrp.num_vehicles
        self.initial_capacity = mdcvrp.capacity
        self.capacities = [mdcvrp.capacity for _ in range(self.num_vehicles)]
        self.num_customers = mdcvrp.num_customers
        self.customers_demands = copy.deepcopy(mdcvrp.customers_demands)
        self.customers_coordinates = mdcvrp.customers_coordinates
        self.depots_coordinates = mdcvrp.depots_coordinates

        self.current_locations = [self.depots_coordinates[i] for i in range(self.num_vehicles)]

        self.total_distances = [0 for _ in range(self.num_vehicles)]
        self.depot_distance = [0 for _ in range(self.num_vehicles)]
        self.total_delivered_load = [0 for _ in range(self.num_vehicles)]

        self.customer_state = [[] for _ in range(self.num_customers)]
        self.previous_customer_state = copy.deepcopy(self.customer_state)
        self.customer_reward = 0
        self.customer_action = 0
        self.edge_indexes = [[], []]
        self.previous_edge_indexes = copy.deepcopy(self.edge_indexes)

        self.vehicle_state = [[] for _ in range(self.num_vehicles)]
        self.previous_vehicle_state = copy.deepcopy(self.vehicle_state)
        self.vehicle_reward = 0
        self.vehicle_action = 0

        self.terminal = 0

        self.initial_state()
        self.total_distance = 0

        self.customer_transition = list()
        self.vehicle_transition = list()
        self.solution = [[self.num_customers] for _ in range(self.num_vehicles)]

        self.process = env.process(self.run(env, mdcvrp, agents))

    def euclidean_distance(self, p1, p2):
        return math.dist(p1, p2)

    def initial_state(self):
        for customer in range(self.num_customers):
            self.customer_state[customer].append(self.customers_coordinates[customer][0])
            self.customer_state[customer].append(self.customers_coordinates[customer][1])
            self.customer_state[customer].append(0)

        for vehicle in range(self.num_vehicles):
            self.vehicle_state[vehicle].append(0)
            self.vehicle_state[vehicle].append(1)

    def run(self, env, mdcvrp, agents):

        while not self.terminal:

            feasible_customer_actions = [1 if (x > 0) else 0 for x in self.customers_demands]
            feasible_customer_action_indexes = [i for i in range(len(feasible_customer_actions)) if feasible_customer_actions[i]]

            self.previous_customer_state = copy.deepcopy(self.customer_state)
            # self.previous_vehicle_state = copy.deepcopy(self.vehicle_state)
            self.previous_edge_indexes = copy.deepcopy(self.edge_indexes)

            self.customer_action = agents.transition(self.customer_state, feasible_customer_action_indexes, torch.tensor(self.edge_indexes, dtype=torch.long), "customer")

            feasible_vehicles_actions_indexes = [i for i in range(self.num_vehicles)]

            for vehicle in range(self.num_vehicles):
                if self.capacities[vehicle] >= self.customers_demands[self.customer_action]:
                    self.vehicle_state[vehicle][0] = self.euclidean_distance(self.customers_coordinates[self.customer_action], self.current_locations[vehicle])
                else:
                    self.vehicle_state[vehicle][0] = self.euclidean_distance(self.depots_coordinates[vehicle], self.current_locations[vehicle]) + self.euclidean_distance(self.depots_coordinates[vehicle], self.customers_coordinates[self.customer_action])
                self.vehicle_state[vehicle][1] = 1 if self.capacities[vehicle] >= self.customers_demands[self.customer_action] else 0
            self.previous_vehicle_state = copy.deepcopy(self.vehicle_state)
            self.vehicle_action = agents.transition(self.vehicle_state, feasible_vehicles_actions_indexes, None, "vehicle")

            vehicle_coordinates = (self.current_locations[self.vehicle_action][0], self.current_locations[self.vehicle_action][1])

            if self.capacities[self.vehicle_action] >= self.customers_demands[self.customer_action]:
                distance = self.euclidean_distance(vehicle_coordinates, self.customers_coordinates[self.customer_action])
            else:
                distance = self.euclidean_distance(vehicle_coordinates, self.depots_coordinates[self.vehicle_action]) + self.euclidean_distance(self.depots_coordinates[self.vehicle_action], self.customers_coordinates[self.customer_action])
            self.total_distance += distance

            demand_fulfilled = self.customers_demands[self.customer_action]

            yield env.timeout(distance)

            self.current_locations[self.vehicle_action] = self.customers_coordinates[self.customer_action]

            if len(self.solution[self.vehicle_action]) == 1:
                self.edge_indexes[0].append(self.customer_action)
                self.edge_indexes[1].append(self.customer_action)
            else:
                self.edge_indexes[0].append(self.solution[self.vehicle_action][-1])
                self.edge_indexes[1].append(self.customer_action)

            if self.capacities[self.vehicle_action] >= self.customers_demands[self.customer_action]:
                self.solution[self.vehicle_action].append(self.customer_action)
            else:
                self.solution[self.vehicle_action].append(self.num_customers)
                self.solution[self.vehicle_action].append(self.customer_action)

            self.customer_state[self.customer_action][2] = 1
            self.vehicle_state[self.vehicle_action][0] = 0

            self.customers_demands[self.customer_action] -= demand_fulfilled
            self.capacities[self.vehicle_action] -= demand_fulfilled

            self.customer_reward = -distance
            self.vehicle_reward = -distance

            if sum(self.customers_demands) == 0:
                self.terminal = 1
                for i in range(self.num_vehicles):
                    self.solution[i].append(self.num_customers)
                    self.total_distance += self.euclidean_distance(self.current_locations[i], self.depots_coordinates[i])

            self.customer_transition = [self.previous_customer_state, self.previous_edge_indexes, self.customer_action, self.customer_reward, self.customer_state, self.edge_indexes, self.terminal]
            self.vehicle_transition = [self.previous_vehicle_state, self.vehicle_action, self.vehicle_reward, self.vehicle_state, self.terminal]

            if s.USE_RAINBOW == 0:
                agents.customer_memory.store(*self.customer_transition)
                agents.vehicle_memory.store(*self.vehicle_transition)
            # else:
            #     if s.N_STEP > 1:
            #         one_step_transition = agents.memory_n.store(*self.transition)
            #     else:
            #         one_step_transition = self.transition
            #
            #     if one_step_transition:
            #         agents.memory.store(*one_step_transition)

            agents.optimize_model()