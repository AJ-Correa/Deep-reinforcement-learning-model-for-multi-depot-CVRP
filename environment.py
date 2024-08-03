import simpy
import math
import random
import numpy as np
import copy
from instance_gen import gen_uniform_instance

class Solution:
    def __init__(self):
        self.results_pool = {
            'solutions': [],
            'objective': []
        }

    def add_pool(self, solution, objective):
        self.results_pool['solutions'].append(solution)
        self.results_pool['objective'].append(objective)

class MDCVRP:
    def __init__(self, n_vehicles, capacity, n_customers, map_size, mode):
        self.map_size = map_size
        self.num_vehicles = n_vehicles
        self.capacity = capacity
        self.num_customers = n_customers

        self.customers_coordinates, self.customers_demands, self.depots_coordinates = gen_uniform_instance(
            self.num_vehicles, self.num_customers, self.capacity, self.map_size)

        self.env = None
        self.solution = Solution()
        self.loss_log = None
        self.steps_log = None
        self.mode = mode

    def reset(self, n_vehicles, capacity, n_customers, map_size):
        self.map_size = map_size
        self.num_vehicles = n_vehicles
        self.capacity = capacity
        self.num_customers = n_customers

        self.customers_coordinates, self.customers_demands, self.depots_coordinates = gen_uniform_instance(
            self.num_vehicles, self.num_customers, self.capacity, self.map_size)

        self.env = simpy.Environment()

        self.vehicles = [Vehicle(self, vec, self.env) for vec in range(self.num_vehicles)]

    def execute(self, eps, num_steps):
        if self.mode == "train":
            print(f'*************************************************************************************')
            print(f'Episode: {eps} | Step Count: {num_steps}')

        self.env.run()

class Vehicle:
    def __init__(self, mdcvrp, vehicle_id, env):
        self.initial_capacity = mdcvrp.capacity
        self.capacity = mdcvrp.capacity
        self.num_vehicles = mdcvrp.num_vehicles
        self.num_customers = mdcvrp.num_customers
        self.customers_demands = copy.deepcopy(mdcvrp.customers_demands)
        self.customers_coordinates = mdcvrp.customers_coordinates
        self.vehicle_id = vehicle_id
        self.depots_coordinates = mdcvrp.depots_coordinates
        self.depot_location = self.depots_coordinates[self.vehicle_id]

        self.current_location = self.depot_location

        self.total_distance = 0
        self.depot_distance = 0
        self.total_delivered_load = 0

        self.state = []
        self.previous_state = []
        self.initial_state()
        self.reward = 0
        self.terminal = 0

        self.process = env.process(self.run(env, mdcvrp))

    def euclidean_distance(self, p1, p2):
        return math.dist(p1, p2)

    def initial_state(self):
        for cust in range(self.num_customers):
            self.state.append(self.euclidean_distance(self.current_location, self.customers_coordinates[cust])) # add distance between vehicle and customers

        self.state = self.state + self.customers_demands # add customer demands
        self.state.append(self.capacity) # register vehicle capacity
        self.state.append(self.depot_distance) # distance between vehicle and depot
        self.state.append(self.total_delivered_load) # total delivered load by vehicle
        self.state.append(self.total_distance) # total travelled distance

    def update_distances(self):
        for cust in range(self.num_customers):
            self.state[cust] = self.euclidean_distance(self.current_location, self.customers_coordinates[cust]) # add distance between vehicle and customers

    def run(self, env, mdcvrp):

        while not self.terminal:
            feasible_actions = [1 if (x > 0 and self.capacity >= x) else 0 for x in self.customers_demands]
            feasible_action_indexes = [i for i in range(len(feasible_actions)) if feasible_actions[i]]

            self.previous_state = copy.deepcopy(self.state)

            if len(feasible_action_indexes) == 0:
                distance = self.euclidean_distance(self.current_location, self.depot_location)

                yield env.timeout(distance)

                self.capacity = self.initial_capacity
                self.current_location = self.depot_location
                self.update_distances()
                self.state[self.num_customers * 2 + 1] = 0
                self.state[self.num_customers * 2 + 3] += distance

                self.reward = -distance
            else:
                action = random.choice(feasible_action_indexes)
                distance = self.euclidean_distance(self.current_location, self.customers_coordinates[action])

                demand_fulfilled = self.customers_demands[action] if self.capacity >= self.customers_demands[action] else self.customers_demands[action] - self.capacity

                for vec in mdcvrp.vehicles:
                    vec.customers_demands[action] -= demand_fulfilled

                yield env.timeout(distance)

                self.current_location = self.customers_coordinates[action]
                self.total_distance += distance
                self.depot_distance = self.euclidean_distance(self.current_location, self.depot_location)
                self.total_delivered_load += demand_fulfilled
                self.capacity -= demand_fulfilled

                self.update_distances()
                for vec in mdcvrp.vehicles:
                    vec.state[action + self.num_customers] -= demand_fulfilled
                self.state[self.num_customers * 2] = self.capacity - demand_fulfilled
                self.state[self.num_customers * 2 + 1] = self.euclidean_distance(self.current_location, self.depot_location)
                self.state[self.num_customers * 2 + 2] += demand_fulfilled
                self.state[self.num_customers * 2 + 3] += distance

                self.reward = -distance

            if sum(self.customers_demands) == 0:
                distance = self.euclidean_distance(self.current_location, self.depot_location)
                yield env.timeout(distance)

                self.total_distance += distance
                self.depot_distance = 0
                self.state[self.num_customers * 2 + 1] = 0
                self.state[self.num_customers * 2 + 3] += distance
                self.terminal = 1
                self.reward = -distance

        print(self.total_distance)
        print(self.vehicle_id)