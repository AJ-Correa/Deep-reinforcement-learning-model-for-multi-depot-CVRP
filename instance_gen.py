import random
import settings as s

def gen_uniform_instance(n_vehicles, n_customers, capacity, map_size):
    customers_coords = [(random.randint(0, map_size), random.randint(0, map_size)) for cust in range(n_customers)]
    demand = [random.randint(int((capacity * n_vehicles) / (n_customers * 2)), int((capacity * n_vehicles) / (n_customers / 2))) for cust in range(n_customers)]
    depots_coords = [(random.randint(0, map_size), random.randint(0, map_size)) for vec in range(n_vehicles)]

    return customers_coords, demand, depots_coords

def read_cvrp_benchmark_instance(instance_dir):
    file = open(instance_dir, "r")
    num_vehicles = int(file.readline().split()[-1][-1])

    for _ in range(2):
        file.readline()
    num_customers = int(file.readline().split()[2]) - 1

    file.readline()
    capacity = int(file.readline().split()[2])

    file.readline()
    depot_values = list(map(int, file.readline().split()))
    depot_coordinates = [(depot_values[1], depot_values[2]) for _ in range(num_vehicles)]

    customers_coordinates = []
    customer_demands = []

    for cust in range(num_customers):
        current_customer_values = list(map(int, file.readline().split()))
        customers_coordinates.append((current_customer_values[1], current_customer_values[2]))

    file.readline()
    file.readline()

    for cust in range(num_customers):
        current_demand_values = list(map(int, file.readline().split()))
        customer_demands.append(current_demand_values[1])

    return customers_coordinates, customer_demands, depot_coordinates, capacity, num_vehicles, num_customers

read_cvrp_benchmark_instance(s.INSTANCE_DIR)