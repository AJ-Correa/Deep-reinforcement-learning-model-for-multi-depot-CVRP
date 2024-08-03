import random

def gen_uniform_instance(n_vehicles, n_customers, capacity, map_size):
    customers_coords = [(random.randint(0, map_size), random.randint(0, map_size)) for cust in range(n_customers)]
    demand = [random.randint(int((capacity * n_vehicles) / (n_customers * 2)), int((capacity * n_vehicles) / (n_customers / 2))) for cust in range(n_customers)]
    depots_coords = [(random.randint(0, map_size), random.randint(0, map_size)) for vec in range(n_vehicles)]

    return customers_coords, demand, depots_coords