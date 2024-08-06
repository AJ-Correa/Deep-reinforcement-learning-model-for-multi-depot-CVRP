import environment
import model
import time
import settings as s

def learn():
    st = time.time()

    world = environment.MDCVRP("train")

    agents = model.Agents(world)

    for epoch in range(s.NUM_EPOCHS):
        world.reset(agents)

        world.execute(epoch)

        agents.epoch_id += 1

        if epoch % s.TARGET_UPDATE == 0:
            agents.customer_target_net.load_state_dict(agents.customer_policy_net.state_dict())  # target network update
            agents.vehicle_target_net.load_state_dict(agents.vehicle_policy_net.state_dict())  # target network update

        if s.USE_RAINBOW:
            fraction = min(epoch / s.NUM_EPOCHS, 1.0)
            agents.beta = agents.beta + fraction * (1.0 - agents.beta)

    et = time.time()

    print("Total computational time: ", et - st, " seconds")

    return world.solution.results_pool, world.solution.map_pool, agents.loss_log, world

if __name__ == '__main__':
    learn()