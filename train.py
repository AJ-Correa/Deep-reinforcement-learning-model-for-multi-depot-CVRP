import environment
import model
import settings as s

def learn():
    world = environment.MDCVRP("train")

    agents = model.Agents(world)

    for epoch in range(s.NUM_EPOCHS):
        world.reset(agents)

        world.execute(epoch)

        agents.epoch_id += 1

        if epoch % s.TARGET_UPDATE == 0:
            agents.target_net.load_state_dict(agents.policy_net.state_dict())  # target network update

    return world.solution.results_pool, agents.loss_log