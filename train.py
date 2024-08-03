import environment
import model
import rainbow_model
import settings as s

def learn():
    world = environment.MDCVRP("train")

    if s.USE_RAINBOW == 0:
        agents = model.Agents(world)
    else:
        agents = rainbow_model.Agents(world)

    for epoch in range(s.NUM_EPOCHS):
        world.reset(agents)

        world.execute(epoch)

        agents.epoch_id += 1

        if epoch % s.TARGET_UPDATE == 0:
            agents.target_net.load_state_dict(agents.policy_net.state_dict())  # target network update

        if s.USE_RAINBOW:
            fraction = min(epoch / s.NUM_EPOCHS, 1.0)
            agents.beta = agents.beta + fraction * (1.0 - agents.beta)

    return world.solution.results_pool, agents.loss_log

learn()