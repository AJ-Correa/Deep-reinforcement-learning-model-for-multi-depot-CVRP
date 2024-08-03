import environment

world = environment.MDCVRP(1, 100, 8, 100, "train")
world.reset(1, 100, 8, 100)

world.execute(0, 0)