import environment

world = environment.MDCVRP(2, 100, 8, 100, "train")
world.reset(2, 100, 8, 100)

world.execute(0, 0)