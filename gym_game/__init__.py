import gymnasium as gym

gym.register(
    id="GridWorld-v0",
    entry_point="gym_game.envs:GridWorldEnv",
)

gym.register(
    id="Pathfinder",
    entry_point="gym_game.envs:PathfinderEnv",
)


gym.register(
    id="Conquer",
    entry_point="gym_game.envs:ConquerEnv",
)

gym.register(
    id="Conquer2",
    entry_point="gym_game.envs:Conquer2Env",
)