import numpy as np

from GridWorld.Parameters import maze_params, agent_params
from GridWorld.Functions.Run import Run
from GridWorld.Enums.Enums import AgentType

def RunRandomSeedSweep():

    agents = [AgentType.CTDL]
    random_seeds = np.arange(0, 30).tolist()

    for i in range(maze_params['num_repeats']):
        for random_seed in random_seeds:
            for agent in agents:
                agent_params['agent_type'] = agent
                maze_params['random_seed'] = random_seed
                Run(maze_params, agent_params)

    return
