import numpy as np

from GridWorld.Parameters import maze_params, agent_params
from GridWorld.Functions.Run import Run
from GridWorld.Classes.Maze import MazeType
from GridWorld.Enums.Enums import AgentType

def RunMazeTypeSweep():

    agents = [AgentType.CTDL]
    maze_types = [MazeType.obstacle1, MazeType.obstacle2]

    if agent_params['bLoad_Exp'] and not agent_params['bShuffle_Exp']:
        explanations = np.arange(5).tolist()
    else:
        explanations = [4]

    for i in range(maze_params['num_repeats']):
        for maze_type in maze_types:
            for agent in agents:
                for exp_num in explanations:
                    agent_params['agent_type'] = agent
                    agent_params['chosen_explanation'] = exp_num
                    maze_params['type'] = maze_type
                    Run(maze_params, agent_params)

    return
