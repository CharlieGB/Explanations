import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from GridWorld.Enums.Enums import MazeType

BORDER_VAL = -2

class Maze(object):


    def __init__(self, directory, maze_params):

        np.random.seed(maze_params['random_seed'])

        self.type = maze_params['type']
        if self.type == MazeType.human:
            self.maze_num = maze_params['maze_num']
        self.width = maze_params['width']
        self.height = maze_params['height']
        self.num_hazards = maze_params['num_hazards']
        self.num_rewards = maze_params['num_rewards']
        self.max_steps = maze_params['max_steps']
        self.directory = directory

        self.ConstructMaze()
        self.Reset()

        self.step = 0

        return


    def ConstructMaze(self):

        self.maze = np.zeros((self.height * self.width))

        if(self.type == MazeType.random):
            self.ConstructRandomMaze()
        elif(self.type == MazeType.direct):
            self.ConstructDirectMaze()
        elif (self.type == MazeType.obstacle1):
            self.ConstructFirstObstacleMaze()
        elif (self.type == MazeType.obstacle2):
            self.ConstructSecondObstacleMaze()
        elif (self.type == MazeType.human):
            self.ConstructHumanMaze()

        plt.figure()
        plt.imshow(self.maze)
        plt.savefig(self.directory + 'Maze.pdf')
        plt.close()

        np.save(self.directory + 'Maze', self.maze)

        self.start = np.squeeze(np.array(np.where(self.maze == 2)))
        self.maze[self.start[0], self.start[1]] = 0

        return

    def ConstructHumanMaze(self):

        mat_file = sio.loadmat('GridWorld/Data/m' + str(self.maze_num))
        maze = [list(s) for s in mat_file['m' + str(self.maze_num)]]
        mapping = {'o': -1, 'e': 1, 'a': 2, 's': 0, 'b': BORDER_VAL}

        for row in range(len(maze)):
            for col in range(len(maze[0])):
                maze[row][col] = mapping[maze[row][col]]

        self.maze = np.array(maze).astype(np.int32)

    def ConstructRandomMaze(self):

        inds = np.random.choice(np.arange(self.height * self.width), self.num_hazards + self.num_rewards + 1,
                                replace=False)
        self.maze[inds[:self.num_hazards]] = -1
        self.maze[inds[self.num_hazards:self.num_hazards + self.num_rewards]] = 1
        self.maze[inds[-1]] = 2
        self.maze = self.maze.reshape((self.height, self.width))

        return

    def ConstructDirectMaze(self):

        self.maze = self.maze.reshape((self.height, self.width))
        self.maze[0, int(self.width / 2)] = 1
        self.maze[-1, int(self.width / 2)] = 2
        self.maze[:, :int(self.width / 2) - 2] = -1
        self.maze[:, int(self.width / 2) + 3:] = -1

        return

    def ConstructFirstObstacleMaze(self):

        self.ConstructDirectMaze()
        self.maze[int(self.height / 2) - 1, int(self.width / 2) - 1: int(self.width / 2) + 2] = -1

        return

    def ConstructSecondObstacleMaze(self):

        self.ConstructDirectMaze()
        self.maze[int(self.height / 3), int(self.width / 2) - 2:int(self.width / 2) + 1] = -1
        self.maze[int(self.height / 3) * 2, int(self.width / 2):int(self.width / 2) + 3] = -1

        return

    def GetMaze(self):

        maze = np.copy(self.maze)
        maze[self.start[0], self.start[1]] = 2

        return maze

    def Reset(self):

        self.working_maze = np.copy(self.maze)
        self.state = np.copy(self.start)

        return

    def Update(self, action):

        self.step += 1
        bTrial_over = False
        self.reward = 0

        self.UpdateState(action)

        if(self.reward > 0 or self.step >= self.max_steps):
            bTrial_over = True
            self.step = 0
            self.Reset()

        self.reward += -.05

        return self.reward, self.state, bTrial_over

    def UpdateState(self, action):

        if (action == 0):
            if (self.state[0] > 0):
                next_state = self.state + np.array([-1, 0])
                if self.working_maze[next_state[0], next_state[1]] != BORDER_VAL:
                    self.state = next_state

        elif (action == 1):
            if (self.state[0] < self.height - 1):
                next_state = self.state + np.array([1, 0])
                if self.working_maze[next_state[0], next_state[1]] != BORDER_VAL:
                    self.state = next_state

        elif (action == 2):
            if (self.state[1] > 0):
                next_state = self.state + np.array([0, -1])
                if self.working_maze[next_state[0], next_state[1]] != BORDER_VAL:
                    self.state = next_state

        elif (action == 3):
            if (self.state[1] < self.width - 1):
                next_state = self.state + np.array([0, 1])
                if self.working_maze[next_state[0], next_state[1]] != BORDER_VAL:
                    self.state = next_state

        self.reward = self.working_maze[self.state[0], self.state[1]]

        if (self.reward > 0):
            self.working_maze[self.state[0], self.state[1]] = 0

        return
