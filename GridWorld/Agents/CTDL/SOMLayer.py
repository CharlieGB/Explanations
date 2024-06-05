import numpy as np

class SOMLayer():

    def __init__(self, maze_dim, input_dim, size, learning_rate, sigma, sigma_const):

        self.size = size
        self.num_units = size * size
        self.num_dims = input_dim
        self.num_weights = input_dim

        self.learning_rate = learning_rate
        self.sigma = sigma
        self.sigma_const = sigma_const

        self.units = {'xy': [], 'w': [], 'etas': np.zeros(self.num_units), 'errors': np.zeros(self.num_units)}
        self.ConstructMap(maze_dim)

        return

    def ConstructMap(self, maze_dim):

        x = 0
        y = 0

        # Construct map
        for u in range(self.num_units):

            self.units['xy'].append([x, y])
            self.units['w'].append(np.random.rand(self.num_weights) * maze_dim)

            x += 1
            if (x >= self.size):
                x = 0
                y += 1

        self.units['xy'] = np.array(self.units['xy'])
        self.units['w'] = np.array(self.units['w'])

        return

    def Update(self, state, unit, delta, update_mask):

        # self.units['w'][unit, :] = state
        # self.units['errors'][unit] = error


        diffs = self.units['xy'] - self.units['xy'][unit, :]
        location_distances = np.sqrt(np.sum(np.square(diffs), axis=-1))

        # sigma = np.clip(reward_value, 0, .01)
        # neighbourhood_values = np.exp(
        #     -np.square(location_distances) / (2.0 * (self.sigma_const + sigma)))

        neighbourhood_values = np.exp(-np.square(location_distances) / (
                2.0 * (self.sigma_const + (delta * self.sigma))))

        # lr = np.clip(reward_value, 0, .01)
        # self.units['w'] += lr * np.expand_dims(neighbourhood_values, axis=-1) * (state - self.units['w'])

        self.units['w'] += np.squeeze((delta * self.learning_rate) * \
                           np.expand_dims(neighbourhood_values, axis=-1) * (
                                   state - self.units['w'])) * \
                           np.tile(np.expand_dims(update_mask, axis=1), (1, 2))

        return

    def GetBestUnit(self, state):

        best_unit = np.argmin(np.sum((self.units['w'] - state) ** 2, axis=-1), axis=0)

        return best_unit