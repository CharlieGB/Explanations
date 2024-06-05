import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from GridWorld.Agents.CTDL.QGraph import QGraph
from GridWorld.Agents.CTDL.SOM import SOM
from GridWorld.Agents.CTDL.QTargetGraph import QTargetGraph
from GridWorld.Functions.ExplanationUtils import ExtractExplanation
from GridWorld.Functions.Plotters import PlotExplanation
from GridWorld.Enums.Enums import MazeType


class Agent(object):

    def __init__(self, directory, maze_params, agent_params):

        self.bSOM = agent_params['bSOM']
        self.directory = directory
        self.maze_width = maze_params['width']
        self.maze_height = maze_params['height']
        self.maze_random_seed = maze_params['random_seed']
        self.maze_type = maze_params['type']

        self.explanation_length = agent_params['exp_length']

        self.q_graph = QGraph(4, self.directory, self.maze_width)
        self.q_graph.SaveGraphAndVariables()
        self.q_target_graph = QTargetGraph(self.directory, self.maze_width)

        self.weighting_decay = agent_params['w_decay']
        self.TD_decay = agent_params['TD_decay']

        self.discount_factor = 0.99
        self.epsilon = 0
        self.final_epsilon = .9 #1
        self.num_epsilon_trials = agent_params['e_trials']
        self.epsilon_increment = self.final_epsilon / self.num_epsilon_trials

        self.c = 10000
        self.ci = 0

        self.results = {'rewards': [], 'lengths': []}
        self.trial_reward = 0
        self.trial_length = 0
        self.plot_num = 0

        self.prev_state = None
        self.prev_action = None
        self.prev_Qvalue = None
        self.bStart_learning = False

        self.ti = 0
        self.w = 0
        self.best_unit = None
        self.eta_counter = 0
        self.trial_num = 0

        self.test_actions = []
        self.test_weights = []
        self.test_memories = []
        self.test_values = []
        self.test_results = []
        self.test_rewards = []

        self.bLoad_exp = agent_params['bLoad_Exp']
        self.bShuffle_Exp = agent_params['bShuffle_Exp']
        self.exp_thresh = agent_params['exp_thresh']
        self.chosen_units = np.array([])

        if (self.bSOM):
            self.CreateSOM(agent_params)

        return

    def CreateSOM(self, agent_params):

        self.SOM = SOM(self.directory, self.maze_width, self.maze_height, 2, agent_params['SOM_size'],
                       agent_params['SOM_alpha'], agent_params['SOM_sigma'],
                       agent_params['SOM_sigma_const'])
        self.Q_alpha = agent_params['Q_alpha']
        self.QValues = np.zeros((agent_params['SOM_size'] * agent_params['SOM_size'], 4))

        self.update_mask = np.ones((self.SOM.SOM_layer.num_units))

        if (agent_params['bLoad_Exp']):
            if self.maze_type == MazeType.random:
                explanation_dir = ('/').join(self.directory.split('/')[:-2]) + \
                                  '/Random_Seed_Chosen_Explanations/Maze' + str(self.maze_random_seed)
            else:
                explanation_dir = ('/').join(self.directory.split('/')[:-2]) + \
                                  '/Maze_Type_Chosen_Explanations/MazeType_' + str(self.maze_type).split('.')[-1]

            num_explanations = len(os.listdir(explanation_dir))
            chosen_explanation = np.random.randint(num_explanations)
            agent_params['chosen_explanation'] = chosen_explanation

            with open(explanation_dir + '/Explanation_' +
                      str(chosen_explanation) + '.pkl', 'rb') as handle:
                explanations = pickle.load(handle)

            actions, memories, values, weights = ExtractExplanation(explanations, self.exp_thresh, self.bShuffle_Exp)

            chosen_units = np.random.choice(self.SOM.SOM_layer.num_units, actions.shape[0], replace=False)

            for i, unit in enumerate(chosen_units.tolist()):
                self.SOM.SOM_layer.units['w'][unit, :] = memories[i, :]
                self.QValues[unit, :] = values[i, :]

            self.chosen_units = chosen_units
            actions_dict = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
            self.explanation_actions = [actions_dict[a] for a in actions.tolist()]
            self.update_mask[self.chosen_units] = 0

        return

    def Update(self, reward, state, bTrial_over, bTest=False, maze=None):

        if(bTest):
            action = self.SelectAction(state, bTest)
            self.test_rewards.append(reward)
            self.test_actions.append(action)
            self.test_weights.append(self.w)
            self.test_memories.append(self.SOM.SOM_layer.units['w'][self.best_unit])
            self.test_values.append(self.QValues[self.best_unit, :])

        else:
            self.RecordResults(bTrial_over, reward)

            if (bTrial_over and self.epsilon < self.final_epsilon):
                self.epsilon += self.epsilon_increment

            if(self.bStart_learning):
                self.UpdateQGraph(reward, state, bTrial_over)

            action = self.SelectAction(state, bTest)

            if(not self.bStart_learning):
                self.bStart_learning = True

            if(bTrial_over):
                self.trial_num += 1

        return action

    def RecordTestResults(self, maze, trial, test_trial):

        self.test_rewards = np.array(self.test_rewards)
        self.test_weights = np.array(self.test_weights)
        self.test_memories = np.array(self.test_memories)
        self.test_values = np.array(self.test_values)

        actions_dict = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        self.test_actions = np.array([actions_dict[a] for a in self.test_actions])

        results = {'rewards': self.test_rewards,
                   'actions': self.test_actions,
                   'weights': self.test_weights,
                   'memories': self.test_memories,
                   'values': self.test_values}

        self.test_results.append(results)

        actions, memories, values, weights = ExtractExplanation(results, self.exp_thresh, self.bShuffle_Exp)
        PlotExplanation(maze, memories, actions, weights,
                        self.directory + '/LearningTrial_' + str(trial) + '_TestTrial_' + str(test_trial))

        self.test_rewards = []
        self.test_actions = []
        self.test_weights = []
        self.test_memories = []
        self.test_values = []

        return

    def SaveTestResults(self):

        with open(self.directory + '/explanation.pkl', 'wb') as f:
            pickle.dump(self.test_results, f, pickle.HIGHEST_PROTOCOL)

        return

    def RecordResults(self, bTrial_over, reward):

        self.trial_reward += reward
        self.trial_length += 1
        if (bTrial_over):
            print('Trial Reward: ' + str(self.trial_reward))
            self.results['rewards'].append(self.trial_reward)
            self.trial_reward = 0

            print('Trial Length: ' + str(self.trial_length) + '\n')
            self.results['lengths'].append(self.trial_length)
            self.trial_length = 0

        return

    def NewMaze(self, directory):

        self.directory = directory
        self.q_graph.directory = directory
        self.SOM.directory = directory
        self.UpdateTargetGraph()

        self.results = {'rewards': [], 'lengths': []}
        self.trial_reward = 0
        self.trial_length = 0
        self.plot_num = 0
        self.prev_state = None
        self.prev_action = None
        self.prev_Qvalue = None
        self.bStart_learning = False
        self.SOM.location_counts = np.zeros((self.maze_height, self.maze_width))

        return

    def GetWeighting(self, best_unit, state):

        diff = np.sum(np.square(self.SOM.SOM_layer.units['w'][best_unit, :] - state))
        w = np.exp(-diff / self.weighting_decay)

        return w

    def GetQValues(self, state, q_graph_values):

        best_unit = self.SOM.GetOutput(state)
        som_action_values = self.QValues[best_unit, :]
        w = self.GetWeighting(best_unit, state)
        q_values = (w * som_action_values) + ((1 - w) * q_graph_values)
        self.w = w
        self.best_unit = best_unit

        return q_values


    def UpdateQGraph(self, reward, state, bTrial_over):

        self.ci += 1
        if (self.ci >= self.c):
            self.UpdateTargetGraph()

        target = self.GetTargetValue(bTrial_over, reward, state)

        self.q_graph.GradientDescentStep(np.expand_dims(self.prev_state, axis=0),
                                         np.expand_dims(self.prev_action, axis=0),
                                         np.expand_dims(target, axis=0))

        if(self.bSOM):
            self.UpdateSOM(target)

        return

    def UpdateTargetGraph(self):
        print('Loading New target Graph')
        self.ci = 0
        self.q_graph.SaveGraphAndVariables()
        self.q_target_graph = QTargetGraph(self.directory, self.maze_width)
        return

    def UpdateSOM(self, target):

        # # calculate the TD error from the DNN
        # error = np.abs(target - np.squeeze(self.q_graph.GetActionValues(np.expand_dims(self.prev_state, axis=0)))[self.prev_action])
        #
        # # get the closest matching unit from memory and calculating weighting
        # prev_best_unit = self.SOM.GetOutput(self.prev_state)
        # w = self.GetWeighting(prev_best_unit, self.prev_state)
        #
        # # decay the errors to get rid of historic memories
        # self.SOM.SOM_layer.units['errors'] *= self.error_decay
        #
        # # update error of closest matching unit based on weighting
        # self.SOM.SOM_layer.units['errors'][prev_best_unit] += self.error_lr * w * (
        #         error - self.SOM.SOM_layer.units['errors'][prev_best_unit])
        #
        # # turn error into a probability of encoding memory
        # delta = np.exp(error / self.TD_decay) - 1
        # prob = np.clip(delta, 0, 1)
        #
        # # print('Error: ' + str(error))
        # # print('Prob: ' + str(prob) + '\n')
        #
        # # encode based on probability and whether it is already in memory or not
        # if (np.random.rand() < prob and self.prev_state.tolist() not in self.SOM.SOM_layer.units['w'].tolist()):
        #
        #     # unit to move is the one with the least amount of accumulated errors
        #     prev_best_unit = np.argmin(self.SOM.SOM_layer.units['errors'])
        #     self.SOM.Update(self.prev_state, prev_best_unit, error)
        #
        #     # if moved then initialise the values of the SOM to the ones in the DNN
        #     vals = np.squeeze(self.q_graph.GetActionValues(np.expand_dims(self.prev_state, axis=0)))
        #     self.QValues[prev_best_unit, :] = vals
        #
        # # update Q value of nearest memory based on w (will be the new memory if it was encoded)
        # w = self.GetWeighting(prev_best_unit, self.prev_state)
        # self.QValues[prev_best_unit, self.prev_action] += self.Q_alpha * w * (
        #             target - self.QValues[prev_best_unit, self.prev_action])

        delta = np.exp(np.abs(target -
                              np.squeeze(self.q_graph.GetActionValues(
                                  np.expand_dims(self.prev_state, axis=0)))[self.prev_action]) / self.TD_decay) - 1

        delta = np.clip(delta, 0, 1)
        # print('Delta: ' + str(delta))

        prev_best_unit = self.SOM.GetOutput(self.prev_state)
        self.SOM.Update(self.prev_state, prev_best_unit, delta, self.update_mask)

        #prev_best_unit = self.SOM.GetOutput(self.prev_state)
        w = self.GetWeighting(prev_best_unit, self.prev_state)
        self.QValues[prev_best_unit, self.prev_action] += self.Q_alpha * w * (
                target - self.QValues[prev_best_unit, self.prev_action]) * self.update_mask[prev_best_unit]


        self.Replay()
        self.SOM.RecordLocationCounts()

        return

    def GetTargetValue(self, bTrial_over, reward, state):

        q_graph_values = np.squeeze(np.array(self.q_target_graph.GetActionValues(np.expand_dims(state, axis=0))))

        if(self.bSOM):
            q_values = self.GetQValues(state, q_graph_values)

        else:
            q_values = q_graph_values

        max_q_value = np.amax(q_values)

        if (bTrial_over):
            target = reward

        else:
            target = reward + (max_q_value * self.discount_factor)

        return target

    def Replay(self):

        units = np.random.randint(0, self.SOM.SOM_layer.num_units, 32)
        actions = np.random.randint(0, 4, 32)

        self.q_graph.GradientDescentStep(self.SOM.SOM_layer.units['w'][units, :], actions, self.QValues[units, actions])

        return


    def SelectAction(self, state, bTest):

        best_unit = self.SOM.GetOutput(state)
        w = self.GetWeighting(best_unit, state)

        if (w > self.exp_thresh and best_unit in self.chosen_units.tolist() and self.bLoad_exp):
            ind = np.where(self.chosen_units == best_unit)[0][0]
            action = self.explanation_actions[ind]
            prev_q = self.QValues[best_unit, action]
        else:
            q_graph_values = np.squeeze(np.array(self.q_graph.GetActionValues(np.expand_dims(state, axis=0))))

            if(self.bSOM):
                q_values = self.GetQValues(state, q_graph_values)
            else:
                q_values = q_graph_values

            if(np.random.rand() > self.epsilon and not bTest):
                action = np.random.randint(4)
            else:
                action = np.argmax(q_values)

            prev_q = q_values[action]

        self.prev_Qvalue = prev_q
        self.prev_action = action
        self.prev_state = np.copy(state)

        return action


    def PlotResults(self):

        plt.figure()
        plt.plot(self.results['rewards'])
        found_goal = np.where(np.array(self.results['rewards']) > 0)
        if(found_goal):
            for loc in found_goal[0]:
                plt.axvline(x=loc, color='g')
        plt.savefig(self.directory + 'AgentTrialRewards.pdf')
        plt.close()

        with open(self.directory + 'Results.pkl', 'wb') as handle:
            pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if (self.bSOM):
            np.save(self.directory + 'LocationCounts', self.SOM.location_counts)

        return

    def PlotValueFunction(self):

        up_value_function = np.zeros((self.maze_height, self.maze_width))
        down_value_function = np.zeros((self.maze_height, self.maze_width))
        left_value_function = np.zeros((self.maze_height, self.maze_width))
        right_value_function = np.zeros((self.maze_height, self.maze_width))

        for row in range(self.maze_height):
            for col in range(self.maze_width):
                q_graph_values = np.squeeze(np.array(self.q_graph.GetActionValues(np.array([[row, col]]))))

                if(self.bSOM):
                    vals = self.GetQValues([row, col], q_graph_values)
                else:
                    vals = q_graph_values

                up_value_function[row, col] = vals[0]
                down_value_function[row, col] = vals[1]
                left_value_function[row, col] = vals[2]
                right_value_function[row, col] = vals[3]

        fig, axes = plt.subplots(2, 2)

        vmin = np.amin([up_value_function, down_value_function, left_value_function, right_value_function])
        vmax = np.amax([up_value_function, down_value_function, left_value_function, right_value_function])

        im = axes[0, 0].imshow(up_value_function, cmap='hot', vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Up Value Function')

        im = axes[0, 1].imshow(down_value_function, cmap='hot', vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Down Value Function')

        im = axes[1, 0].imshow(left_value_function, cmap='hot', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title('Left Value Function')

        im = axes[1, 1].imshow(right_value_function, cmap='hot', vmin=vmin, vmax=vmax)
        axes[1, 1].set_title('Right Value Function')

        for axis in axes.ravel():
            axis.set_xticklabels([])
            axis.set_xticks([])
            axis.set_yticklabels([])
            axis.set_yticks([])

        fig.colorbar(im, ax=axes.ravel().tolist())
        plt.savefig(self.directory + 'ValueFunction' + str(self.plot_num) + '.png')
        plt.close()

        if(self.bSOM):
            self.SOM.PlotResults(self.plot_num)
            self.PlotEtaHeatMap()

        self.plot_num += 1

        return

    def PlotEtaHeatMap(self):
        # Plot eta heat map
        eta_map = np.zeros((self.maze_height, self.maze_width))
        for y in range(self.maze_height):
            for x in range(self.maze_width):
                best_unit = self.SOM.GetOutput([y, x])
                eta_map[y, x] = self.GetWeighting(best_unit, [y, x])
        plt.figure()
        plt.imshow(eta_map, cmap='hot')
        plt.colorbar()
        plt.savefig(self.directory + '/EtaHeatMap' + str(self.eta_counter) + '.png')
        plt.close()

        self.eta_counter += 1

        return


