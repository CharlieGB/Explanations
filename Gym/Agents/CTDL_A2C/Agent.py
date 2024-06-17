import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pickle
import sklearn.preprocessing

from Gym.Agents.CTDL_A2C.ACGraph import ACGraph
from Gym.Agents.CTDL_A2C.SOM import DeepSOM
from Gym.Enums.Enums import EnvType
from Gym.Functions.ExplanationUtils import ExtractExplanation
from Gym.Functions.Plotters import PlotExplanation


class Agent(object):

    def __init__(self, directory, env_params, agent_params):

        self.bSOM = agent_params['bSOM']
        self.directory = directory
        self.action_maxs = env_params['action_maxs']
        self.action_mins = env_params['action_mins']
        self.input_dim = env_params['state_dim']

        self.ac_graph = ACGraph(self.input_dim, self.action_mins, self.action_maxs, self.directory)
        self.ac_graph.SaveGraphAndVariables()

        self.bLoad_exp = agent_params['bLoad_Exp']
        self.exp_thresh = agent_params['exp_thresh']
        self.bShuffle_Exp = agent_params['bShuffle_Exp']

        if (self.bSOM):
            self.CreateSOM(agent_params)

        if (self.bLoad_exp):
            self.LoadExplanation(agent_params)

        self.weighting_decay = agent_params['w_decay']
        self.TD_decay = agent_params['TD_decay']

        self.discount_factor = 0.99
        self.epsilon = 1

        self.results = {'rewards': [], 'lengths': []}
        self.trial_reward = 0
        self.trial_length = 0
        self.plot_num = 0

        self.prev_state = None
        self.prev_action = None
        self.prev_Vvalue = None
        self.bStart_learning = False

        self.beta = .1
        self.beta_trials = 25
        self.beta_increment = self.beta / self.beta_trials
        self.trial_num = 0

        state_space_samples = np.array(
            [env_params['env_obj'].observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(state_space_samples)

        agent_params['mean'] = self.scaler.mean_
        agent_params['var'] = self.scaler.var_

        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07

        self.test_actions = []
        self.test_weights = []
        self.test_memories = []
        self.test_states = []
        self.test_values = []
        self.test_results = []
        self.test_rewards = []

        self.ti = 0

        return

    def CreateSOM(self, agent_params):
        self.SOM = DeepSOM(self.directory, self.input_dim, agent_params['SOM_size'],
                           agent_params['SOM_alpha'], agent_params['SOM_sigma'],
                           agent_params['SOM_sigma_const'])
        self.V_alpha = agent_params['Q_alpha']
        self.VValues = np.zeros((agent_params['SOM_size'] * agent_params['SOM_size']))
        self.update_mask = np.ones((self.SOM.SOM_layer.num_units))


    def LoadExplanation(self, agent_params):
        explanation_dir = ('/').join(self.directory.split('/')[:-2]) + '/ChosenExplanations'
        num_explanations = len(os.listdir(explanation_dir))
        chosen_explanation = np.random.randint(num_explanations)
        agent_params['chosen_explanation'] = chosen_explanation

        with open(explanation_dir + '/Explanation_' +
                  str(chosen_explanation) + '.pkl', 'rb') as handle:
            explanations = pickle.load(handle)

        actions, memories, values, weights = ExtractExplanation(explanations, self.exp_thresh, self.bShuffle_Exp)

        if agent_params["bSOM"]:
            chosen_units = np.random.choice(self.SOM.SOM_layer.num_units, actions.shape[0], replace=False)

            for i, unit in enumerate(chosen_units.tolist()):
                self.SOM.SOM_layer.units['w'][unit, :] = memories[i, :]
                self.VValues[unit] = values[i]

            self.chosen_units = chosen_units
            self.update_mask[self.chosen_units] = 0

        self.explanation_actions = actions
        self.explanation_memories = memories
        self.explanation_values = values
        self.explanation_weights = weights

    def ScaleState(self, state):
        scaled = self.scaler.transform([state])
        return scaled

    def Update(self, reward, state, bTrial_over, bTest):

        state = self.ScaleState(np.squeeze(state))

        if (bTest):
            action = self.SelectAction(state)

            best_unit = self.SOM.GetOutput(state)
            som_value = self.VValues[best_unit]
            w = self.GetWeighting(best_unit, state)
            memory = tuple(self.SOM.SOM_layer.units['w'][best_unit].tolist())

            self.test_states.append(state)
            self.test_actions.append(action)
            self.test_weights.append(w)
            self.test_memories.append(memory)
            self.test_values.append(som_value)
            self.test_rewards.append(reward)

            if(self.ti == 0):
                self.test_fig, self.test_ax = plt.subplots(1)
                self.test_ax.set_xlabel('Position')
                self.test_ax.set_ylabel('Speed')

            cmap = plt.cm.get_cmap('hot')
            norm = colors.Normalize(vmin=0, vmax=1)

            self.test_ax.plot(state[0,0], state[0,1], marker='o', color=cmap(norm(w)))
            self.ti += 1

        else:
            self.RecordResults(bTrial_over, reward)

            if (self.bStart_learning):
                self.UpdateACGraph(bTrial_over, reward, state)

            action = self.SelectAction(state)

            if (not self.bStart_learning):
                self.bStart_learning = True

        return action

    def RecordTestResults(self, trial, test_trial):

        self.test_fig.savefig(self.directory + '/LearningTrial_' + str(trial) + 'TestTrial_' + str(test_trial) + '.png')
        plt.close(self.test_fig)
        self.ti = 0

        results = {'rewards': self.test_rewards,
                   'actions': self.test_actions,
                   'weights': self.test_weights,
                   'memories': self.test_memories,
                   'states': self.test_states,
                   'values': self.test_values}

        self.test_results.append(results)

        states = self.test_states
        actions, memories, values, weights = ExtractExplanation(results, self.exp_thresh, self.bShuffle_Exp)
        PlotExplanation(states, memories, actions, weights,
                        self.directory + '/LearningTrial_' + str(trial) + '_TestTrial_' + str(test_trial), self.scaler)

        self.test_actions = []
        self.test_weights = []
        self.test_memories = []
        self.test_states = []
        self.test_values = []
        self.test_rewards = []

        return

    def SaveTestResults(self):

        with open(self.directory + '/explanation.pkl', 'wb') as f:
            pickle.dump(self.test_results, f, pickle.HIGHEST_PROTOCOL)

        return


    def RecordResults(self, bTrial_over, reward):

        self.trial_reward += reward
        self.trial_length += 1
        if (bTrial_over):
            self.trial_num += 1
            self.results['rewards'].append(self.trial_reward)
            print('Cumulative Episode Reward: ' + str(self.trial_reward))

            if (self.trial_reward > 0):
                print('Found Goal!')

            self.trial_reward = 0

            self.results['lengths'].append(self.trial_length)
            self.trial_length = 0

            if(self.trial_num <= self.beta_trials):
                self.beta -= self.beta_increment

        return

    def GetWeighting(self, best_unit, state):

        diff = np.sum(np.square(self.SOM.SOM_layer.units['w'][best_unit, :] - state))
        w = np.exp(-diff / self.weighting_decay)

        return w

    def GetVValues(self, state, critic_value):

        best_unit = self.SOM.GetOutput(state)
        som_value = self.VValues[best_unit]
        w = self.GetWeighting(best_unit, state)
        state_value = (w * som_value) + ((1 - w) * critic_value)

        return state_value

    def UpdateACGraph(self, bTrial_over, reward, state):

        prev_state_value = self.ac_graph.GetStateValue(self.prev_state)
        target = self.GetTargetValue(bTrial_over, reward, state)

        delta = target - prev_state_value
        self.ac_graph.GradientDescentStep(self.prev_state, self.prev_action, target, delta, self.beta)

        if (self.bSOM):
            self.UpdateSOM(target)

        return

    def UpdateSOM(self, target):

        prev_best_unit = self.SOM.GetOutput(self.prev_state)

        delta = np.exp(np.abs(target -
                              np.squeeze(self.ac_graph.GetStateValue(
                                  self.prev_state))) / self.TD_decay) - 1


        delta = np.clip(delta, 0, 1)
        #print('Delta: ' + str(delta))
        self.SOM.Update(self.prev_state, prev_best_unit, delta, self.update_mask)

        # prev_best_unit = self.SOM.GetOutput(self.prev_state)
        w = self.GetWeighting(prev_best_unit, self.prev_state)
        self.VValues[prev_best_unit] += self.V_alpha * w * (
                target - self.VValues[prev_best_unit]) * self.update_mask[prev_best_unit]

        return

    def GetTargetValue(self, bTrial_over, reward, state):

        critic_value = np.squeeze(np.array(self.ac_graph.GetStateValue(state)))

        if(self.bSOM):
            state_value = self.GetVValues(state, critic_value)
        else:
            if self.bLoad_exp:
                diff = np.sum(np.square(self.explanation_memories - state), axis=-1)
                w = np.exp(-diff / self.weighting_decay)

                max_w = np.amax(w)
                max_ind = np.argmax(w)

                if max_w > self.exp_thresh:
                    state_value = np.array((max_w * self.explanation_values[max_ind]) + ((1 - max_w) * critic_value))
                    # state_value = np.array([self.explanation_values[max_ind]])
                else:
                    state_value = critic_value
            else:
                state_value = critic_value

        if (bTrial_over):
            target = reward
        else:
            target = reward + (state_value * self.discount_factor)

        return target


    def SelectAction(self, state):

        if(self.bLoad_exp):
            if self.bSOM:
                best_unit = self.SOM.GetOutput(state)
                w = self.GetWeighting(best_unit, state)

                if(w > self.exp_thresh and best_unit in self.chosen_units.tolist()):
                    ind = np.where(self.chosen_units == best_unit)[0][0]
                    action = np.array([[self.explanation_actions[ind]]])
                else:
                    action = self.ac_graph.GetAction(state)
            else:
                diff = np.sum(np.square(self.explanation_memories - state), axis=-1)
                w = np.exp(-diff / self.weighting_decay)

                max_w = np.amax(w)
                max_ind = np.argmax(w)

                if max_w > self.exp_thresh:
                    action = np.array([[self.explanation_actions[max_ind]]])
                else:
                    action = self.ac_graph.GetAction(state)

        else:
            action = self.ac_graph.GetAction(state)

        self.prev_action = action
        self.prev_state = np.copy(state)

        return action


    def PlotResults(self, task):

        plt.switch_backend('agg')
        plt.figure()
        plt.plot(self.results['rewards'])
        plt.savefig(self.directory + 'AgentTrialRewards.pdf')
        plt.close()

        with open(self.directory + 'Results.pkl', 'wb') as handle:
            pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def PlotSOMResults(self, trial, task):

        if(task == EnvType.MountainCarContinuous):

            print('Plotting SOM Units...')
            SOM_units = self.SOM.SOM_layer.units['w']

            cmap = plt.cm.get_cmap('hot')
            norm = colors.Normalize(vmin=np.amin(SOM_units[:,1]), vmax=np.amax(SOM_units[:,1]))

            inds = np.argsort(SOM_units[:, 0])

            plt.figure()
            for i, ind in enumerate(inds):
                unit = SOM_units[ind, :]
                plt.plot(unit[0], i, color=cmap(norm(unit[1])), marker='o', markersize=1)

            plt.yticks([])
            plt.savefig(self.directory + 'SOM_locations' + str(trial) + '.png')
            plt.close()

            print('Plotting Eta Map...')
            positions = np.linspace(self.min_position, self.max_position, 100).tolist()
            speeds = np.linspace(-self.max_speed, self.max_speed, 100)
            eta_map = np.zeros((len(speeds), len(positions)))

            for x, pos in enumerate(positions):
                for y, speed in enumerate(speeds):
                    state = self.ScaleState(np.array([pos, speed]))
                    best_unit = self.SOM.GetOutput(state)
                    eta_map[y, x] = self.GetWeighting(best_unit, state)

            plt.figure()
            plt.imshow(eta_map, cmap='hot')
            plt.colorbar()
            plt.xlabel('Position')
            plt.ylabel('Velocity')
            plt.savefig(self.directory + '/EtaHeatMap' + str(trial) + '.png')
            plt.close()

        return

