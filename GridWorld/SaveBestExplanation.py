import os
import shutil
import pickle
import numpy as np

from GridWorld.Functions.Parsers import ParseIntoDataframes
from GridWorld.Functions.ExplanationUtils import ExtractExplanation
from GridWorld.Functions.Plotters import PlotAllExplanations, PlotExplanation
from GridWorld.Parameters import agent_params


directory = 'Maze_Type_No_Explanation'
directory_to_save = 'Maze_Type_Chosen_Explanations'
var = 'type'

# directory = 'Random_Seed_No_Explanation'
# directory_to_save = 'Random_Seed_Chosen_Explanations'
# var = 'random_seed'

# load all the explanations
data_frames = ParseIntoDataframes([directory])
df = data_frames[0]

for name, df_group in df.groupby(var):
    name = 'Maze' + str(int(name)) if isinstance(name, float) else name
    name = name.replace('.', '_').replace('\n', '')
    maze = df_group['maze'].tolist()[0]

    final_test_rewards = np.array([np.sum(r['rewards']) for r in df_group['Explanation_4'].tolist()])
    best_final = np.squeeze(np.argwhere(final_test_rewards == np.amax(final_test_rewards)))

    if best_final.shape == ():
        best_agent = int(best_final)
    else:
        training_rewards = np.sum(df_group['rewards'].tolist(), axis=1)[best_final]
        best_training = np.argmax(training_rewards)
        best_agent = best_final[best_training]

    training_rewards = df_group['rewards'].tolist()
    training_lengths = df_group['lengths'].tolist()
    # training_rewards = np.sum(df_group['rewards'].tolist(), axis=1)
    # best_agent = np.argmax(training_rewards)

    # Save the explanations from the best performing agent
    save_dir = 'Results/' + directory_to_save + '/' + str(name)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    df_group = df_group.filter(like='Explanation')

    for key in df_group.keys():
        values = df_group[key].values

        with open(save_dir + '/' + key + '.pkl', 'wb') as f:
            pickle.dump(values[best_agent], f, pickle.HIGHEST_PROTOCOL)

        actions, memories, _, weights = ExtractExplanation(values[best_agent], agent_params['exp_thresh'], False)
        PlotExplanation(maze, memories, actions, weights, 'Plots/Best_Explanation_' +
                        str(name) + '_' + str(key) + '_' + str(best_agent) + '.pdf')

        actions_list = []
        memories_list = []
        weights_list = []
        test_rewards = []

        for agent, test_results in enumerate(values.tolist()):

            actions, memories, _, weights = ExtractExplanation(test_results, agent_params['exp_thresh'], False)
            actions_list.append(actions)
            memories_list.append(memories)
            weights_list.append(weights)
            test_rewards.append(np.sum(test_results['rewards']))


        PlotAllExplanations(maze, memories_list, actions_list, weights_list, training_rewards, training_lengths,
                            test_rewards, 'Plots/All_Explanations' + str(name) + '_' + str(key) + '.pdf', best_agent)

    # for agent, test_results in values.items():
    #     actions, memories, values, weights = ExtractExplanation(test_results, agent_params['exp_thresh'])
    #     weights_list.append(actions)
    #     memories_list.append(memories)
    #     actions_list.append(actions)
    # PlotExplanationHeatMap(name, maze, weights_list, memories_list, actions_list)
