import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

from Gym.Functions.ExplanationUtils import ExtractExplanation
from Gym.Functions.Parsers import ParseIntoDataframes
from Gym.Functions.Plotters import PlotExplanation
from Gym.Parameters import agent_params

directory = 'NoExplanation'
directory_to_save = 'ChosenExplanations'

# load all the explanations
data_frames = ParseIntoDataframes([directory])
df = data_frames[0]

# get the best agent
final_rewards = np.array(df['rewards'].tolist())[:, -1]
best_final = np.squeeze(np.argwhere(final_rewards == np.amax(final_rewards)))

if best_final.shape == ():
    best_agent = int(best_final)
else:
    summed_rewards = np.sum(df['rewards'].tolist(), axis=1)[best_final]
    best_training = np.argmax(summed_rewards)
    best_agent = best_final[best_training]

means = df['mean'].values
vars = df['var'].values
df = df.filter(like='Explanation')

for key, values in df.to_dict().items():

    with open('Results/' + directory_to_save + '/' + key + '.pkl', 'wb') as f:
        pickle.dump(values[best_agent], f, pickle.HIGHEST_PROTOCOL)

    actions, memories, _, weights = ExtractExplanation(values[best_agent], agent_params['exp_thresh'], False)

    states = np.array(values[best_agent]['states'])
    mean = np.array([float(x) for x in means[best_agent].replace('[', '').replace(']\n','').split(' ')])
    var = np.array([float(x) for x in vars[best_agent].replace('[', '').replace(']\n','').split(' ')])

    scaler = StandardScaler()
    scaler.scale_ = np.sqrt(var)
    scaler.mean_ = mean
    scaler.var_ = var

    PlotExplanation(states=states, memories=memories,
                    actions=actions, weights=weights,
                    save_name='Plots/Best_Explanation_' + str(key) + '.pdf',
                    scaler=scaler)
