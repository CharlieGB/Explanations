import numpy as np

def ExtractExplanation(test_results, exp_thresh):
    weights = np.array(test_results['weights'])
    actions = np.array(test_results['actions'])[weights > exp_thresh]
    memories = np.array(test_results['memories'])[weights > exp_thresh, :]
    values = np.array(test_results['values'])[weights > exp_thresh]
    weights = weights[weights > exp_thresh]

    memories_dict = {}
    for i in range(weights.shape[0]):
        m = tuple(memories[i, :])

        if (m in memories_dict):
            if (memories_dict[m]['weight'] < weights[i]):
                memories_dict[m]['weight'] = weights[i]
                memories_dict[m]['ind'] = i
        else:
            memories_dict[m] = {}
            memories_dict[m]['weight'] = weights[i]
            memories_dict[m]['ind'] = i
    final_inds = []
    for key, item in memories_dict.items():
        final_inds.append(item['ind'])

    weights = weights[final_inds]
    actions = actions[final_inds]
    memories = memories[final_inds, :]
    values = values[final_inds]
    return np.squeeze(actions), memories, values, weights

