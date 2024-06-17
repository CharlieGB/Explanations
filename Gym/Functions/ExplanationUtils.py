import numpy as np

def ExtractExplanation(test_results, exp_thresh, bShuffle_Exp):
    weights = np.array(test_results['weights'])
    actions = np.array(test_results['actions'])
    memories = np.array(test_results['memories'])
    values = np.array(test_results['values'])

    thresholded_actions = actions[weights > exp_thresh]
    thresholded_memories = memories[weights > exp_thresh, :]
    thresholded_values = values[weights > exp_thresh]
    thresholded_weights = weights[weights > exp_thresh]

    memories_dict = {}
    for i in range(thresholded_weights.shape[0]):
        m = tuple(thresholded_memories[i, :])

        if (m in memories_dict):
            if (memories_dict[m]['weight'] < thresholded_weights[i]):
                memories_dict[m]['weight'] = thresholded_weights[i]
                memories_dict[m]['ind'] = i
        else:
            memories_dict[m] = {}
            memories_dict[m]['weight'] = thresholded_weights[i]
            memories_dict[m]['ind'] = i
    final_inds = []
    for key, item in memories_dict.items():
        final_inds.append(item['ind'])

    explanation_length = len(final_inds)

    if bShuffle_Exp:
        final_inds = np.random.choice(weights.shape[0], explanation_length, replace=False).tolist()
        exp_weights = weights[final_inds]
        exp_actions = actions[final_inds]
        exp_memories = memories[final_inds, :]
        exp_values = values[final_inds]
    else:
        exp_weights = thresholded_weights[final_inds]
        exp_actions = thresholded_actions[final_inds]
        exp_memories = thresholded_memories[final_inds, :]
        exp_values = thresholded_values[final_inds]

    return np.squeeze(exp_actions), exp_memories, exp_values, exp_weights

