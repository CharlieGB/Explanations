import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from Gym.Functions.Parsers import ParseIntoDataframes

bVideos = False
directory = 'Explanations2'
explanation_lengths = [5, 3]
explanation_thresholds = [.5, .75, .9]

data_frames, labels = ParseIntoDataframes(directory, ['CTDL_A2C'])
df = data_frames[0]

dirs = df['dir'].tolist()

df = df.filter(like='Explanation')

cmap = plt.cm.get_cmap('hot')
norm = colors.Normalize(vmin=0, vmax=1)

for key, value_list in df.to_dict().items():

    heat_map = np.zeros((10,10))

    all_memory_fig, all_memory_ax = plt.subplots(2, 3)
    all_memory_ax = all_memory_ax.reshape(-1)

    all_action_fig, all_action_ax = plt.subplots(2, 3)
    all_action_ax = all_action_ax.reshape(-1)

    weights = []
    memories = []
    actions = []
    states = []

    for k, v in value_list.items():
        weights.append(v['weights'])
        memories.append(v['memories'])
        actions.append(v['actions'])
        states.append(v['states'])

    top_weights = []
    top_memories = []
    top_actions = []

    for w, m, a, s, d in zip(weights, memories, actions, states, dirs):

        w = np.array(w)
        a = np.squeeze(np.array(a))
        a_text = np.array(['placeholder' for _ in range(a.shape[0])])
        a_text[a > 0] = 'forward'
        a_text[a < 0] = 'backward'

        a = np.abs(a)

        m = np.array(m)

        #binned_memories = np.rint(m)
        #heat_map[binned_memories[:, 0].astype(int), binned_memories[:, 1].astype(int)] += w

        ind_fig, ind_ax = plt.subplots(2, 3, figsize=(7,5))
        ind_ax = ind_ax.reshape(-1)
        for ax in ind_ax:
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])

        all_memory_ax[0].set_title('All Memories')
        all_action_ax[0].set_title('All Memories')
        ind_ax[0].set_title('All Memories')

        for i, (memory, action, weight, state, action_text) in enumerate(zip(m, a, w, s, a_text)):
            ind_ax[0].plot(memory[0], memory[1], marker='*', color=cmap(norm(weight)), markeredgecolor=(0, 0, 0, 1), markeredgewidth=.1)
            ind_ax[0].text(memory[0], memory[1] + .1, action_text, ha='center', fontsize=10 * action)
            all_memory_ax[0].plot(memory[0], memory[1], marker='*', color=cmap(norm(weight)), markeredgecolor=(0, 0, 0, 1), markeredgewidth=.1)
            all_action_ax[0].text(memory[0], memory[1], action, ha='center', va='center', fontsize=10)# * weight)

            if(bVideos):
                plt.figure()
                plt.plot(state[0, 0], state[0, 1], marker='o', color=cmap(norm(weight)))
                plt.plot(memory[0], memory[1], marker='*', color='k')
                plt.text(state[0, 0], state[0, 1] + .1, action, ha='center', fontsize=10)
                plt.xlim([-1.5, 1.5])
                plt.ylim([-1.5, 1.5])
                plt.xticks([])
                plt.yticks([])
                plt.xlabel('Position')
                plt.ylabel('Speed')
                plt.savefig('Results/' + directory + '/' + d + '/' + key + 'AllTest' + str(i) + '.png')
                plt.close()

        for i, explanation_length in enumerate(explanation_lengths):
            all_memory_ax[i + 1].set_title('Length: ' + str(explanation_length))
            all_action_ax[i + 1].set_title('Length: ' + str(explanation_length))
            ind_ax[i + 1].set_title('Length: ' + str(explanation_length))

            top_inds = np.sort(np.argsort(w)[-explanation_length:])

            for j, (memory, action, weight, state, action_text) in enumerate(zip(m[top_inds], a[top_inds],
                                              w[top_inds], s[top_inds], a_text[top_inds])):
                ind_ax[i + 1].plot(memory[0], memory[1], marker='*', color=cmap(norm(weight)), markeredgecolor=(0, 0, 0, 1), markeredgewidth=.1)
                ind_ax[i + 1].text(memory[0], memory[1] + .1, action_text, ha='center', fontsize=10 * action)
                all_memory_ax[i + 1].plot(memory[0], memory[1], marker='*', color=cmap(norm(weight)), markeredgecolor=(0, 0, 0, 1), markeredgewidth=.1)
                all_action_ax[i + 1].text(memory[0], memory[1], action, ha='center', va='center', fontsize=10)# * weight)

                if(bVideos):
                    plt.figure()
                    plt.plot(state[0, 0], state[0, 1], marker='o', color=cmap(norm(weight)))
                    plt.plot(memory[0], memory[1], marker='*', color='k')
                    plt.text(state[0, 0], state[0, 1] + .1, action, ha='center', fontsize=10)
                    plt.xlim([-1.5, 1.5])
                    plt.ylim([-1.5, 1.5])
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlabel('Position')
                    plt.ylabel('Speed')
                    plt.savefig('Results/' + directory + '/' + d + '/' + key + 'Length' + str(explanation_length) + 'Test' + str(j) + '.png')
                    plt.close()



        for i, explanation_threshold in enumerate(explanation_thresholds):
            all_memory_ax[i + 1 + len(explanation_lengths)].set_title('Threshold: ' + str(explanation_threshold))
            all_action_ax[i + 1 + len(explanation_lengths)].set_title('Threshold: ' + str(explanation_threshold))
            ind_ax[i + 1 + len(explanation_lengths)].set_title('Threshold: ' + str(explanation_threshold))

            for j, (memory, action, weight, state, action_text) in enumerate(zip(m[w > explanation_threshold], a[w > explanation_threshold],
                                              w[w > explanation_threshold], s[w > explanation_threshold], a_text[w > explanation_threshold])):
                ind_ax[i + 1 + len(explanation_lengths)].plot(memory[0], memory[1], marker='*', color=cmap(norm(weight)), markeredgecolor=(0, 0, 0, 1), markeredgewidth=.1)
                ind_ax[i + 1 + len(explanation_lengths)].text(memory[0], memory[1] + .1, action_text, ha='center', fontsize=10 * action)
                all_memory_ax[i + 1 + len(explanation_lengths)].plot(memory[0], memory[1], marker='*', color=cmap(norm(weight)), markeredgecolor=(0, 0, 0, 1), markeredgewidth=.1)
                all_action_ax[i + 1 + len(explanation_lengths)].text(memory[0], memory[1], action, ha='center', va='center', fontsize=10)# * weight)

                if(bVideos):
                    plt.figure()
                    plt.plot(state[0, 0], state[0, 1], marker='o', color=cmap(norm(weight)))
                    plt.plot(memory[0], memory[1], marker='*', color='k')
                    plt.text(state[0, 0], state[0, 1] + .1, action, ha='center', fontsize=10)
                    plt.xlim([-1.5, 1.5])
                    plt.ylim([-1.5, 1.5])
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlabel('Position')
                    plt.ylabel('Speed')
                    plt.savefig('Results/' + directory + '/' + d + '/' + key + 'Threshold' + str(explanation_threshold) + 'Test' + str(j) + '.png')
                    plt.close()

        ind_fig.tight_layout()
        ind_fig.savefig('Results/' + directory + '/' + d + '/' + key + 'all_explanations.png')
        plt.close(ind_fig)

    all_memory_fig.tight_layout()
    all_memory_fig.savefig('Plots/' + key + 'memory_explanation.png')
    plt.close(all_memory_fig)

    all_action_fig.tight_layout()
    all_action_fig.savefig('Plots/' + key + 'action_explanation.png')
    plt.close(all_action_fig)

    # plt.figure()
    # plt.imshow(heat_map, cmap='hot')
    # plt.colorbar()
    # plt.savefig('Plots/explanation_heat_map.png')
    # plt.close()
