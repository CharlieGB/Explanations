import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

from GridWorld.Functions.Parsers import ParseIntoDataframes

from enum import Enum

directory = 'Sumo'
var = 'type'
explanation_lengths = [5, 3]
explanation_thresholds = [.5, .75, .9]

data_frames, labels = ParseIntoDataframes(directory, ['CTDL', 'DQN'])
df = data_frames[0]

cmap = plt.cm.get_cmap('hot')
norm = colors.Normalize(vmin=0, vmax=1)

for name, maze_df in df.groupby(var):

    maze = maze_df['maze'].tolist()[0]
    heat_map = np.zeros(maze.shape)

    all_memory_fig, all_memory_ax = plt.subplots(2, 3)
    all_memory_ax = all_memory_ax.reshape(-1)
    for ax in all_memory_ax:
        ax.imshow(maze, alpha=.5)
        ax.set_xticks([])
        ax.set_yticks([])

    all_action_fig, all_action_ax = plt.subplots(2, 3)
    all_action_ax = all_action_ax.reshape(-1)
    for ax in all_action_ax:
        ax.imshow(maze, alpha=.5)
        ax.set_xticks([])
        ax.set_yticks([])

    # Take weighted average of top five memories
    weights = maze_df['weights'].tolist()
    memories = maze_df['memories'].tolist()
    actions = maze_df['actions'].tolist()

    dirs = maze_df['dir'].tolist()

    top_weights = []
    top_memories = []
    top_actions = []

    for w, m, a, d in zip(weights, memories, actions, dirs):

        w = np.array(w)
        m = np.array(m)
        a = np.array(a)

        binned_memories = np.rint(m)
        heat_map[binned_memories[:, 0].astype(int), binned_memories[:, 1].astype(int)] += w

        ind_fig, ind_ax = plt.subplots(2, 3)
        ind_ax = ind_ax.reshape(-1)
        for ax in ind_ax:
            ax.imshow(maze, alpha=.5)
            ax.set_xticks([])
            ax.set_yticks([])

        all_memory_ax[0].set_title('All Memories')
        all_action_ax[0].set_title('All Memories')
        ind_ax[0].set_title('All Memories')

        for memory, action, weight in zip(m, a, w):
            ind_ax[0].plot(memory[1], memory[0], marker='*', color=cmap(norm(weight)))
            ind_ax[0].text(memory[1], memory[0] + .5, action, ha='center', fontsize=10 * weight)
            all_memory_ax[0].plot(memory[1], memory[0], marker='*', color=cmap(norm(weight)))
            all_action_ax[0].text(memory[1], memory[0], action, ha='center', va='center', fontsize=10 * weight)

        for i, explanation_length in enumerate(explanation_lengths):
            all_memory_ax[i + 1].set_title('Length: ' + str(explanation_length))
            all_action_ax[i + 1].set_title('Length: ' + str(explanation_length))
            ind_ax[i + 1].set_title('Length: ' + str(explanation_length))

            top_inds = np.sort(np.argsort(w)[-explanation_length:])
            top_weights.append(w[top_inds])
            top_memories.append(m[top_inds])
            top_actions.append(a[top_inds])

            for memory, action, weight in zip(m[top_inds], a[top_inds],
                                              w[top_inds]):
                ind_ax[i + 1].plot(memory[1], memory[0], marker='*', color=cmap(norm(weight)))
                ind_ax[i + 1].text(memory[1], memory[0] + .5, action, ha='center', fontsize=10 * weight)
                all_memory_ax[i + 1].plot(memory[1], memory[0], marker='*', color=cmap(norm(weight)))
                all_action_ax[i + 1].text(memory[1], memory[0], action, ha='center', va='center', fontsize=10 * weight)

        for i, explanation_threshold in enumerate(explanation_thresholds):
            all_memory_ax[i + 1 + len(explanation_lengths)].set_title('Threshold: ' + str(explanation_threshold))
            all_action_ax[i + 1 + len(explanation_lengths)].set_title('Threshold: ' + str(explanation_threshold))
            ind_ax[i + 1 + len(explanation_lengths)].set_title('Threshold: ' + str(explanation_threshold))

            for memory, action, weight in zip(m[w > explanation_threshold], a[w > explanation_threshold],
                                              w[w > explanation_threshold]):
                ind_ax[i + 1 + len(explanation_lengths)].plot(memory[1], memory[0], marker='*', color=cmap(norm(weight)))
                ind_ax[i + 1 + len(explanation_lengths)].text(memory[1], memory[0] + .5, action, ha='center', fontsize=10 * weight)
                all_memory_ax[i + 1 + len(explanation_lengths)].plot(memory[1], memory[0], marker='*', color=cmap(norm(weight)))
                all_action_ax[i + 1 + len(explanation_lengths)].text(memory[1], memory[0], action, ha='center', va='center', fontsize=10 * weight)

        ind_fig.tight_layout()
        ind_fig.savefig('Results/' + directory + '/' + d + '/all_explanations.png')
        plt.close(ind_fig)

    all_memory_fig.tight_layout()
    all_memory_fig.savefig('Plots/' + str(name).replace('\n', '') + '_memory_explanation.png')
    plt.close(all_memory_fig)

    all_action_fig.tight_layout()
    all_action_fig.savefig('Plots/' + str(name).replace('\n', '') + '_action_explanation.png')
    plt.close(all_action_fig)

    plt.figure()
    plt.imshow(maze, alpha=.5)
    plt.imshow(heat_map, cmap='hot')
    plt.colorbar()
    plt.savefig('Plots/' + str(name).replace('\n', '') + '_explanation_heat_map.png')
    plt.close()
