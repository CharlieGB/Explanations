import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def PlotComparisons(data_frames, labels, num_trials_list):

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes = axes.reshape(-1)
    colors = ['b', 'r', 'g', 'k', 'c', 'm']

    for ax, num_trials in zip(axes, num_trials_list):

        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        for df, label, color in zip(data_frames, labels, colors):

            reward_results = []
            cum_reward_results = []

            for rewards, lengths in zip(df['rewards'], df['lengths']):
                reward_results.append(rewards)
                cum_reward_results.append(np.cumsum(rewards))

            reward_results = np.array(reward_results)
            best_run = np.argmax(np.sum(reward_results, axis=1))
            reward_results = reward_results[:, :num_trials]

            y = np.mean(reward_results, axis=0)
            x = np.arange(y.shape[0])
            error = np.std(reward_results, axis=0)

            ax.plot(x, y, color=color, label=label, linewidth=3)
            ax.fill_between(x, y-error, y+error, color=color, alpha=.25)
            ax.plot(x, reward_results[best_run, :], color=color, linestyle=':')

        ax.legend(loc='lower left')

    fig.tight_layout()
    fig.savefig('Plots/ComparisonPlot.pdf')
    plt.close(fig)

    return


def PlotExplanation(states, memories, actions, weights, save_name, scaler):

    try:

        states = np.squeeze(np.array(states))
        states = scaler.inverse_transform(states)
        memories = scaler.inverse_transform(memories)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        img = plt.imread(dir_path + "/continuous_mountain_car.jpg")

        cmap = plt.cm.get_cmap('hot')
        norm = colors.Normalize(vmin=0, vmax=1)

        a_text = np.array(['placeholder' for _ in range(actions.shape[0])])
        a_text[actions > 0] = 'forward'
        a_text[actions < 0] = 'backward'
        actions = np.abs(actions)

        fig, ax = plt.subplots(1, 1)
        # ax.set_xlim([-1.5, 1.5])
        ax.set_title('Explanation')
        ax.imshow(img, extent=[-1.2, 0.6, -0.07, 0.07], aspect='auto')

        for i, (memory, action, weight, action_text) in enumerate(zip(memories, actions, weights, a_text)):
            ax.plot(memory[0], memory[1], marker='*', color=cmap(norm(weight)),
                    markeredgecolor=(0, 0, 0, 1), markeredgewidth=.1)
            ax.text(memory[0], memory[1] + .005, action_text, ha='center', fontsize=5 + (5 * action))

        ax.plot(states[:, 0], states[:, 1], 'k--', linewidth=.5)
        fig.savefig(save_name)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(memories[:, 0], memories[:, 1], np.arange(memories.shape[0]), markerfacecolor='r',
                marker='o', linestyle='--', color='k')
        for i, (memory, action, weight, action_text) in enumerate(zip(memories, actions, weights, a_text)):
            ax.text(memory[0], memory[1], i + .5, action_text, ha='center', fontsize=5 + (5 * action))
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_zlabel('Time')
        fig.tight_layout()
        fig.savefig(save_name + '_3D')
        plt.close(fig)

    except:
        pass

    return


def PlotExplanationResults(var, data_frames):



    return

def PlotExplanationHeatMap(name, weights, memories, actions):



    return
