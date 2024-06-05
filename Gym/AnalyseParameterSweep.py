import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from Gym.Functions.Parsers import ParseDataFrame

dir = 'ParamSweep'

all_files = os.listdir('Results/' + dir)
folders = []

for f in all_files:

    if (f == '.DS_Store' or f == '.keep'):
        pass
    else:
        folders.append(f)

df = ParseDataFrame(folders, dir)


rewards = np.sum(np.array(df['rewards'].tolist()), axis=1)
lengths = np.sum(np.array(df['lengths'].tolist()), axis=1)

max_reward = np.argmax(rewards)
min_length = np.argmin(lengths)

print('Max Reward:')
print('Directory: ' + df['dir'][max_reward])
print('SOM_alpha: ' + str(df['SOM_alpha'][max_reward]))
print('SOM_sigma: ' + str(df['SOM_sigma'][max_reward]))
print('w_decay: ' + str(df['w_decay'][max_reward]))
print('TD_decay: ' + str(df['TD_decay'][max_reward]))

print('Min Length:')
print('Directory: ' + df['dir'][min_length])
print('SOM_alpha: ' + str(df['SOM_alpha'][min_length]))
print('SOM_sigma: ' + str(df['SOM_sigma'][min_length]))
print('w_decay: ' + str(df['w_decay'][min_length]))
print('TD_decay: ' + str(df['TD_decay'][min_length]))

plt.figure()
plt.plot(rewards, lengths, 'ro')
plt.xlabel('Total Reward')
plt.ylabel('Total Steps')
plt.savefig('Plots/ParamSweep.png')
plt.close()

params = ['SOM_alpha', 'SOM_sigma', 'w_decay', 'TD_decay']

for param in params:

    fig, ax = plt.subplots(1)
    im = ax.scatter(rewards, lengths, marker='o', c=df[param].tolist(),
                    cmap=plt.cm.get_cmap('inferno'), norm=colors.LogNorm())
    fig.colorbar(im, ax=ax)
    plt.xlabel('Total Reward')
    plt.ylabel('Total Steps')
    plt.suptitle(param)
    plt.savefig('Plots/ParamSweep_' + param + '.png')
    plt.close()



