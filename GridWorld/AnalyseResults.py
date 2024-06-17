import  matplotlib.pyplot as plt

from GridWorld.Functions.Parsers import ParseIntoDataframes
from GridWorld.Functions.Plotters import PlotComparisons, PlotPairwiseComparison

# Time comparison

dirs_to_compare = ['Maze_Type_No_Explanation', 'Maze_Type_Explanation']
data_frames = ParseIntoDataframes(dirs_to_compare)

all_data_frames = [data_frames[0]]
names = ['No Explanation', 'Explanation 200', 'Explanation 400',
         'Explanation 600', 'Explanation 800', 'Explanation 1000']
linestyles = ['-', '--', '--', '--', '--', '--']
colors = plt.get_cmap('plasma', 5).colors.tolist()
colors = ['k'] + colors

for name, group_df in data_frames[1].groupby('chosen_explanation'):
    all_data_frames.append(group_df)

PlotComparisons('type', all_data_frames, names, colors, linestyles, '_time')
# PlotPairwiseComparison(data_frames[0], data_frames[1], names)

# Shuffled Comparison

dirs_to_compare = ['Maze_Type_No_Explanation', 'Maze_Type_Shuffled_Explanation', 'Maze_Type_Explanation']
data_frames = ParseIntoDataframes(dirs_to_compare)

all_data_frames = [data_frames[0], data_frames[1], data_frames[2][data_frames[2]["chosen_explanation"] == 4.0]]
names = ['No Explanation', 'Shuffled Explanation', 'Explanation']
linestyles = ['-', ':', '--']
colors = ['b', 'r', 'g']

PlotComparisons('type', all_data_frames, names, colors, linestyles, '_shuffled')
