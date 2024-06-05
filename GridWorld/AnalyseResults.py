from GridWorld.Functions.Parsers import ParseIntoDataframes
from GridWorld.Functions.Plotters import PlotComparisons, PlotPairwiseComparison


dirs_to_compare = ['Maze_Type_Explanation', 'Maze_Type_No_Explanation']

data_frames = ParseIntoDataframes(dirs_to_compare)


explanation_ind = 0 if data_frames[0]['chosen_explanation'].values[0] != -1 else 1
no_explanation_ind = 0 if explanation_ind == 1 else 1
explanation_df = data_frames[explanation_ind]
no_explanation_df = data_frames[no_explanation_ind]

data_frames = [no_explanation_df]
names = ['No Explanation', 'Explanation 200', 'Explanation 400',
         'Explanation 600', 'Explanation 800', 'Explanation 1000']

for name, group_df in explanation_df.groupby('chosen_explanation'):
    data_frames.append(group_df)

PlotComparisons('type', data_frames, names)
# PlotPairwiseComparison(data_frames[0], data_frames[1], names)
