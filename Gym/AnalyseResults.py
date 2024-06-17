from Gym.Functions.Parsers import ParseIntoDataframes
from Gym.Functions.Plotters import PlotComparisons


dirs_to_compare = ['A2CExplanation', 'A2C']

data_frames = ParseIntoDataframes(dirs_to_compare)

PlotComparisons(data_frames, dirs_to_compare, num_trials_list=[50, 1000])
