import Utilities
from GridWorld.Functions.RandomSeedSweep import RunRandomSeedSweep
from GridWorld.Functions.MazeTypeSweep import RunMazeTypeSweep
from GridWorld.Functions.RevaluationSweep import RunRevaluationSweep

Utilities.InitialiseGPU()

RunMazeTypeSweep()