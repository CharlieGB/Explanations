import Utilities
from GridWorld.Functions.RandomSeedSweep import RunRandomSeedSweep
from GridWorld.Functions.MazeTypeSweep import RunMazeTypeSweep
from GridWorld.Functions.RevaluationSweep import RunRevaluationSweep
from GridWorld.Functions.HumanMazeSweep import HumanMazeSweep

Utilities.InitialiseGPU()

HumanMazeSweep()
#RunMazeTypeSweep()