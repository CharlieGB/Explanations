from GridWorld.Enums.Enums import MazeType, AgentType


maze_params = {'type': MazeType.random,
               'width': 10,
               'height': 10,
               'num_rewards': 1,
               'num_trials': 1000,
               'random_seed': 0,
               'max_steps': 1000,
               'num_repeats': 20,
               'print_freq': 1000,
               'explanation_freq': 200,
               'num_test_trials': 1
               }

agent_params = {'agent_type': AgentType.CTDL,
                'bSOM': True,
                'bLoad_Exp': False,
                'bShuffle_Exp': False,
                'exp_length': 20,
                'exp_thresh': .5,
                'SOM_alpha': 1,#.1,
                'SOM_sigma': 1,
                'SOM_sigma_const': .001,
                'Q_alpha': .1,
                'w_decay': 1,#1,
                'TD_decay': 100,
                'SOM_size': 4,
                'e_trials': 200,
                }
