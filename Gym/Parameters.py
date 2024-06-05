from Gym.Enums.Enums import EnvType, AgentType

env_params = {'env': EnvType.MountainCarContinuous,
              'num_trials': 1000,
              'max_steps': 1000,
              'num_repeats': 50,
              'print_freq': 100,
              'explanation_freq': 1000,
              'num_test_trials': 20,
              }

agent_params = {'agent_type': AgentType.CTDL_A2C,
                'bSOM': True,
                'bLoad_Exp': True,
                'bShuffle_Exp': False,
                'exp_length': 20,
                'exp_thresh': .5,
                'SOM_alpha': 1,
                'SOM_sigma': 1,
                'SOM_sigma_const': .001,
                'Q_alpha': .9,
                'w_decay': .1,
                'TD_decay': 100,
                'SOM_size': 8,
                'e_trials': 200
                }
