import os
import gym
import numpy as np

from datetime import datetime
from Utilities import RecordSettings
from Gym.Enums.Enums import EnvType, AgentType


def Run(env_params, agent_params):

    results_dir = CreateResultsDirectory()

    # Setup envrionment
    if (env_params['env'] == EnvType.CartPole):
        env = gym.make('CartPole-v1')
    elif (env_params['env'] == EnvType.MountainCarContinuous):
        env = gym.make('MountainCarContinuous-v0')

    env_params['env_obj'] = env
    env_params['state_mins'] = env.observation_space.low
    env_params['state_maxs'] = env.observation_space.high
    env_params['state_dim'] = env.observation_space.shape[0]

    if(isinstance(env.action_space, gym.spaces.Box)):
        env_params['action_maxs'] = env.action_space.high
        env_params['action_mins'] = env.action_space.low
    else:
        env_params['num_actions'] = env.action_space.n

    # Setup agent
    if(agent_params['agent_type'] == AgentType.CTDL):
        from Gym.Agents.CTDL.Agent import Agent
    elif(agent_params['agent_type'] == AgentType.DQN):
        from Gym.Agents.DQN.Agent import Agent
    elif (agent_params['agent_type'] == AgentType.A2C):
        from Gym.Agents.A2C.Agent import Agent
    elif (agent_params['agent_type'] == AgentType.CTDL_A2C):
        from Gym.Agents.CTDL_A2C.Agent import Agent

    agent = Agent(results_dir, env_params, agent_params)

    # Record settings
    RecordSettings(results_dir, env_params, agent_params)

    # Run
    RunEnv(agent, env, env_params, agent_params)

    return


def RunEnv(agent, env, env_params, agent_params):

    trial = 1
    reward = 0
    bTrial_over = False
    state = env.reset()
    ti = 0

    print('Starting Trial ' + str(trial) + '...')
    if agent_params['bSOM']:
        agent.PlotSOMResults(trial, env_params['env'])
    while trial <= env_params['num_trials']:

        ti += 1

        action = agent.Update(reward, state, bTrial_over, False)
        state, reward, bTrial_over, info = env.step(action)
        np.clip(reward, -1, 1)
        #print(reward)

        if(ti % env_params['max_steps'] == 0):
            bTrial_over = True

        if (bTrial_over):

            if(trial % env_params['explanation_freq'] == 0) and agent_params['bSOM']:

                # Save reward for resuming training
                final_reward = np.copy(reward)

                for test_trial in range(env_params['num_test_trials']):
                    # Freeze learning and do a test run
                    print('Starting Test Trial ' + str(test_trial) + '...')

                    reward = 0
                    state = env.reset()
                    bTrial_over = False

                    while not bTrial_over:
                        action = agent.Update(reward, state, bTrial_over, True)
                        state, reward, bTrial_over, info = env.step(action)

                    agent.RecordTestResults(trial, test_trial)

                agent.SaveTestResults()

                reward = final_reward
                bTrial_over = True

            ti = 0
            trial += 1
            state = env.reset()
            print('\nStarting Trial ' + str(trial) + '...')

            if (trial % env_params['print_freq'] == 0) and agent_params['bSOM']:
                agent.PlotSOMResults(trial, env_params['env'])

    agent.PlotResults(env_params['env'])
    env.close()

    return


def CreateResultsDirectory():
    date_time = str(datetime.now())
    date_time = date_time.replace(" ", "_")
    date_time = date_time.replace(".", "_")
    date_time = date_time.replace("-", "_")
    date_time = date_time.replace(":", "_")
    # Make the results directory
    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    results_dir = dir_path + '/../Results/' + date_time + '/'
    os.mkdir(results_dir)
    return results_dir



