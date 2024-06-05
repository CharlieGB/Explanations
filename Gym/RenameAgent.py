import os
import numpy as np

dir = 'Temp'
new_name = 'CTDL_A2C_3'

folders = os.listdir('Results/' + dir)

for folder in folders:
    if (folder == '.DS_Store' or folder == '.keep'):
        pass
    else:
        files = os.listdir('Results/' + dir + '/' + folder)

        if ('.DS_Store' in files):
            files.remove('.DS_Store')

        file = open('Results/' + dir + '/' + folder + '/Settings.txt', 'r')
        lines = file.readlines()
        file.close()

        new_file = open('Results/' + dir + '/' + folder + '/Settings.txt', 'w')

        for line in lines:
            if 'agent_type' in line:
                line = 'agent_type: AgentType.' + new_name + '\n'
            new_file.write(line)

        new_file.close()