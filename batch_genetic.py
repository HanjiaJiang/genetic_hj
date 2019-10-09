import os
import numpy as np


def batch_genetic(conn_maps, g, map_ids):
    simname = 'INs_genetic'

    # where your python code for the microcircuit model resides
    workingdir = os.getcwd()
    # output base directory
    output_dir = workingdir + '/output/'

    # python file to be executed by the queue
    pyf_name = 'run_network.py'

    # job description file
    jdf_name = simname + '_batch.jdf'

    for i, map in enumerate(conn_maps):
        # print(map)
        # output directory for this parameter combination
        this_output_dir = 'g={0:02d}_ind={1:02d}'.format(g, map_ids[i])
        full_output_dir = output_dir + this_output_dir

        # create directory if it doesn't exist yet
        if this_output_dir not in os.listdir(output_dir):
            os.system('mkdir ' + full_output_dir)
            os.system('mkdir ' + full_output_dir + '/data')

        os.chdir(workingdir)

        # copy all the relevant files to the output directory
        os.system('cp run_network.py ' + full_output_dir)
        os.system('cp network.py ' + full_output_dir)
        os.system('cp network_params.py ' + full_output_dir)
        os.system('cp sim_params.py ' + full_output_dir)
        os.system('cp helpers.py ' + full_output_dir)
        os.system('cp stimulus_params.py ' + full_output_dir)
        os.system('cp conn.py ' + full_output_dir)
        os.system('cp functions.py ' + full_output_dir)
        os.system('cp scan_params.py ' + full_output_dir)
        os.system('cp microcircuit_tools.py ' + full_output_dir)
        np.save(os.path.join(full_output_dir, 'conn_probs.npy'), map)

        os.chdir(full_output_dir)

        this_pyf_name = full_output_dir + '/' + pyf_name

        # write job description file
        f = open(full_output_dir + '/' + jdf_name, 'w')
        f.write('#!/bin/bash \n')
        # set job name
        f.write('#SBATCH --job-name ' + simname + '_' + this_output_dir + '\n')
        # output file to send standard output stream to
        f.write('#SBATCH -o ./data/outfile.txt' + '\n')
        # send error stream to same file
        f.write('#SBATCH -e ./data/errorfile.txt' + '\n')
        # request a total of nodes*processes_per_node for this job
        f.write('#SBATCH --ntasks=1\n')
        f.write('#SBATCH --cpus-per-task=24\n')
        f.write('#SBATCH --ntasks-per-node=1\n')
        # request processor time
        f.write('#SBATCH --time=04:00:00\n')
        f.write('source $HOME/.bashrc\n')
        f.write('conda activate nest-log\n')
        f.write(
            'source $HOME/opt/nest-lognormal/bin/nest_vars.sh\n')
        f.write('python %s\n' % this_pyf_name)
        f.close()

        # submit job
        os.system('sbatch ' + jdf_name)
    os.chdir(workingdir)
