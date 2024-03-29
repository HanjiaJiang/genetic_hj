#!/bin/bash
#SBATCH --job-name genetic_hj
#SBATCH --time 96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o ./output/outfile.txt
#SBATCH -e ./output/errorfile.txt
#SBATCH --ntasks-per-node=1
source $HOME/.bashrc
conda activate nest-log
python $PWD/genetic_fr.py

