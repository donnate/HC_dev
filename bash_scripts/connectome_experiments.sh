#!/bin/bash
#
#SBATCH --job-name=HC_brain
#SBATCH --output=/scratch/users/cdonnat/convex_clustering/connectome_%A.out
#SBATCH --error=/scratch/users/cdonnat/convex_clustering/connectome_%A.err
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cdonnat@stanford.edu
#SBATCH --partition=hns,stat,normal,owners
# load modules

ml python/3.6.1
python connectome_experiments.py 
