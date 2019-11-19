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
ml py-numpy/1.17.2_py36
ml py-scikit-learn/0.19.1_py36
ml py-pandas/0.23.0_py36
ml py-scipy/1.1.0_py36

python3 connectome_experiments.py 
