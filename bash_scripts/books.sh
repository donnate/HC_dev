#!/bin/bash
#
#SBATCH --job-name=books_submit$1_$2
#SBATCH --output=/scratch/users/cdonnat/convex_clustering/HC_dev/experiments/logs/books$1_$2.out
#SBATCH --error=/scratch/users/cdonnat/convex_clustering/HC_dev/experiments/logs/books$1_$2.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
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
 
# execute script
cd $SCRATCH/convex_clustering/HC_dev/experiments
FILENAME=books_$2_lap_$1.pkl
python3 books_experiment.py -a $2 -type_lap $1 -savefile ${FILENAME} 
