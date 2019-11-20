#!/bin/bash
#
#SBATCH --job-name=khan.job
#SBATCH --output=/scratch/users/cdonnat/convex_clustering/HC_dev/experiments/logs/khan_%A.out
#SBATCH --error=/scratch/users/cdonnat/convex_clustering/HC_dev/experiments/logs/khan_%A.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cdonnat@stanford.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G

HOMEDIR=/scratch/users/cdonnat/convex_clustering/HC_dev
JOBDIR=/scratch/users/cdonnat/convex_clustering/HC_dev/experiments/logs

cd ${HOMEDIR}
TYPE_LAP=$1
ALGO=$2
for ALPHA in 0 0.05 0.25 0.5 0.75 0.95 1; do
    for run in 0 1; do
       sbatch -p stat,hns,normal  bash_scripts/khan.sh ${run} ${TYPE_LAP} ${ALPHA} ${ALGO}
       echo "Im done with job ${ALPHA} and run ${run}"
     done
done                                      