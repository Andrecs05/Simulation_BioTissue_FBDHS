#!/bin/bash
#SBATCH --job-name=tdcompress
#SBATCH --partition=longjobs
#SBATCH --array=0-15                    # one task per row of runs/manifest.csv, same range as run_lammps.sh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

##### ENVIRONMENT #####
module purge
eval "$(conda shell.bash hook)"
conda activate lammps-mpi               # needs numpy

##### JOB COMMANDS #####
# Row SLURM_ARRAY_TASK_ID of the manifest -> runs/<level>/box_<k>/traj.npz
cd $SLURM_SUBMIT_DIR
echo "row $SLURM_ARRAY_TASK_ID on $(hostname), started $(date)"
python compress_dumps.py --manifest runs/manifest.csv --task $SLURM_ARRAY_TASK_ID
echo "Finished $(date)"
