#!/bin/bash
#SBATCH --job-name=sphoptics
#SBATCH --partition=longjobs
#SBATCH --array=0-4                     # one task per crowding level x N_CHUNKS (5 levels, 1 chunk each)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32              # one full node: 32 frames at a time
#SBATCH --mem=64G                       # phi30 has ~9 million spheres in the sample; ~1 GB per worker
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

##### ENVIRONMENT #####
module purge
eval "$(conda shell.bash hook)"
conda activate lammps-mpi               # needs numpy, matplotlib and miepython
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

##### JOB COMMANDS #####
# Task t simulates frame chunk t % N_CHUNKS of level t / N_CHUNKS -> results/<level>/S_chunk_XX.npy
N_CHUNKS=1
cd $SLURM_SUBMIT_DIR
echo "task $SLURM_ARRAY_TASK_ID on $(hostname), $SLURM_CPUS_PER_TASK cores, started $(date)"
python optics_run.py frames --task $SLURM_ARRAY_TASK_ID --n-chunks $N_CHUNKS --workers $SLURM_CPUS_PER_TASK
echo "Finished $(date)"
