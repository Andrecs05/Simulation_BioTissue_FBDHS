#!/bin/bash
#SBATCH --job-name=tdrun
#SBATCH --partition=longjobs
#SBATCH --array=0-15                    # one task per row of runs/manifest.csv (make_runs.py prints the range)
#SBATCH --nodes=1
#SBATCH --ntasks=24                     # MPI processes per run; submit_all.sh overrides it so every round fills the CPU limit
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

##### ENVIRONMENT #####
module purge
eval "$(conda shell.bash hook)"
conda activate lammps-mpi
export UCX_NET_DEVICES=mlx5_0:1         # silences the mlx4_0 warning
export OMP_NUM_THREADS=1

##### JOB COMMANDS #####
# row SLURM_ARRAY_TASK_ID of the manifest (line 1 is the header); column 2 is run_dir
RUN_DIR=$(awk -F, -v i=$SLURM_ARRAY_TASK_ID 'NR == i + 2 {print $2}' $SLURM_SUBMIT_DIR/runs/manifest.csv)
cd $SLURM_SUBMIT_DIR/$RUN_DIR
echo "$RUN_DIR on $(hostname), $SLURM_NTASKS tasks, started $(date)"
mpirun -np $SLURM_NTASKS ~/lammps/build/lmp -in in.box
echo "Finished $(date)"
