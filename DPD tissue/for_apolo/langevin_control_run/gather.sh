#!/bin/bash
#SBATCH --job-name=lcgather
#SBATCH --partition=longjobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

##### ENVIRONMENT #####
module purge
eval "$(conda shell.bash hook)"
conda activate lammps-mpi

##### JOB COMMANDS #####
# Joins the chunks of every level, PSD, knee and D -> results/summary.csv, results/spectra.png
cd $SLURM_SUBMIT_DIR
python optics_run.py gather
