#!/bin/bash
# Submits the whole chain, each step waiting for the one before:
#   LAMMPS runs -> compress each run -> optics per level -> gather
# Usage (on Apolo, in tether_run, after python make_runs.py):
#   bash submit_all.sh             everything
#   bash submit_all.sh compress    from the compression on (the LAMMPS runs are already done)
#   bash submit_all.sh optics      from the optics on (traj.npz files already there)
set -e
cd "$(dirname "$0")"
mkdir -p logs results

N_RUNS=$(($(wc -l < runs/manifest.csv) - 1))
N_LEVELS=$(awk -F, 'NR == 1 {for (i = 1; i <= NF; i++) if ($i == "level") c = i} NR > 1 {print $c}' runs/manifest.csv | sort -u | wc -l)
N_CHUNKS=$(grep -oP '^N_CHUNKS=\K[0-9]+' optics.sh)
FROM=${1:-lammps}
echo "$N_RUNS runs, $N_LEVELS levels, $N_CHUNKS chunks per level"

# The Langevin runs are tiny (one particle per sphere), so all of them go at once with the cores of run_lammps.sh

dep=""
if [ "$FROM" = lammps ]; then
    j=$(sbatch --parsable --array=0-$((N_RUNS - 1)) run_lammps.sh)
    echo "lammps   $j"; dep="--dependency=aftercorr:$j"        # compress task i waits only for run i
fi
if [ "$FROM" = lammps ] || [ "$FROM" = compress ]; then
    j=$(sbatch --parsable $dep --array=0-$((N_RUNS - 1)) compress.sh)
    echo "compress $j"; dep="--dependency=afterok:$j"
fi
j=$(sbatch --parsable $dep --array=0-$((N_LEVELS * N_CHUNKS - 1)) optics.sh)
echo "optics   $j"
j=$(sbatch --parsable --dependency=afterok:$j gather.sh)
echo "gather   $j"
echo "watch with: squeue -u \$USER"
