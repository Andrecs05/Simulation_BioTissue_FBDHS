#!/bin/bash
# Submits the whole chain, each step waiting for the one before:
#   LAMMPS runs -> compress each run -> optics per crowding level -> gather
# Usage (on Apolo, in sphere_crowding_run, after python make_runs.py):
#   bash submit_all.sh             everything
#   bash submit_all.sh compress    from the compression on (the LAMMPS runs are already done)
#   bash submit_all.sh optics      from the optics on (traj.npz files already there)
#   MAX_CPUS=64 bash submit_all.sh if the per-user CPU limit is different (default 96)
set -e
cd "$(dirname "$0")"
mkdir -p logs results

N_RUNS=$(($(wc -l < runs/manifest.csv) - 1))
N_LEVELS=$(awk -F, 'NR == 1 {for (i = 1; i <= NF; i++) if ($i == "level") c = i} NR > 1 {print $c}' runs/manifest.csv | sort -u | wc -l)
N_CHUNKS=$(grep -oP '^N_CHUNKS=\K[0-9]+' optics.sh)
FROM=${1:-lammps}
echo "$N_RUNS runs, $N_LEVELS levels, $N_CHUNKS chunks per level"

# Cores per LAMMPS run: Apolo's QOS caps the CPUs per user (MAX_CPUS), so the runs go in rounds of
# K = MAX_CPUS / C at once. Pick C (one node at most, counts that split a cube well) with the fewest
# rounds x time per run, i.e. the smallest ceil(N_RUNS / K) / C, so no round leaves cores idle for long.
# The array is throttled to K runs at a time (%K), so each round starts as the previous one ends.
MAX_CPUS=${MAX_CPUS:-96}
read C K R < <(awk -v n=$N_RUNS -v m=$MAX_CPUS 'BEGIN {
    best = 1e9
    split("16 18 20 24 27 32", cs, " ")
    for (i in cs) {
        c = cs[i]; k = int(m / c); if (k < 1) continue; if (k > n) k = n
        r = int((n + k - 1) / k); cost = r / c
        if (cost < best - 1e-9 || (cost < best + 1e-9 && c < bc)) { best = cost; bc = c; bk = k; br = r }
    }
    print bc, bk, br }')
echo "LAMMPS: $C cores per run, $K runs at a time ($((C * K)) of $MAX_CPUS CPUs), $R rounds"

dep=""
if [ "$FROM" = lammps ]; then
    j=$(sbatch --parsable --ntasks=$C --array=0-$((N_RUNS - 1))%$K run_lammps.sh)
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
