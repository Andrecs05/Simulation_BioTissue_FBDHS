# tether_dpd_run

The DPD twin of `tether_run`. The sample is the phi05 boxes of `sphere_crowding_run`, with the same placement and
DPD seeds, so box k is phi05 box k. The only change is that every sphere is held to its starting position by a
harmonic spring, at four cage sizes. The free reference is `sphere_crowding_run` phi05, which was already run.

The Langevin tether runs follow the simple caged-sphere prediction to within a few percent. This run asks whether a
sphere caged in a fluid with hydrodynamics (and DPD's momentum memory) does the same.

## Model

- Spring: `fix spring/self` on every bead of each sphere, with K_bead = kT / (δ² × 160 beads). The rigid body then
  feels K_body = kT / δ² on its centre, which gives an rms displacement δ per axis. It also gets a weak hold on its
  rotation, which the point-scatterer optics doesn't see.
- Check: in a 6 µm test box (DPD, 6000 steps), the 15 nm and 50 nm cages gave 14.8 and 48.6 nm rms.
- Cages: `CAGE_LEVELS = [0.015, 0.02, 0.03, 0.05]` µm, levels `c015nm ... c050nm`, 4 boxes each, 16 runs.
- Everything else is as in `sphere_crowding_run`: a = 25 / 40 / 30, γ = 4.5, DT = 0.02, 83 steps per frame,
  154,380 steps, and the same outputs (`com_org.dat`, `msd_org.dat`).

Prediction, using D = 1.03e-4 µm²/s (the D the free phi05 DPD knee gives) and τ_c = δ²/D:

| cage | 15 nm | 20 nm | 30 nm | 50 nm | free (measured) |
|---|---|---|---|---|---|
| knee (mHz) | ~75 | 44 | 22 | 12 | 12.2 |
| τ_c (s) | 2.2 | 3.9 | 8.7 | 24 | |
| moving fraction | 0.15 | 0.26 | 0.49 | 0.85 | ~1 |

In DPD the cage relaxes in 2–24 s. That is shorter than the vorticity time of the fluid around a sphere
(a²ρ/η ≈ 11 τ ≈ 30 s), so the sphere drags fluid momentum while it bounces in the cage. If hydrodynamic memory
matters at the scale the instrument sees, the DPD knees will fall below the prediction, most for the tightest
cages. If they land on it, the cage dominates and the fluid doesn't change what the instrument reads.

## Time (one day maximum)

- `make_runs.py`: about 15 min (filling the cytoplasm), so run it in a short interactive job.
- LAMMPS: `submit_all.sh` picks 24 cores × 4 runs at a time (96 CPUs), which is 4 rounds. phi05 is the cheapest
  sphere_crowding_run level, at about 3.5 h per run on 24 cores (estimated from crowding_run's 14 steps/s on 32
  cores for 636k atoms; the 20-run sphere sweep finished in under a day). That makes about 14 h for all 16 runs.
- Optics: 4 levels, about 10 min each on 32 cores, 3 at a time.
- Total: about 15 h.
- If the first runs take more than 5 h each (check `logs/tdrun-*.out`), set `RUNS_PER_LEVEL = 3` and regenerate
  before the next submission: 12 runs in 3 rounds.

## Files

Copies of `sphere_crowding_run`, except:

| File | What changes |
|---|---|
| `make_runs.py` | One φ (5%), `CAGE_LEVELS` instead of crowding levels, `fix spring/self` in `in.box`, seeds of phi05 box k. `cage_um` column in the manifest and `summary.csv` |
| `plot_knee_vs_cage.py` | From `tether_run`. Knee and moving fraction against δ. It shows the DPD points, the prediction for the free DPD D, the free phi05 run from `../sphere_crowding_run/results` and the Langevin D = 1e-4 points from `../tether_run/results` (each only if that folder is on Apolo) → `results/knee_vs_cage.png` |
| array sizes | 16 runs, 4 optics tasks |
| job names | Start with `td` |

The optics is the shared copy in `../shared_optics`.

## On Apolo

```powershell
scp -r tether_dpd_run ugr-afcontrerm@apolo-3.eafit.edu.co:~/DPD_runs/
```
```bash
cd ~/DPD_runs/tether_dpd_run
conda activate lammps-mpi
srun --partition=longjobs --time=01:00:00 --mem=8G --pty python make_runs.py   # about 15 min
bash submit_all.sh
squeue -u $USER
```
When gather has finished:
```bash
MPLBACKEND=Agg python plot_knee_vs_cage.py
sbatch --partition=longjobs --time=01:00:00 --mem=16G --output=logs/msd-%j.out --wrap 'module load miniconda3; eval "$(conda shell.bash hook)"; conda activate lammps-mpi; MPLBACKEND=Agg python plot_msd_sections.py'
```
Download `results/summary.csv`, `results/spectra.png`, `results/knee_vs_cage.png` and `results/msd_sections.png`.
