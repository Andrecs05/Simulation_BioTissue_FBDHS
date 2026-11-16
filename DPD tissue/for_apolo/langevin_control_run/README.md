# langevin_control_run

Control for `sphere_crowding_run`: the same spheres, boxes, crowding levels and starting positions, with no
DPD fluid. Each sphere is one particle with Langevin friction, so on its own it diffuses with a set D and no
memory. The spheres repel each other as hard spheres, and there are no hydrodynamic interactions. The optics
should then give a knee at q²D/π for a dilute box. Comparing with the DPD sweep isolates what hydrodynamics
(and the DPD fluid) add.

## Files

All the files are copies of `sphere_crowding_run` except `make_runs.py`. Compress, optics, gather and the MSD
plot run unchanged, because the outputs (`box.data`, `com_org.dat`, `msd_org.dat`) have the same format.

| File | What changes from sphere_crowding_run |
|---|---|
| `make_runs.py` | One particle per sphere, WCA repulsion (contact at 0.88 µm), `fix langevin`. Same placement seeds, so box k starts where its DPD twin starts |
| `run_lammps.sh` | 2 cores, 4 GB, 2 h per run |
| `submit_all.sh` | All 20 runs at once (40 CPUs), no rounds |
| job names | Start with `lc` |

## Model

- Box 14 µm, periodic; spheres 0.88 µm across. Levels phi05, phi10, phi15, phi20 and phi30, 4 boxes each, 385–2307 particles.
- D = 1e-4 µm²/s for a free sphere (0.0048 r_c²/τ with TAU_S = 3 s). That puts the dilute knee at
  q²D/π ≈ 12 mHz, the same as the DPD phi05 knee, so both sweeps start at the same place.
  It is also close to a 0.88 µm sphere in a liquid of about 5 Pa·s.
- Mass 10, damping time 0.048 τ (0.14 s), much shorter than one frame, so the motion is diffusive at every
  lag the camera sees. DT = 0.005 τ, 333 steps per frame (4.995 s), 619,380 steps per run.
- WCA pair: `lj/cut` with σ = 0.88 µm and cutoff 2^(1/6) σ. It is purely repulsive, so there is no
  effective attraction (the suspected cause of the clustering in the DPD sweep).

## Local test

All 20 full runs ran here (2 cores each) in 15 s (phi05) to 85 s (phi30), and compress, the MSD plot and 8
optics frames ran unchanged on the output. On Apolo the LAMMPS step takes a few minutes in all, and the optics
is what takes time (up to about 40 min per level, as in sphere_crowding_run).

`plot_msd_sections.py` on those runs:

| level | α per section (5–20, 20–76, 76–296, 296–1150, 1150–4500 s) | D at 4496 s (µm²/s) |
|---|---|---|
| phi05 | 1.01, 0.99, 0.98, 0.99, 0.99 | 9.2e-5 |
| phi10 | 1.01, 0.99, 0.97, 0.96, 0.97 | 8.4e-5 |
| phi15 | 1.00, 0.97, 0.95, 0.94, 0.96 | 7.4e-5 |
| phi20 | 0.99, 0.95, 0.93, 0.92, 0.95 | 6.7e-5 |
| phi30 | 0.97, 0.90, 0.86, 0.87, 0.94 | 5.0e-5 |

There is no inertia artifact (α = 1 at short lags), and collisions alone slow long-time diffusion by about 1.8× from
phi05 to phi30. That is what hard spheres without hydrodynamics should do. At the knee time (10–20 s) the levels
differ by less than about 15%, so expect knees of roughly 10–12 mHz at every level, against 12.2 → 5.7 mHz in the DPD sweep.

## On Apolo

```powershell
scp -r langevin_control_run ugr-afcontrerm@apolo-3.eafit.edu.co:~/DPD_runs/
```
```bash
cd ~/DPD_runs/langevin_control_run
conda activate lammps-mpi
srun --partition=longjobs --time=00:30:00 --mem=4G --pty python make_runs.py   # about 2 min
bash submit_all.sh
squeue -u $USER
```
When gather has finished, run the MSD analysis as a batch job (it keeps running if you close the PC):
```bash
sbatch --partition=longjobs --time=01:00:00 --mem=16G --output=logs/msd-%j.out --wrap 'module load miniconda3; eval "$(conda shell.bash hook)"; conda activate lammps-mpi; MPLBACKEND=Agg python plot_msd_sections.py'
```
Results go to `results/summary.csv`, `results/spectra.png` and `results/msd_sections.png`, the same files as
sphere_crowding_run.
