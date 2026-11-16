# active_cell_run

A living cell on the instrument. It uses the crowding_run cell (13 µm, 8 µm nucleus, 10% mitochondria, which is
247 of them), but with overdamped Langevin mitochondria instead of DPD. The mitochondria also swim, which is the
simplest model of motor-driven transport. The sweep is over activity, with and without a cytoskeleton cage. The
question is how the knee and the moving fraction of S+ change when a cell is active, ATP-depleted or caged.

## Model

- **Cell:** the membrane (sphere of 13 µm) and the nucleus (8 µm) are walls (`fix wall/region`, WCA). The nucleus
  is also a static scatterer in the optics (radius 4 µm, n = 1.39), as in crowding_run, so it adds a static part to
  every hologram, as in a real cell.
- **Mitochondria:** spheres of 0.88 µm (the volume of a 0.5 × 2 µm mitochondrion), WCA contacts between them.
- **Thermal motion:** `fix brownian/sphere`, overdamped, so there is no inertia. D0 = 1e-4 µm²/s, as in the
  Langevin runs.
- **Activity:** `fix propel/self dipole`. Each mitochondrion swims at v0 along its own direction, and the direction
  turns by rotational diffusion with a persistence time of 30 s. This is an active Brownian particle, with runs of
  about 30 s × v0 that change direction. v0 = 0 is an ATP-depleted cell. Long-time D = D0 + v0² × 30 s / 3, so
  v0 = 10 nm/s gives 1.1e-3 µm²/s, 11× the thermal value.
- **Cage:** `fix spring/self` (as tether_run), δ = 30 nm rms per axis. This is the cytoskeleton holding each
  mitochondrion near its place. The swimming force can pull it out of the cage, by v0 δ²/D0 = 90 nm at 10 nm/s.
- **Levels:** `V_LEVELS = [0, 2, 5, 10, 20]` nm/s × `CAGE_LEVELS = [0, 0.03]` µm, named `v10_free`, `v05_c030nm`,
  and so on. 4 cells each (the same 4 cells at every level), 40 runs.
- **Time:** as the sphere runs. TAU_S = 3 s, DT = 0.02, 83 steps per frame, 154,380 steps. It writes
  `com_org.dat` and `msd_org.dat` (row 1 is the nucleus).

The instrument scale: at |q| ≈ 19.3 µm⁻¹, a speed of 1/(q × 5 s) ≈ 10 nm/s moves a mitochondrion by 1/q
in one frame. So the useful speeds are 1–30 nm/s. Real motor runs are faster (0.1–1 µm/s) but short and
intermittent, and the net, averaged motion is what this models. Faster speeds put the knee above Nyquist.

Short checks here (one cell, 1000 s): v0 = 0 gives D = 1.0e-4 µm²/s at 5 s. v0 = 10 nm/s gives D_eff rising from
1.6e-4 (5 s) to 7.3e-4 (300 s), as the active-particle MSD predicts. The 30 nm cage keeps D_eff falling with lag.

## Files

Copies of `d_sweep_run`, except:

| File | What changes |
|---|---|
| `make_runs.py` | New: cell, walls, active Brownian mitochondria, cage. `box.data` is `atom_style hybrid sphere dipole bond` |
| `compress_dumps.py` | Reads the hybrid data file (molecule id in the last column) |
| `plot_msd_sections.py` | Leaves the nucleus out |
| `plot_knee_vs_activity.py` | New. Knee and moving fraction against v0, free and caged. The prediction comes from a numpy simulation of the same active particles without walls → `results/knee_vs_activity.png` (about 3 min) |
| `run_lammps.sh` | 1 core per run, so all 40 runs go at once |
| job names | Start with `ac` |

The optics is the shared copy in `../shared_optics` (the `nucleus` and `mitochondrion` amplitudes are already there).

## Local test

All 40 runs (about 2 s each), compress, the optics at 1 wavelength (850 nm), gather and
`plot_knee_vs_activity.py` ran here. The outputs are in `local_test_1wavelength/`. The Apolo run, with 11
wavelengths, replaces them.

| v0 (nm/s) | knee, no cage (mHz) | predicted | moving fraction | knee, 30 nm cage (mHz) | predicted | moving fraction |
|---|---|---|---|---|---|---|
| 0 | 11.7 | 11.8 | 0.996 | 22.3 | 21.9 | 0.54 |
| 2 | 12.2 | 12.7 | 0.996 | 19.7 | 19.2 | 0.57 |
| 5 | 15.5 | 16.9 | 0.997 | 13.1 | 13.4 | 0.70 |
| 10 | 23.9 | 26.2 | 0.998 | 11.1 | 10.9 | 0.91 |
| 20 | 47.9 | 57.9 | 0.999 | 17.1 | 18.3 | 1.00 |

- **No cage:** activity raises the knee, 4× from 0 to 20 nm/s, and the spectrum gets steeper than a Lorentzian
  (ballistic runs). The cell sits 8–17% below the wall-free prediction at high v0, because the membrane, the
  nucleus and the other mitochondria interrupt the runs.
- **30 nm cage:** the knee moves the opposite way at first. It falls from 22 to 11 mHz as activity pulls the
  mitochondria out of their cages, then rises again. The knee alone is ambiguous here too: an ATP-depleted caged
  cell (22 mHz) and an active free cell (about 8 nm/s) give the same knee.
- **Moving fraction:** in the caged cell it rises steadily with activity, from 0.54 (thermal) to 1 (20 nm/s), so
  it reads activity directly where the knee does not. Without a cage it is always about 1.
- **The nucleus:** its static field barely lowers the moving fraction (0.996 with v0 = 0 and no cage). The
  slow moving mitochondria in front of it are enough to scramble it.

## On Apolo

### 1. Add two packages to LAMMPS (once)

Your build has MOLECULE, RIGID and DPD-BASIC. This run also needs BROWNIAN and DIPOLE. Check first:
```bash
~/lammps/build/lmp -h | grep -iE "brownian|propel"
```
If nothing prints, rebuild (about 10 min; the old binary keeps working for the other runs until `make` replaces it):
```bash
srun --partition=longjobs --time=01:00:00 --cpus-per-task=16 --mem=16G --pty bash
conda activate lammps-mpi
cd ~/lammps/build
cmake -D PKG_BROWNIAN=on -D PKG_DIPOLE=on .
make -j 16
./lmp -h | grep -iE "brownian|propel"      # should list brownian/sphere and propel/self
exit
```

### 2. Run

```powershell
scp -r active_cell_run ugr-afcontrerm@apolo-3.eafit.edu.co:~/DPD_runs/
```
```bash
cd ~/DPD_runs/active_cell_run
conda activate lammps-mpi
python make_runs.py          # seconds, fine on the login node
bash submit_all.sh
squeue -u $USER
```
LAMMPS takes about 10 s per run. The optics is about 7 min per level on 32 cores, 3 levels at a time, so about
30 min in all. When gather has finished:
```bash
MPLBACKEND=Agg python plot_knee_vs_activity.py
sbatch --partition=longjobs --time=01:00:00 --mem=16G --output=logs/msd-%j.out --wrap 'module load miniconda3; eval "$(conda shell.bash hook)"; conda activate lammps-mpi; MPLBACKEND=Agg python plot_msd_sections.py'
```
Download `results/summary.csv`, `results/spectra.png`, `results/knee_vs_activity.png` and `results/msd_sections.png`.
