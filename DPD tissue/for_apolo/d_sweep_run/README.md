# d_sweep_run

D sweep: dilute spheres (phi = 5%, as phi05 of `langevin_control_run`) at 10 diffusion coefficients, from
1e-6 to 5e-3 µm²/s. The knee should follow q²D/π inside the instrument's range and fail outside it:
- above D ≈ 8e-4 µm²/s the knee passes the 100 mHz Nyquist frequency of the 0.2 Hz recording (aliasing);
- below D ≈ 1e-6 µm²/s the knee drops under 1/7000 s = 0.14 mHz, the frequency resolution of 1400 frames.

The result is the curve of measured D against input D, which shows the range of D the EAFIT instrument can measure.

## Files

Copies of `langevin_control_run`, except:

| File | What changes |
|---|---|
| `make_runs.py` | `D_LEVELS` instead of crowding levels, one φ (5%). The damping time is the same at every D (0.048 τ = 0.14 s) and the mass changes with D, so the motion is diffusive at every lag the camera sees. Box k uses the same placement seed as phi05 box k of the other sweeps. `D_in` is a column of the manifest and of `summary.csv` |
| `plot_knee_vs_D.py` | New: measured D against input D, and their ratio, with the Nyquist and resolution limits shaded → `results/knee_vs_D.png` |
| job names | Start with `ds` |

The optics is the shared copy in `../shared_optics`.

| level | D (µm²/s) | expected knee |
|---|---|---|
| d0_1e-06 | 1e-6 | 0.12 mHz, below the resolution |
| d1_3e-06 | 3e-6 | 0.36 mHz |
| d2_1e-05 | 1e-5 | 1.2 mHz |
| d3_3e-05 | 3e-5 | 3.6 mHz |
| d4_1e-04 | 1e-4 | 11.9 mHz (as langevin_control_run phi05) |
| d5_3e-04 | 3e-4 | 36 mHz |
| d6_6e-04 | 6e-4 | 71 mHz |
| d7_1e-03 | 1e-3 | 119 mHz, above Nyquist |
| d8_2e-03 | 2e-3 | 238 mHz, aliased |
| d9_5e-03 | 5e-3 | 595 mHz, aliased |

40 runs (10 D × 4 boxes), 385 spheres each, 2 cores per run, all at once (80 CPUs). LAMMPS takes a few minutes in all.
The optics is about 7 min per level on 32 cores (phi05 has few spheres); 3 levels run at a time under the CPU limit.

## Local test

LOCALTEST

## On Apolo

`shared_optics` must be in `~/DPD_runs/` (copy it once if you haven't).
```powershell
scp -r d_sweep_run ugr-afcontrerm@apolo-3.eafit.edu.co:~/DPD_runs/
```
```bash
cd ~/DPD_runs/d_sweep_run
conda activate lammps-mpi
srun --partition=longjobs --time=00:30:00 --mem=4G --pty python make_runs.py   # under a minute
bash submit_all.sh
```
When gather has finished:
```bash
MPLBACKEND=Agg python plot_knee_vs_D.py        # seconds, fine on the login node
```
It prints measured against input D for every level and writes `results/knee_vs_D.png`. The spectra are in
`results/spectra.png`, as for the other sweeps.
