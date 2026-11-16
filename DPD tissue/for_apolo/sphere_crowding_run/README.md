# sphere_crowding_run

Crowding sweep without a cell: spherical organelles in a periodic box of cytoplasm, no membrane and no
nucleus, at several volume fractions. Then one simulated FB-DHS recording per crowding level, to see how the
knee (and D) changes with crowding alone, and to compare with `crowding_run` (same organelles inside a cell).

Everything except the sample and the timestep is the same as `crowding_run`: DPD parameters, TAU_S, frame
cadence, recording length, tiling and optics. The timestep is 0.02 instead of 0.01 (see below).

## Files

| File | What it does | From crowding_run |
|---|---|---|
| `make_runs.py` | Writes `runs/<level>/box_<k>/` (box.data, in.box) for every level and `runs/manifest.csv` | new |
| `run_lammps.sh` | Slurm array, one LAMMPS run per manifest row | cores per run set by `submit_all.sh`, `in.box` |
| `compress_dumps.py` | com_org.dat of one run → `traj.npz` (centres relative to the box centre) | membrane parts removed |
| `compress.sh` | Slurm array, one compression per manifest row | same |
| `optics_run.py` | Optics of one crowding level (`frames`), then PSD, knee and D (`gather`) | `sphere` amplitude, `box.data`, faster bilinear placement |
| `optics.sh`, `gather.sh`, `submit_all.sh` | Optics array, gather job, whole chain | `submit_all.sh` also sizes the LAMMPS rounds (job names start with `sph`) |
| `common/`, `optics_code/` | Copies of the same modules | unchanged |

## LAMMPS runs

- Box: periodic cube of 56 r_c = 14 µm, the same size and depth as the crowding_run cells. It is full of
  cytoplasm (DPD, ρ = 3, a = 25, γ = 4.5), so each run is a piece of an endless crowded cytoplasm.
- Spheres: 0.88 µm across (r = 1.76 r_c, 160 beads, rigid bodies with `fix rigid/small`). They have the
  volume of a 0.5 × 2 µm mitochondrion (0.36 µm³), so they are the crowding_run mitochondria made round.
  They use the same mitochondrion Mie amplitude (radius 0.44 µm, n = 1.41) and the same pair coefficients
  (40 against cytoplasm, 30 between spheres) as before.
- Levels: volume fraction of the box, `PHI_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30]`, named
  `phi05 ... phi30` (385, 770, 1154, 1539, 2307 spheres), with `RUNS_PER_LEVEL = 4` boxes each (20 runs).
  The first four match crowding_run. Spheres pack more easily than capsules, so 30% also places (in about
  30 s, with a 0.1 r_c gap).
- 560k (phi05) to 740k (phi30) atoms per run.
- Time: TAU_S = 3.0 s, DT = 0.02, 83 steps per frame (4.98 s), 1860 frames (60 equilibration, 20 × 20 delay
  frames, 1400 recorded), 154,380 steps. Each run writes `com_org.dat` (centre of every sphere, every frame) and
  `msd_org.dat`. There is no bead dump.
- `make_runs.py` takes about 20 min in all (mostly removing the cytoplasm inside the spheres), so it can run in
  a short interactive job.
- Timestep: DT = 0.02 halves the steps per run. In the pilot (below) it kept the temperature at 1.01 and gave
  the same sphere D_eff as 0.01 within 5% at every lag (for example phi20 at 1.67 τ: 0.0045 at both).
- Cores: the Apolo QOS allows 96 CPUs per user. `submit_all.sh` picks the cores per run C (16, 18, 20, 24, 27
  or 32, one node at most) that gives the shortest total, ceil(N_runs / K) / C with K = 96 / C runs at a time,
  and throttles the array to K (`--array=0-19%K`), so a new run starts as soon as one ends. For 20 runs it
  picks 24 cores × 4 runs = 96 CPUs, 5 full rounds (16 cores × 6 would leave 64 CPUs idle in the last round).
  Use `MAX_CPUS=64 bash submit_all.sh` if the limit changes.
- Speed: crowding_run measured 6.2 h per 636k-atom run on 32 cores at DT = 0.01. A 200-step test of a full
  phi30 box (738k atoms, 2307 rigid bodies) ran cleanly here at 4.4 steps/s on 4 cores. With half the steps
  on 24 cores expect 2–5 h per run (not measured on Apolo), so about half a day to a day for the 20 runs.
  The temperature jumps to 2 in the first 50 steps (cytoplasm beads right at the
  sphere surfaces) and is back to 1.1 by step 200, well inside the 300 s equilibration.

### What a short pilot showed (6 µm box, 300 τ, DT = 0.01, on 2 cores)

| lag (τ) | 0.5 | 1 | 1.67 (one frame) | 2 | 5 | 10 |
|---|---|---|---|---|---|---|
| phi05 D_eff (r_c²/τ) | 0.0025 | 0.0039 | 0.0056 | 0.0061 | 0.0105 | 0.0143 |
| phi20 D_eff (r_c²/τ) | 0.0024 | 0.0034 | 0.0044 | 0.0047 | 0.0070 | 0.0087 |

- Free spheres move 2–3 times faster than the mitochondria inside the cell (about 0.004 r_c²/τ), so the
  field decorrelates in about 4 τ (12 s, 2–3 frames). Expect knees around 25–30 mHz, against 4–6 mHz in the
  crowding_run cells. That is still below the 100 mHz Nyquist frequency.
- D_eff keeps rising with lag at both levels. This is the DPD momentum memory already seen in the m1 tests. The
  knee time in τ does not depend on TAU_S, so changing TAU_S would only move the knee in Hz, not fix this.
  Expect spectra that are not single Lorentzians.
- Crowding slows the spheres by about 1.5× from phi05 to phi20 at the knee time, and the difference grows with
  lag.

## Optics

Same as crowding_run (see its README): the 1 mm sample is tiled with the boxes of one level, 4 µm pixels
with bilinear placement, 11 wavelengths, stadium sideband window, knee of the field PSD and D = π f_k / ⟨|q|²⟩.
The box centre goes to the tile centre and the z of each sphere is its depth in the box (−7 to 7 µm).

phi30 puts about 9 million spheres in the sample, 4.7 times more than the phi20 cells. The bilinear
placement uses `np.bincount` instead of `np.add.at` (same result, much faster). See the measured time and
memory per frame below.

Measured here on fake phi30 trajectories of the right size (4 boxes × 2307 spheres × 1860 frames): set-up
70 s, then about 52 s per frame per core with 11 wavelengths, so a phi30 level is about 40 min on 32 cores
(lower levels are faster). Each worker peaks at about 2.7 GB resident, most of it shared with the parent, so
`optics.sh` asks for 64 GB. The bincount placement gives exactly the same S+ as `np.add.at`.

## On Apolo

1. Copy the folder (PowerShell, from its parent folder on your PC, VPN on):
   ```powershell
   scp -r sphere_crowding_run ugr-afcontrerm@apolo-3.eafit.edu.co:~/DPD_runs/
   ```
2. Generate the runs and submit everything:
   ```bash
   cd ~/DPD_runs/sphere_crowding_run
   conda activate lammps-mpi
   srun --partition=longjobs --time=01:00:00 --mem=8G --pty python make_runs.py   # about 20 min
   bash submit_all.sh                # or: bash submit_all.sh compress / bash submit_all.sh optics
   squeue -u $USER
   ```
3. Download the results:
   ```powershell
   scp -r ugr-afcontrerm@apolo-3.eafit.edu.co:~/DPD_runs/sphere_crowding_run/results .
   ```
   For only the summary, download `results/summary.csv` and `results/spectra.png`.

Logs go to `logs/<job>-<id>_<task>.out`. If an optics task dies, resubmit only that task, for example
`sbatch --array=4 optics.sh` for phi30, then `sbatch gather.sh`.

To test the optics before the full run, once a level's runs are compressed:
```bash
srun --partition=longjobs --cpus-per-task=32 --mem=64G --pty bash
conda activate lammps-mpi
python optics_run.py frames --task 0 --max-frames 64 --results results_test
python optics_run.py gather --results results_test
```
