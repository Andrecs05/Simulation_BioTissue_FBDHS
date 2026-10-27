# crowding_run

Crowding sweep: N LAMMPS cells at several mitochondrial volume fractions, then one simulated FB-DHS
recording per crowding level, to see how the knee (and D) changes with crowding.

## Files

| File | What it does |
|---|---|
| `make_runs.py` | Writes `runs/<level>/cell_<k>/` (cell.data, in.cell) for every level and `runs/manifest.csv` |
| `run_lammps.sh` | Slurm array, one LAMMPS run per manifest row |
| `compress_dumps.py` | LAMMPS output of one run → `traj.npz` |
| `compress.sh` | Slurm array, one compression per manifest row |
| `optics_run.py` | Optics of one crowding level, frames in parallel (`frames`), then PSD, knee and D (`gather`) |
| `optics.sh` | Slurm array, one task per crowding level |
| `gather.sh` | Slurm job that runs `optics_run.py gather` |
| `submit_all.sh` | Submits the whole chain, each step waiting for the one before |
| `common/` | `utilities.py`, `geometry.py` from `DPD tissue/code` (used by make_runs.py) |
| `optics_code/` | Copies of `DHS instrument/code/*.py` and `DPD tissue/code/tile_sample.py` (used by optics_run.py) |

## LAMMPS runs

`make_runs.py` builds a HeLa-sized cell: a 13 µm membrane shell, an 8 µm nucleus and mitochondria as
0.5 × 2 µm capsules pointing in random directions, in a 56 r_c (14 µm) box, about 600k–680k atoms. The
sources for each size are in the comments next to the values: Fujioka 2006, Luciani 2001, Maul & Deaven 1977,
Baffou 2014, Roskams & Rodgers, Phillips, Posakony 1977 and Weibel 1969, all via BioNumbers.
Levels are mitochondrial volume fractions of the cytoplasm, `PHI_LEVELS = [0.05, 0.10, 0.15, 0.20]`
(123, 245, 368, 490 mitochondria), named `phi05 ... phi20`, with `RUNS_PER_LEVEL = 4` cells each (16 runs).
HeLa is about 10%, and liver cells are about 22% of the whole cell. Each run writes, every 5 s of simulated
time (one hologram):

- `com_org.dat`: centre of mass of every organelle (row 1 nucleus, then the mitochondria)
- `com_mem.dat`: centre of mass of the membrane
- `dump.membrane`: membrane beads, every 20 frames, only to look at the cell

1860 frames per run: 60 of equilibration, 20 delay slots of 20 frames for the tiling, 1400 recorded.

Notes from the DPD side:
- Placing the mitochondria takes about 20 min per `phi20` cell and under a minute for the others, so
  `make_runs.py` takes about 1.5 h in all. Run it on Apolo in an interactive session rather than on the
  login node:
  `srun --partition=longjobs --time=03:00:00 --mem=8G --pty python make_runs.py`
  20% is close to the limit for random placement. Every size, level and time setting is a variable at the
  top of the file.
- TAU_S = 3.0 s for every run (167 steps per frame, 310,620 steps per run). It only maps the fluid's time
  scale, so the slowdown from crowding comes out of the simulation.
- The nucleus is a single rigid body under its own `fix rigid`, and the mitochondria use `fix rigid/small`.
  LAMMPS warns "More than one fix rigid", which is harmless because they act on different groups.
- `run_lammps.sh` runs `~/lammps/build/lmp` (your own build with DPD-BASIC) on one 32-core node per run.
  A 167-step test of a `phi15` cell (665k atoms) ran cleanly. The estimate is 4 to 6 h per run on Apolo,
  not measured yet, with a 2-day limit set. The 16 runs go in two rounds on the 9 longjobs nodes.
- Checked end to end on a short test run: make_runs.py → LAMMPS → compress_dumps.py gives a traj.npz
  with nucleus first and then the mitochondria.

## Compression

`compress_dumps.py` turns each run folder into `traj.npz` with the same keys as `compress_dump` in
`DPD tissue/code/utilities.py`: `steps`, `com` (organelle centres relative to the membrane centre, r_c),
`org_type`, `type_names`, plus `mem` and `mem_steps` from `dump.membrane`. Organelle names come from the
Masses comments of `cell.data`. It also reads the older runs that only have `dump.cell`.

```bash
python compress_dumps.py --manifest runs/manifest.csv            # every run
python compress_dumps.py runs/phi10/cell_0                       # one folder
```

## Optics

For each level, `optics_run.py` tiles the 1 mm sample with that level's cells (`tile_sample.build_layout`,
tile = box size read from `cell.data`) and simulates the 1400 holograms the same way as cells 11–21 of
`per_surface_analysis.ipynb`:

illumination (source → M1 → sample) → point scatterers with phase q·r → `fresnel_fft` 2f → objective
(`lens_transmitance`) → `fresnel_fft` 2f → field stop → `camera_hologram` → +1 sideband.

At 11 wavelengths it keeps the broadband sideband S+ = Σ w_j S_j inside a stadium window that follows the
carrier from 934 to 780 nm (radius 1.3·2R_IP/(λ f2)). `gather` then runs `field_spectrum`, `knee_frequency`
and D = π f_k / ⟨|q|²⟩.

Parallelism: holograms are independent, so each level is one array task with 32 processes, each
simulating whole frames (all wavelengths). On my 4-core test, one frame with 9 wavelengths took about
5.5 s on one core, so a level of 1400 frames is about 2 core-hours for the old 6 µm cells (11 wavelengths: about 2.5). The new
cells have up to 2.7 times more organelles per mm² (at phi20), so expect 5 to 10 minutes per level
on a full node.

Outputs, per level: `results/<level>/S.npy` (1400 × n_px complex64, about 150 MB) and `spectrum.npz` (f, P,
knee, D). For all levels: `results/summary.csv` (level, phi, n_mito, knee, D) and `results/spectra.png`.

### Field sampling

The object/image grid uses 4 µm pixels (512 × 512 = 2 mm), and each scatterer is shared bilinearly among its
4 nearest pixels instead of being snapped to one. The notebook uses 2 µm and snapping.

I compared 40 holograms of the same trajectories at several pixel sizes against a 0.5 µm reference
(camera sideband, 850 nm):

| pixel | placement | same field as 0.5 µm (ρ) | max error in g(τ) | s / frame / λ |
|---|---|---|---|---|
| 1 µm | snapped | 0.985 | 0.002 | 2.75 |
| 2 µm | snapped (notebook) | 0.956 | 0.004 | 0.67 |
| 3 µm | snapped | 0.906 | 0.017 | 0.56 |
| 4 µm | snapped | 0.853 | 0.008 | 0.42 |
| 2 µm | bilinear | 0.997 | 0.002 | 0.85 |
| **4 µm** | **bilinear** | **0.993** | **0.004** | **0.46** |
| 5 µm | bilinear | 0.985 | 0.004 | 0.41 |

- ρ is the overlap |⟨S, S_ref⟩| / (‖S‖‖S_ref‖) of each hologram's sideband with the reference.
- g(τ) is the temporal correlation of S+ over 1 to 20 frames. g(τ) is what sets the knee.

Snapping is what spoils the field. An organelle that moves jumps one whole pixel at a time, so its image
jumps too. Bilinear placement moves the image smoothly.

The upper limit is the objective. The grid at the lens is λ·2f1/dx wide and must hold the 25.4 mm lens,
so dx ≤ λ f1 / R1 = 4.6 µm at the shortest wavelength (780 nm). Bigger pixels clip the pupil, which lowers
the NA. 4 µm is the largest pixel that keeps the lens whole at every wavelength. Below that, the camera band
(image frequencies up to about 0.07 µm⁻¹) is far from aliasing.

### Wavelengths

With 5, 9 and 17 wavelengths, g(τ) agreed to 0.004 or better (30 frames). 11 is the default because the new
cells are about 13 µm deep. That gives round-trip path differences of about 35 µm, and 11 samples keep the
spectral revival distance (2π/δk ≈ 47 µm) beyond that (9 would give 38 µm, too close). Change it with `--n-k`. `--n-k 1` gives the
monochromatic 850 nm case.

## On Apolo

1. Copy the folder (PowerShell, from its parent folder on your PC, VPN on):
   ```powershell
   scp -r crowding_run ugr-afcontrerm@apolo-3.eafit.edu.co:~/DPD_runs/
   ```
2. Once, add the Python packages the optics needs to the environment:
   ```bash
   conda activate lammps-mpi
   pip install numpy matplotlib miepython
   ```
3. Generate the runs and submit everything:
   ```bash
   cd ~/DPD_runs/crowding_run
   srun --partition=longjobs --time=03:00:00 --mem=8G --pty python make_runs.py   # about 1.5 h, mostly the phi20 placement
   bash submit_all.sh                # or: bash submit_all.sh compress / bash submit_all.sh optics
   squeue -u $USER
   ```
   `submit_all.sh` sizes the arrays from `runs/manifest.csv`. Each compression starts as soon as its own
   LAMMPS run finishes (`aftercorr`). The optics starts when every compression is done, and `gather` starts
   after the optics.
4. Download the results:
   ```powershell
   scp -r ugr-afcontrerm@apolo-3.eafit.edu.co:~/DPD_runs/crowding_run/results .
   ```
   For only the summary, download `results/summary.csv` and `results/spectra.png`.

Logs go to `logs/<job>-<id>_<task>.out`. If an optics task dies, resubmit only that task, for example
`sbatch --array=3 optics.sh` for the fourth level. Finished chunks are skipped. Then run `sbatch gather.sh`.

To test before the full run, simulate 64 frames of the first level in an interactive session:
```bash
srun --partition=longjobs --cpus-per-task=32 --mem=32G --pty bash
conda activate lammps-mpi
python optics_run.py frames --task 0 --max-frames 64 --results results_test
python optics_run.py gather --results results_test
```
