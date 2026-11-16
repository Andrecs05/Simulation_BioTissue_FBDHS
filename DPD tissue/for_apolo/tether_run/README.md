# tether_run

Tethered (caged) spheres. The sample is the dilute Langevin box of `d_sweep_run` (φ = 5%, 385 spheres of 0.88 µm,
same placement seeds), but every sphere is held to its starting position by a harmonic spring (`fix spring/self`).
This is the simplest model of a particle trapped in a gel or a protein network, such as a fat globule in the
cheese matrix. The question is what the instrument reads when the particle cannot diffuse away.

## Cage size

The spring acts at every distance, pulling the sphere back harder the farther it goes. There is no wall. The cage
size δ is the rms displacement the spring allows along one axis, δ² = kT / K, so the sphere spends 95% of the time
within ±2δ of its anchor. Inside the cage it moves with its free D, and it reaches the cage edge after
τ_c = δ² / D.

The length the optics compares δ with is 1/q ≈ 50 nm (|q| ≈ 19.3 µm⁻¹):

- **δ ≫ 1/q:** the field decorrelates before the sphere feels the spring, so the knee is q²D/π, as if free.
- **δ ≪ 1/q:** the phase of each sphere only wobbles, so only a fraction 1 − e^(−2q²δ²) ≈ 2q²δ² of the S+ signal
  moves. That part relaxes in τ_c, so the knee is about D / (2πδ²), **above** the free knee. A tight cage looks like
  fast diffusion with a weak signal.

The prediction for every level (from F(t) = exp(−q²δ²(1 − e^(−t/τ_c))), sampled every 5 s and fitted like gather):

| δ (nm) | 5 | 10 | 20 | 30 | 50 | 100 | free |
|---|---|---|---|---|---|---|---|
| knee, D = 1e-5 (mHz) | 64 | 16 | 4.5 | 2.3 | 1.3 | 1.1 | 1.2 |
| knee, D = 1e-4 (mHz) | 460 (flat) | 160 (flat) | 43 | 22 | 12 | 11 | 12 |
| moving fraction | 0.02 | 0.07 | 0.26 | 0.49 | 0.85 | 1.00 | 1 |

So a sphere with D = 1e-5 µm²/s in a 10 nm cage gives the same knee as a free sphere with D ≈ 1.4e-4. Only the
size of the moving signal (and the static part of the hologram) tells them apart.

## Files

Copies of `d_sweep_run`, except:

| File | What changes |
|---|---|
| `make_runs.py` | `D_LEVELS = [1e-5, 1e-4]` × `CAGE_LEVELS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0]` µm (0 = free), 14 levels named like `D1e-05_c020nm` and `D1e-04_free`, 4 boxes each. `fix spring/self` with K = kT / δ². In LAMMPS E = K r²/2; checked here, 5 nm and 20 nm cages gave 5.0 and 19.9 nm rms. The manifest and `summary.csv` have `D_in` and `cage_um` |
| `plot_knee_vs_cage.py` | New. Knee against δ and moving fraction of S+ against δ, measured and predicted → `results/knee_vs_cage.png` |
| `run_lammps.sh` | 1 core per run, so all 56 runs run at once (56 CPUs) |
| job names | Start with `tc` |

The optics is the shared copy in `../shared_optics`. `plot_msd_sections.py` works unchanged. For caged spheres the MSD
flattens at 6δ² (the α per section drops to 0).

## Local test

The local check (all 56 runs, compress and the optics at 1 wavelength) is running; its numbers go here.

## On Apolo

`shared_optics` must be in `~/DPD_runs/`.
```powershell
scp -r tether_run ugr-afcontrerm@apolo-3.eafit.edu.co:~/DPD_runs/
```
```bash
cd ~/DPD_runs/tether_run
conda activate lammps-mpi
python make_runs.py          # under a second, fine on the login node
bash submit_all.sh
squeue -u $USER
```
LAMMPS takes about 20 s per run. The optics is about 7 min per level on 32 cores, 3 levels at a time, so about
40 min in all. When gather has finished:
```bash
MPLBACKEND=Agg python plot_knee_vs_cage.py     # under a minute (it reads S.npy of every level)
```
Download `results/summary.csv`, `results/spectra.png` and `results/knee_vs_cage.png`.
