"""Langevin control for sphere_crowding_run: the same spheres, boxes and crowding levels, without the DPD fluid.

Each sphere is one particle with Langevin friction set so that, alone, it diffuses with D_UM2_S (Stokes-Einstein
like, no hydrodynamics), and the spheres repel each other as hard spheres (WCA, contact at one diameter).
The sphere centres are placed with the same seeds as sphere_crowding_run, so box k of each level starts from
the same configuration as its DPD twin. Run
    python make_runs.py
For every crowding level it writes RUNS_PER_LEVEL boxes, runs/<level>/box_<k>/ (box.data + in.box), and
runs/manifest.csv with one row per run (the row number is the Slurm array index). Outputs (com_org.dat,
msd_org.dat) have the same format as sphere_crowding_run, so compress, optics and plot_msd_sections run unchanged.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))   # utilities.py / geometry.py
from geometry import dist_to_point

# ---------------- sample (as sphere_crowding_run) ----------------
RC_UM = 0.25           # one length unit in um
BOX_UM = 14.0          # periodic box
SPHERE_DIAM_UM = 0.88  # sphere with the volume of a 0.5 x 2 um mitochondrion

PHI_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30]   # volume fraction of the box taken by the spheres
RUNS_PER_LEVEL = 4     # independent boxes per level
GAP = 0.1              # minimum clearance between spheres when placing them (r_c)

# ---------------- dynamics ----------------
D_UM2_S = 1e-4         # free-sphere diffusion coefficient (um^2/s); knee q^2 D / pi ~ 12 mHz, the DPD phi05 knee
MASS = 10.0            # sphere mass (LJ units); with D it sets the momentum time damp = M D / kT
T = 1.0

# ---------------- time (as sphere_crowding_run) ----------------
TAU_S = 3.0            # seconds per time unit
DT = 0.005             # ~1/10 of the momentum time, so the WCA contacts and the friction are resolved
FRAME_S = 5.0          # one hologram every 5 s (0.2 Hz)
RECORD_S = 7000.0      # 1400 holograms
N_DELAYS, DELAY_S = 20, 100.0   # delay slots for the tiling
EQUIL_S = 300.0

steps_per_frame = round(FRAME_S / TAU_S / DT)
n_frames = round((EQUIL_S + RECORD_S + N_DELAYS * DELAY_S) / FRAME_S)
STEPS = steps_per_frame * n_frames

BOX = BOX_UM / RC_UM
R_SPH = SPHERE_DIAM_UM / 2 / RC_UM
D_LJ = D_UM2_S * TAU_S / RC_UM ** 2         # r_c^2 / tau
DAMP = MASS * D_LJ / T                      # Langevin damping time: D = kT damp / M


def place_spheres(n, rng, gap=GAP):
    """
    Random sequential addition of n spheres in the periodic box, same algorithm and random stream as
    SphereRun.place_spheres in sphere_crowding_run.

    Parameters:
    n : int
        Number of spheres.
    rng : int
        Seed.
    gap : float, optional
        Minimum clearance between spheres (r_c). Default is GAP.

    Returns:
    c : 2D array
        Sphere centres (r_c), shape (n, 3).
    """
    rng = np.random.default_rng(rng)
    box = np.array([BOX] * 3)
    c = np.empty((n, 3))
    k = 0
    while k < n:
        for p in rng.uniform(0, 1, (1000, 3)) * box:
            if k == 0 or dist_to_point(c[:k], p, box).min() >= 2 * R_SPH + gap:
                c[k] = p
                k += 1
                if k == n:
                    break
    return c


def write_data(run_dir, c):
    """
    Data file with one atom (and one molecule) per sphere, in the 'Atoms # bond' format of the DPD runs,
    so compress_dumps.py reads it unchanged.

    Parameters:
    run_dir : Path
        Run folder.
    c : 2D array
        Sphere centres (r_c).
    """
    L = ["box", "", f"{len(c)} atoms", "1 atom types", "",
         f"0 {BOX} xlo xhi", f"0 {BOX} ylo yhi", f"0 {BOX} zlo zhi", "",
         "Masses", "", f"1 {MASS}  # sphere", "", "Atoms # bond", ""]
    L += [f"{i} {i} 1 {x:.5f} {y:.5f} {z:.5f} 0 0 0" for i, (x, y, z) in enumerate(c, start=1)]
    (run_dir / "box.data").write_text("\n".join(L) + "\n")


def write_input(run_dir, seed):
    """
    Input script: hard spheres (WCA) with Langevin friction, and the same outputs as sphere_crowding_run.

    Parameters:
    run_dir : Path
        Run folder.
    seed : int
        Seed of the Langevin noise and the initial velocities.
    """
    sigma = 2 * R_SPH
    spf = steps_per_frame
    L = ["units         lj", "atom_style    bond", "boundary      p p p",
         "neighbor      0.5 bin", "neigh_modify  every 1 delay 0 check yes",
         "read_data     box.data", "",
         f"pair_style    lj/cut {2 ** (1 / 6) * sigma:.4f}",
         f"pair_coeff    1 1 {T} {sigma:.4f}   # WCA: purely repulsive, contact at one diameter",
         "pair_modify   shift yes", "",
         f"velocity      all create {T} {seed + 1} mom yes rot no",
         "fix           move all nve",
         f"fix           bath all langevin {T} {T} {DAMP:.5g} {seed}   # D = {D_LJ:.5g} r_c^2/tau alone", "",
         "compute       chunks all chunk/atom molecule compress yes",
         "compute       com all com/chunk chunks",
         f"fix           comout all ave/time {spf} 1 {spf} c_com[*] file com_org.dat mode vector",
         "compute       msd all msd/chunk chunks",
         f"fix           msdout all ave/time {spf} 1 {spf} c_msd[4] file msd_org.dat mode vector", "",
         f"timestep      {DT}", f"thermo        {100 * spf}", "thermo_modify flush yes",
         f"run           {STEPS}", ""]
    (run_dir / "in.box").write_text("\n".join(L))


def sphere_count(phi):
    """Number of spheres for a volume fraction phi of the box."""
    return int(round(phi * BOX ** 3 / (4 / 3 * np.pi * R_SPH ** 3)))


def level_name(phi):
    return f"phi{round(100 * phi):02d}"          # 0.05 -> phi05


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    rows = []
    for phi in PHI_LEVELS:
        n_sph = sphere_count(phi)
        for k in range(RUNS_PER_LEVEL):
            t0 = time.time()
            i = len(rows)                                     # Slurm array index of this run
            run_dir = here / f"runs/{level_name(phi)}/box_{k}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_data(run_dir, place_spheres(n_sph, rng=2000 + i))   # same seed as the DPD box
            write_input(run_dir, seed=54387 + 10 * i)
            rows.append(dict(index=i, run_dir=f"runs/{level_name(phi)}/box_{k}", level=level_name(phi), phi=phi,
                             n_sph=n_sph, atoms=n_sph, seed=54387 + 10 * i))
            print(f"{rows[-1]['run_dir']}: phi = {phi:.2f}, {n_sph} spheres, {time.time() - t0:.0f} s", flush=True)
    with open(here / "runs" / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} runs, {STEPS} steps per run, one frame every {steps_per_frame} steps, "
          f"D = {D_LJ:.4g} r_c^2/tau, damp = {DAMP:.4g} tau")
