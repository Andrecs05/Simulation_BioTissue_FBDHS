"""D sweep: dilute Langevin spheres (as langevin_control_run, phi = 5%) at diffusion coefficients that span the
instrument's range and go past it on both sides.

The knee should follow q^2 D / pi until it reaches the 0.1 Hz Nyquist frequency (D ~ 8e-4 um^2/s) or the
1 / 7000 s resolution of the recording (D ~ 1e-6 um^2/s). Each level is one D; the damping time is the same
at every level (the mass changes with D), so the motion is diffusive at every lag the camera sees. Run
    python make_runs.py
For every D it writes RUNS_PER_LEVEL boxes, runs/<level>/box_<k>/ (box.data + in.box), and runs/manifest.csv
with one row per run (the row number is the Slurm array index) and the input D in column D_in.
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

PHI = 0.05             # volume fraction of the box taken by the spheres (dilute)
RUNS_PER_LEVEL = 4     # independent boxes per level
GAP = 0.1              # minimum clearance between spheres when placing them (r_c)

# ---------------- dynamics ----------------
# free-sphere diffusion coefficients (um^2/s). knee ~ 1.19e5 D mHz (from langevin_control_run: 1e-4 -> 11.9 mHz),
# so 1e-6 -> 0.12 mHz (below the 0.14 mHz resolution), 8.4e-4 -> 100 mHz (Nyquist), 2e-3 and 5e-3 aliased
D_LEVELS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 6e-4, 1e-3, 2e-3, 5e-3]
DAMP = 0.048           # Langevin damping time (tau), the same at every D; mass M = kT damp / D
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


def d_lj(D_um2_s):
    """Diffusion coefficient in r_c^2 / tau."""
    return D_um2_s * TAU_S / RC_UM ** 2


def mass(D_um2_s):
    """Sphere mass that gives D with the damping time DAMP: D = kT damp / M."""
    return T * DAMP / d_lj(D_um2_s)


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


def write_data(run_dir, c, M):
    """
    Data file with one atom (and one molecule) per sphere, in the 'Atoms # bond' format of the DPD runs,
    so compress_dumps.py reads it unchanged.

    Parameters:
    run_dir : Path
        Run folder.
    c : 2D array
        Sphere centres (r_c).
    M : float
        Sphere mass (LJ units).
    """
    L = ["box", "", f"{len(c)} atoms", "1 atom types", "",
         f"0 {BOX} xlo xhi", f"0 {BOX} ylo yhi", f"0 {BOX} zlo zhi", "",
         "Masses", "", f"1 {M:.6g}  # sphere", "", "Atoms # bond", ""]
    L += [f"{i} {i} 1 {x:.5f} {y:.5f} {z:.5f} 0 0 0" for i, (x, y, z) in enumerate(c, start=1)]
    (run_dir / "box.data").write_text("\n".join(L) + "\n")


def write_input(run_dir, seed, D_um2_s):
    """
    Input script: hard spheres (WCA) with Langevin friction, and the same outputs as sphere_crowding_run.

    Parameters:
    run_dir : Path
        Run folder.
    seed : int
        Seed of the Langevin noise and the initial velocities.
    D_um2_s : float
        Free-sphere diffusion coefficient (um^2/s), only for the comment in the script.
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
         f"fix           bath all langevin {T} {T} {DAMP:.5g} {seed}   # D = {d_lj(D_um2_s):.5g} r_c^2/tau alone", "",
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


def level_name(i, D):
    return f"d{i}_{D:.0e}"                      # 0, 1e-6 -> d0_1e-06 (sorts in D order)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    rows = []
    n_sph = sphere_count(PHI)
    for j, D in enumerate(D_LEVELS):
        for k in range(RUNS_PER_LEVEL):
            i = len(rows)                                     # Slurm array index of this run
            run_dir = here / f"runs/{level_name(j, D)}/box_{k}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_data(run_dir, place_spheres(n_sph, rng=2000 + k), mass(D))   # box k = phi05 box k of the other sweeps
            write_input(run_dir, seed=64387 + 10 * i, D_um2_s=D)
            rows.append(dict(index=i, run_dir=f"runs/{level_name(j, D)}/box_{k}", level=level_name(j, D), D_in=D,
                             phi=PHI, n_sph=n_sph, seed=64387 + 10 * i))
        print(f"{level_name(j, D)}: D = {D:.0e} um^2/s = {d_lj(D):.4g} r_c^2/tau, mass {mass(D):.4g}", flush=True)
    with open(here / "runs" / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} runs, {n_sph} spheres each, {STEPS} steps per run, one frame every {steps_per_frame} steps")
