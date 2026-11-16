"""Active cell: one cell (as crowding_run: 13 um, nucleus 8 um, 10% mitochondria) with Langevin (overdamped)
mitochondria that also swim, sweeping the activity, with and without a cytoskeleton cage.

Each mitochondrion is an active Brownian particle: thermal diffusion D0 plus a self-propulsion v0 along its own
direction, which turns by rotational diffusion with persistence time TAU_RUN. This is the simplest model of
motor-driven transport (runs along microtubules that change direction). v0 = 0 is an ATP-depleted cell. The cage is
fix spring/self (as tether_run), the cytoskeleton holding each mitochondrion near its starting place.
The membrane and the nucleus are walls (fix wall/region); the nucleus is also a static scatterer in the optics.

Needs the LAMMPS packages BROWNIAN (fix brownian/sphere, fix propel/self) and DIPOLE (the swimming direction).
Run
    python make_runs.py
For every level it writes RUNS_PER_LEVEL cells, runs/<level>/cell_<k>/ (box.data + in.box), and runs/manifest.csv
with one row per run (the row number is the Slurm array index) and the columns v0_nm_s and cage_um (0 = free).
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))   # utilities.py / geometry.py
from geometry import dist_to_point

# ---------------- cell (as crowding_run) ----------------
RC_UM = 0.25           # one length unit in um
BOX_UM = 14.0          # box around the cell; also the tile pitch in the optics (a monolayer of cells)
CELL_DIAM_UM = 13.0    # cell (membrane) diameter; HeLa 12-15 um (Fujioka 2006, Luciani 2001)
NUC_DIAM_UM = 8.0      # nucleus diameter; HeLa 7.5-9 um (Fujioka 2006, Maul & Deaven 1977)
MITO_DIAM_UM = 0.88    # sphere with the volume of a 0.5 x 2 um mitochondrion (optics: 'mitochondrion')
PHI_MITO = 0.10        # mitochondria / cytoplasm volume; HeLa ~10% (Posakony 1977), crowding_run phi10
RUNS_PER_LEVEL = 4     # independent cells per level (the same 4 cells at every level)
GAP = 0.1              # minimum clearance when placing (r_c)

# ---------------- dynamics ----------------
D0 = 1e-4              # thermal diffusion of a mitochondrion (um^2/s), as the Langevin runs (knee 11.9 mHz)
TAU_RUN = 30.0         # persistence time of the swimming direction (s), runs of tens of seconds
V_LEVELS = [0, 2, 5, 10, 20]        # swimming speed v0 (nm/s); knee of pure swimming ~ q v0 / 2, 10 nm/s -> ~30 mHz
CAGE_LEVELS = [0, 0.03]             # cage size (um rms per axis, tether_run); 0 = no cage
T = 1.0

# ---------------- time (as the sphere runs) ----------------
TAU_S = 3.0            # seconds per time unit
DT = 0.02              # overdamped, so only the thermal step (5 nm) and the wall contacts limit DT
FRAME_S = 5.0          # one hologram every 5 s (0.2 Hz)
RECORD_S = 7000.0      # 1400 holograms
N_DELAYS, DELAY_S = 20, 100.0   # delay slots for the tiling
EQUIL_S = 300.0

steps_per_frame = round(FRAME_S / TAU_S / DT)
n_frames = round((EQUIL_S + RECORD_S + N_DELAYS * DELAY_S) / FRAME_S)
STEPS = steps_per_frame * n_frames

BOX = BOX_UM / RC_UM
R_CELL = CELL_DIAM_UM / 2 / RC_UM
R_NUC = NUC_DIAM_UM / 2 / RC_UM
R_MITO = MITO_DIAM_UM / 2 / RC_UM
GAMMA_T = T * RC_UM ** 2 / (D0 * TAU_S)        # D0 = kT / gamma_t
GAMMA_R = 2 * T * TAU_RUN / TAU_S             # direction correlation exp(-2 D_r t), D_r = kT / gamma_r


def propulsion(v_nm_s):
    """Swimming force for a speed v0 (nm/s): F = gamma_t v0 (LJ units)."""
    return GAMMA_T * v_nm_s * 1e-3 / RC_UM * TAU_S


def k_spring(cage_um):
    """fix spring/self constant for a cage of rms size cage_um per axis (E = K r^2 / 2, so K = kT / delta^2)."""
    return T / (cage_um / RC_UM) ** 2


def mito_count():
    """Number of mitochondria: PHI_MITO of the cytoplasm volume (cell minus nucleus)."""
    return int(round(PHI_MITO * (R_CELL ** 3 - R_NUC ** 3) / R_MITO ** 3))


def place_mitos(n, rng, gap=GAP):
    """
    Random sequential addition of n mitochondria in the cytoplasm shell, between the nucleus and the membrane.

    Parameters:
    n : int
        Number of mitochondria.
    rng : int
        Seed.
    gap : float, optional
        Minimum clearance between mitochondria and from the walls (r_c). Default is GAP.

    Returns:
    c : 2D array
        Centres (r_c, box coordinates), shape (n, 3).
    """
    rng = np.random.default_rng(rng)
    box = np.array([BOX] * 3)
    r_in, r_out = R_NUC + R_MITO + gap, R_CELL - R_MITO - gap
    c = np.empty((n, 3))
    k = 0
    while k < n:
        # Candidates uniform in the shell volume: random direction, radius with p(r) ~ r^2
        u = rng.normal(size=(1000, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        r = (r_in ** 3 + rng.uniform(0, 1, 1000) * (r_out ** 3 - r_in ** 3)) ** (1 / 3)
        for p in box / 2 + u * r[:, None]:
            if k == 0 or dist_to_point(c[:k], p, box).min() >= 2 * R_MITO + gap:
                c[k] = p
                k += 1
                if k == n:
                    break
    return c


def write_data(run_dir, c, rng):
    """
    Data file: atom 1 is the nucleus (static), then one atom per mitochondrion with a random swimming direction.
    atom_style hybrid sphere dipole bond: id type x y z diameter density q mux muy muz molecule.
    The Masses comments name the types for compress_dumps.py and the optics.

    Parameters:
    run_dir : Path
        Run folder.
    c : 2D array
        Mitochondrion centres (r_c).
    rng : int
        Seed of the directions.
    """
    mu = np.random.default_rng(rng).normal(size=c.shape)
    mu /= np.linalg.norm(mu, axis=1)[:, None]
    L = ["active cell", "", f"{len(c) + 1} atoms", "2 atom types", "",
         f"0 {BOX} xlo xhi", f"0 {BOX} ylo yhi", f"0 {BOX} zlo zhi", "",
         "Masses", "", "1 1.0  # nucleus", "2 1.0  # mitochondrion", "", "Atoms # hybrid", "",
         f"1 1 {BOX / 2} {BOX / 2} {BOX / 2} {2 * R_NUC} 1 0 0 0 0 1"]
    L += [f"{i} 2 {x:.5f} {y:.5f} {z:.5f} {2 * R_MITO:.4f} 1 0 {a:.5f} {b:.5f} {d:.5f} {i}"
          for i, ((x, y, z), (a, b, d)) in enumerate(zip(c, mu), start=2)]
    (run_dir / "box.data").write_text("\n".join(L) + "\n")


def write_input(run_dir, seed, v_nm_s, cage_um):
    """
    Input script: overdamped mitochondria (fix brownian/sphere) with WCA contacts, membrane and nucleus walls,
    swimming (fix propel/self) and the cage (fix spring/self), with the same outputs as the sphere runs.

    Parameters:
    run_dir : Path
        Run folder.
    seed : int
        Seed of the Brownian noise.
    v_nm_s : float
        Swimming speed (nm/s); 0 for none.
    cage_um : float
        Cage size (um rms per axis); 0 for none.
    """
    sigma = 2 * R_MITO
    s_wall = R_MITO / 2 ** (1 / 6)            # wall WCA: the push starts when the centre is R_MITO from the wall
    c, spf = BOX / 2, steps_per_frame
    L = ["units         lj", "atom_style    hybrid sphere dipole bond", "boundary      p p p",
         "neighbor      0.5 bin", "neigh_modify  every 1 delay 0 check yes",
         "read_data     box.data", "",
         "group         mito type 2", "",
         f"pair_style    lj/cut {2 ** (1 / 6) * sigma:.4f}",
         "pair_coeff    * * 0.0 1.0 1.0          # the nucleus is a wall, not a pair partner",
         f"pair_coeff    2 2 {T} {sigma:.4f}   # WCA between mitochondria",
         "pair_modify   shift yes", "",
         f"region        cell sphere {c} {c} {c} {R_CELL} side in",
         f"region        nuc sphere {c} {c} {c} {R_NUC} side out",
         f"fix           membrane mito wall/region cell lj126 {T} {s_wall:.4f} {R_MITO:.4f}",
         f"fix           envelope mito wall/region nuc lj126 {T} {s_wall:.4f} {R_MITO:.4f}", "",
         f"fix           bd mito brownian/sphere {T} {seed} gamma_t {GAMMA_T:.5g} gamma_r {GAMMA_R:.5g}"
         f"   # D0 = {D0:g} um^2/s, run persistence {TAU_RUN:g} s"]
    if v_nm_s:
        L += [f"fix           swim mito propel/self dipole {propulsion(v_nm_s):.5g}   # v0 = {v_nm_s} nm/s"]
    if cage_um:
        L += [f"fix           cage mito spring/self {k_spring(cage_um):.5g}   # cage {cage_um} um rms per axis"]
    L += ["",
          "compute       chunks all chunk/atom molecule compress yes",
          "compute       com all com/chunk chunks",
          f"fix           comout all ave/time {spf} 1 {spf} c_com[*] file com_org.dat mode vector",
          "compute       msd all msd/chunk chunks",
          f"fix           msdout all ave/time {spf} 1 {spf} c_msd[4] file msd_org.dat mode vector", "",
          f"timestep      {DT}", f"thermo        {100 * spf}", "thermo_modify flush yes",
          f"run           {STEPS}", ""]
    (run_dir / "in.box").write_text("\n".join(L))


def level_name(v, cage):
    return f"v{v:02d}_" + (f"c{round(cage * 1e3):03d}nm" if cage else "free")     # v10_free, v05_c030nm


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    n = mito_count()
    cells = [place_mitos(n, rng=3000 + k) for k in range(RUNS_PER_LEVEL)]   # the same cells at every level
    rows = []
    for cage in CAGE_LEVELS:
        for v in V_LEVELS:
            name = level_name(v, cage)
            for k in range(RUNS_PER_LEVEL):
                i = len(rows)                                 # Slurm array index of this run
                run_dir = here / f"runs/{name}/cell_{k}"
                run_dir.mkdir(parents=True, exist_ok=True)
                write_data(run_dir, cells[k], rng=3100 + k)
                write_input(run_dir, seed=83411 + 10 * i, v_nm_s=v, cage_um=cage)
                rows.append(dict(index=i, run_dir=f"runs/{name}/cell_{k}", level=name, v0_nm_s=v, cage_um=cage,
                                 n_mito=n, seed=83411 + 10 * i))
            print(f"{name}: v0 = {v} nm/s (F = {propulsion(v):.4g}), cage {cage * 1e3:.0f} nm", flush=True)
    with open(here / "runs" / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} runs, {n} mitochondria per cell, gamma_t {GAMMA_T:.4g}, gamma_r {GAMMA_R:.4g}, "
          f"{STEPS} steps per run, one frame every {steps_per_frame} steps")
