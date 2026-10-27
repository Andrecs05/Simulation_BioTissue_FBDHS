"""N LAMMPS runs of one cell (membrane + nucleus + mitochondria), sweeping mitochondrial crowding.

Needs utilities.py and geometry.py (from "DPD tissue/code") in crowding_run/common/. Run
    python make_runs.py
For every crowding level it writes RUNS_PER_LEVEL independent cells, runs/<level>/cell_<k>/ (cell.data +
in.cell), and runs/manifest.csv with one row per run (the row number is the Slurm array index).
Several cells per level are what the tiling needs to build one sample per level.
Copy crowding_run to Apolo and submit run_lammps.sh with sbatch.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))   # utilities.py / geometry.py
import utilities
from utilities import Run, Type

# ---------------- sample ----------------
RC_UM = 0.25           # one DPD length unit in um
CELL_DIAM_UM = 13.0    # cell (membrane) diameter; HeLa 12-15 um (Fujioka 2006, Luciani 2001)
NUC_DIAM_UM = 8.0      # nucleus diameter; HeLa 7.5-9 um (Fujioka 2006, Maul & Deaven 1977)
MITO_DIAM_UM = 0.5     # mitochondrion diameter; ~0.5 um (Baffou 2014)
MITO_LEN_UM = 2.0      # mitochondrion length, tip to tip; 1-2 um (Roskams & Rodgers; Phillips, Physical Biology of the Cell)
MARGIN_RC = 2.0        # medium between the membrane and the box edge, each side

PHI_LEVELS = [0.05, 0.10, 0.15, 0.20]   # mitochondrial volume fraction of the cytoplasm (nucleus excluded);
                                        # HeLa ~10% (Posakony 1977), rat hepatocyte 22% of the cell (Weibel 1969)
RUNS_PER_LEVEL = 4     # independent cells per level (different placement and seeds)
GAP = 0.3              # minimum clearance between objects when placing them (r_c)

# ---------------- time ----------------
TAU_S = 3.0            # seconds per DPD time unit (calibrated on the small cell, kept for every run)
DT = 0.01
FRAME_S = 5.0          # one hologram every 5 s (0.2 Hz)
RECORD_S = 7000.0      # 1400 holograms
N_DELAYS, DELAY_S = 20, 100.0   # delay slots for the tiling
EQUIL_S = 300.0
MEM_EVERY = 20         # dump the membrane beads every 20 frames (100 s), only to look at the cell

steps_per_frame = round(FRAME_S / TAU_S / DT)
n_frames = round((EQUIL_S + RECORD_S + N_DELAYS * DELAY_S) / FRAME_S)
STEPS = steps_per_frame * n_frames

R_CELL = CELL_DIAM_UM / 2 / RC_UM
R_NUC = NUC_DIAM_UM / 2 / RC_UM
R_MITO = MITO_DIAM_UM / 2 / RC_UM
L_MITO = MITO_LEN_UM / RC_UM
BOX = 2 * (R_CELL + MARGIN_RC)


def fast_shell_bonds(p, s):
    """Same bonds as geometry.shell_bonds (pairs closer than 1.5 s) without the N x N distance matrix,
    which needs several GB for a 10 um cell. Shell beads sit on a grid of spacing s, so only the
    26 neighbouring grid sites can be bonded."""
    ijk = np.rint((p - p.min(axis=0)) / s).astype(np.int64)
    M = ijk.max() + 3
    key = (ijk[:, 0] + 1) * M * M + (ijk[:, 1] + 1) * M + (ijk[:, 2] + 1)
    order = np.argsort(key)
    sorted_key = key[order]
    I, J = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) <= (0, 0, 0):        # each pair once
                    continue
                k2 = key + dx * M * M + dy * M + dz
                pos = np.clip(np.searchsorted(sorted_key, k2), 0, len(key) - 1)
                hit = sorted_key[pos] == k2
                I.append(np.nonzero(hit)[0]); J.append(order[pos[hit]])
    i, j = np.concatenate(I), np.concatenate(J)
    i, j = np.minimum(i, j), np.maximum(i, j)
    d = np.linalg.norm(p[i] - p[j], axis=1)
    keep = d < 1.5 * s
    i, j, d = i[keep], j[keep], d[keep]
    o = np.lexsort((j, i))
    return i[o], j[o], d[o]


utilities.shell_bonds = fast_shell_bonds     # write_data_file uses it for the membrane


class CellRun(Run):
    """Run that draws mitochondria inside the cytoplasm only (between nucleus and membrane) and points
    them in any 3D direction. Run.random_pos draws them anywhere in the box and keeps them in the xy
    plane, so most tries land outside the cell and dense cells cannot be filled."""
    def random_pos(self, t):
        if t.geo != "capsule":
            return super().random_pos(t)
        r_in, r_out = R_NUC + t.r, R_CELL - t.r
        r = (self.rng.uniform(r_in ** 3, r_out ** 3)) ** (1 / 3)      # uniform in the spherical shell
        u = self.rng.normal(size=3); u /= np.linalg.norm(u)
        a = self.rng.normal(size=3); a /= np.linalg.norm(a)
        return self.BOX / 2 + r * u, a


def write_input(run, T=1.0, a_cc=25.0, gamma=4.5, k_bond=100.0, seed=34387):
    """Input script for membrane + one nucleus + mitochondria. Differences from Run.write_input_file:
    the nucleus is one rigid body under fix rigid (global), so the ghost cutoff only has to cover a
    mitochondrion; positions are saved as one centre of mass per organelle (com_org.dat) and the
    membrane centre (com_mem.dat) every frame instead of a bead dump."""
    mem, nuc, mito = run.types
    m = run.medium_type
    spf = steps_per_frame
    L = ["units         lj", "atom_style    bond", "boundary      p p p",
         f"comm_modify   vel yes cutoff {mito.half_extent() + 1.5:.1f}",
         "neighbor      0.3 bin", "neigh_modify  every 1 delay 0 check yes",
         f"read_data     {run.in_name}.data", "",
         f"pair_style    dpd {T} 1.0 {seed}",
         f"pair_coeff    1 1 {a_cc} {gamma}"]
    for i, t in enumerate(run.types):
        L.append(f"pair_coeff    1 {t.atom_type} {t.a_cyto} {gamma}   # cytoplasm-{t.name}")
        L.append(f"pair_coeff    {t.atom_type} {t.atom_type} {t.a_self} {gamma}")
        for u in run.types[i + 1:]:
            L.append(f"pair_coeff    {t.atom_type} {u.atom_type} {(t.a_self + u.a_self) / 2} {gamma}")
    L += [f"pair_coeff    1 {m} {a_cc} {gamma}   # cytoplasm-medium",
          f"pair_coeff    {m} {m} {a_cc} {gamma}"]
    L += [f"pair_coeff    {t.atom_type} {m} {t.a_cyto} {gamma}   # {t.name}-medium" for t in run.types]
    L += ["", "bond_style    harmonic",
          f"bond_coeff    1 {k_bond} {mem.s}",
          f"bond_coeff    2 {k_bond} {mem.s * np.sqrt(2):.4f}", "",
          f"group         mobile type 1 {mem.atom_type} {m}",
          f"group         org type {nuc.atom_type} {mito.atom_type}"]
    L += [f"group         {t.name} type {t.atom_type}" for t in run.types]
    L += ["neigh_modify  exclude molecule/intra org", "",
          f"velocity      mobile create {T} {seed + 1} mom yes rot no",
          "fix           fluid mobile nve",
          f"fix           nuc {nuc.name} rigid single",
          f"fix           mito {mito.name} rigid/small molecule", "",
          "compute       chunks org chunk/atom molecule compress yes",
          "compute       com org com/chunk chunks",
          f"fix           comout org ave/time {spf} 1 {spf} c_com[*] file com_org.dat mode vector",
          f"compute       memc {mem.name} com",
          f"fix           memout all ave/time {spf} 1 {spf} c_memc[*] file com_mem.dat",
          "compute       msd org msd/chunk chunks",
          f"fix           msdout org ave/time {spf} 1 {spf} c_msd[4] file msd_org.dat mode vector", "",
          f"timestep      {DT}", f"thermo        {10 * spf}", "thermo_modify flush yes",
          f"dump          mem {mem.name} custom {MEM_EVERY * spf} dump.membrane id x y z ix iy iz",
          f"run           {STEPS}", ""]
    (run.run_dir / f"in.{run.in_name}").write_text("\n".join(L))


def mito_count(phi):
    """Number of mitochondria for a volume fraction phi of the cytoplasm."""
    v_cyto = 4 / 3 * np.pi * (R_CELL ** 3 - R_NUC ** 3)
    v_mito = np.pi * R_MITO ** 2 * (L_MITO - 2 * R_MITO) + 4 / 3 * np.pi * R_MITO ** 3
    return int(round(phi * v_cyto / v_mito))


def level_name(phi):
    return f"phi{round(100 * phi):02d}"          # 0.05 -> phi05


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    rows = []
    for phi in PHI_LEVELS:
        n_mito = mito_count(phi)
        for k in range(RUNS_PER_LEVEL):
            i = len(rows)                                     # Slurm array index of this run
            run_dir = f"runs/{level_name(phi)}/cell_{k}"
            run = CellRun(here / run_dir, "cell", BOX=[BOX] * 3, rng=1000 + i, gap=GAP)
            run.add_type(Type("membrane", 1, "shell", r=R_CELL, a_cyto=50.0))
            run.add_type(Type("nucleus", 1, "sphere", r=R_NUC, rel_density=1.10, a_cyto=40.0))
            run.add_type(Type("mitochondrion", n_mito, "capsule", r=R_MITO, l=L_MITO, rel_density=1.1),
                         max_tries=2000 * n_mito + 10000)
            run.fill_cytoplasm(RHO=3)
            n_atoms = run.write_data_file()
            write_input(run, seed=34387 + 10 * i)
            rows.append(dict(index=i, run_dir=run_dir, level=level_name(phi), phi=phi, n_mito=n_mito,
                             atoms=n_atoms, seed=34387 + 10 * i))
            print(f"{run_dir}: phi = {phi:.2f}, {n_mito} mitochondria, {n_atoms} atoms")
    with open(here / "runs" / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} runs -> use #SBATCH --array=0-{len(rows) - 1} in run_lammps.sh")
    print(f"box {BOX:.0f} r_c ({BOX * RC_UM:.1f} um), {STEPS} steps per run, one frame every {steps_per_frame} steps")
