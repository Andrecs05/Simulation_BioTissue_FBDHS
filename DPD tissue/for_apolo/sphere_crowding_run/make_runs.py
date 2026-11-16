"""N LAMMPS runs of spherical organelles in cytoplasm (no membrane, no nucleus), sweeping crowding.

Needs utilities.py and geometry.py (from "DPD tissue/code") in sphere_crowding_run/common/. Run
    python make_runs.py
For every crowding level it writes RUNS_PER_LEVEL independent boxes, runs/<level>/box_<k>/ (box.data +
in.box), and runs/manifest.csv with one row per run (the row number is the Slurm array index).
The box is periodic and filled with cytoplasm, so each run is a piece of an endless crowded cytoplasm.
Copy sphere_crowding_run to Apolo and submit run_lammps.sh with sbatch (or everything with submit_all.sh).
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))   # utilities.py / geometry.py
from geometry import dist_to_point
from utilities import Run, Type

# ---------------- sample ----------------
RC_UM = 0.25           # one DPD length unit in um
BOX_UM = 14.0          # periodic box, same size (and depth) as the crowding_run cells
SPHERE_DIAM_UM = 0.88  # sphere with the volume of a 0.5 x 2 um mitochondrion (0.36 um^3), as in the optics

PHI_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30]   # volume fraction of the box taken by the spheres
RUNS_PER_LEVEL = 4     # independent boxes per level (different placement and seeds)
GAP = 0.1              # minimum clearance between spheres when placing them (r_c)

# ---------------- time (as crowding_run, except DT) ----------------
TAU_S = 3.0            # seconds per DPD time unit
DT = 0.02             # crowding_run uses 0.01; 0.02 gives the same T and sphere D (pilot) in half the steps
FRAME_S = 5.0          # one hologram every 5 s (0.2 Hz)
RECORD_S = 7000.0      # 1400 holograms
N_DELAYS, DELAY_S = 20, 100.0   # delay slots for the tiling
EQUIL_S = 300.0

steps_per_frame = round(FRAME_S / TAU_S / DT)
n_frames = round((EQUIL_S + RECORD_S + N_DELAYS * DELAY_S) / FRAME_S)
STEPS = steps_per_frame * n_frames

BOX = BOX_UM / RC_UM
R_SPH = SPHERE_DIAM_UM / 2 / RC_UM


class SphereRun(Run):
    """Run that places hard spheres by random sequential addition with a centre-to-centre test.
    Run.add_type tests every bead of the candidate against every object, which is far too slow for
    thousands of spheres; for spheres the centre distance alone is enough."""
    def place_spheres(self, t, max_tries):
        t.atom_type = len(self.types) + 2            # 1 is cytoplasm
        c = np.empty((t.n, 3))
        n, tries = 0, 0
        while n < t.n:
            # Candidates in batches, each one tested against the placed spheres and the earlier ones of its batch
            for p in self.rng.uniform(0, 1, (1000, 3)) * self.BOX:
                tries += 1
                if n == 0 or dist_to_point(c[:n], p, self.BOX).min() >= 2 * t.r + self.gap:
                    c[n] = p
                    n += 1
                    if n == t.n:
                        break
            if tries > max_tries:
                raise RuntimeError(f"placed only {n} of {t.n} {t.name}; box too crowded")
        t.positions = list(c)
        self.types.append(t)


def write_input(run, T=1.0, a_cc=25.0, gamma=4.5, seed=34387):
    """Input script for cytoplasm + spheres. Same interactions and outputs as crowding_run's in.cell
    without the membrane and the nucleus: one centre of mass per sphere every frame (com_org.dat)
    and their MSD (msd_org.dat)."""
    (sph,) = run.types
    spf = steps_per_frame
    L = ["units         lj", "atom_style    bond", "boundary      p p p",
         f"comm_modify   vel yes cutoff {sph.half_extent() + 1.5:.1f}",
         "neighbor      0.3 bin", "neigh_modify  every 1 delay 0 check yes",
         f"read_data     {run.in_name}.data", "",
         f"pair_style    dpd {T} 1.0 {seed}",
         f"pair_coeff    1 1 {a_cc} {gamma}",
         f"pair_coeff    1 {sph.atom_type} {sph.a_cyto} {gamma}   # cytoplasm-{sph.name}",
         f"pair_coeff    {sph.atom_type} {sph.atom_type} {sph.a_self} {gamma}", "",
         "group         cyto type 1",
         f"group         org type {sph.atom_type}",
         "neigh_modify  exclude molecule/intra org", "",
         f"velocity      cyto create {T} {seed + 1} mom yes rot no",
         "fix           fluid cyto nve",
         "fix           spheres org rigid/small molecule", "",
         "compute       chunks org chunk/atom molecule compress yes",
         "compute       com org com/chunk chunks",
         f"fix           comout org ave/time {spf} 1 {spf} c_com[*] file com_org.dat mode vector",
         "compute       msd org msd/chunk chunks",
         f"fix           msdout org ave/time {spf} 1 {spf} c_msd[4] file msd_org.dat mode vector", "",
         f"timestep      {DT}", f"thermo        {10 * spf}", "thermo_modify flush yes",
         f"run           {STEPS}", ""]
    (run.run_dir / f"in.{run.in_name}").write_text("\n".join(L))


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
            run_dir = f"runs/{level_name(phi)}/box_{k}"
            run = SphereRun(here / run_dir, "box", BOX=[BOX] * 3, rng=2000 + i, gap=GAP)
            run.place_spheres(Type("sphere", n_sph, "sphere", r=R_SPH, rel_density=1.1), max_tries=10 ** 7)
            run.fill_cytoplasm(RHO=3)
            n_atoms = run.write_data_file()
            write_input(run, seed=44387 + 10 * i)
            rows.append(dict(index=i, run_dir=run_dir, level=level_name(phi), phi=phi, n_sph=n_sph,
                             atoms=n_atoms, seed=44387 + 10 * i))
            print(f"{run_dir}: phi = {phi:.2f}, {n_sph} spheres, {n_atoms} atoms, {time.time() - t0:.0f} s", flush=True)
    with open(here / "runs" / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} runs -> use #SBATCH --array=0-{len(rows) - 1} in run_lammps.sh")
    print(f"box {BOX:.0f} r_c ({BOX_UM:.1f} um), {STEPS} steps per run, one frame every {steps_per_frame} steps")
