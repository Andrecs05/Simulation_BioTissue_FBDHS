"""
Parse the thermo table(s) from a LAMMPS log file and plot MSD(t) for the
two tracer groups (orgA, orgB) defined in dpd_tissue_multitype.in.

Usage:
    python plot_msd.py dpd_run.log
"""

import sys
import numpy as np
import matplotlib.pyplot as plt


def parse_thermo_blocks(path):
    """Extract every thermo table in a LAMMPS log file.

    LAMMPS prints one thermo table per `run` command, each starting with
    a header line (first token "Step") and ending at the "Loop time"
    summary line. This script issued two `run` commands (equilibration,
    then production after `reset_timestep 0`), so the log contains two
    separate tables with independent step counters.

    Returns a list of dicts: {"header": [...], "data": np.ndarray}
    """
    blocks = []
    header, rows, in_table = None, [], False

    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("Step"):
                if in_table and rows:
                    blocks.append({"header": header, "data": np.array(rows, dtype=float)})
                header, rows, in_table = stripped.split(), [], True
                continue

            if in_table:
                if stripped.startswith("Loop time") or stripped == "":
                    if rows:
                        blocks.append({"header": header, "data": np.array(rows, dtype=float)})
                    header, rows, in_table = None, [], False
                    continue
                try:
                    rows.append([float(p) for p in stripped.split()])
                except ValueError:
                    # Line inside the table region that isn't numeric -> table ended
                    if rows:
                        blocks.append({"header": header, "data": np.array(rows, dtype=float)})
                    header, rows, in_table = None, [], False

    if in_table and rows:
        blocks.append({"header": header, "data": np.array(rows, dtype=float)})

    return blocks


def main(logfile):
    blocks = parse_thermo_blocks(logfile)
    print(f"Found {len(blocks)} thermo block(s) in {logfile}")
    for i, b in enumerate(blocks):
        print(f"  block {i}: columns = {b['header']}, {b['data'].shape[0]} rows")

    if len(blocks) < 2:
        print("Warning: expected 2 blocks (equilibration + production). "
              "Using the last block found.")

    # Production run is the LAST block (equilibration comes first,
    # then reset_timestep 0 restarts the step counter for production).
    prod = blocks[-1]
    header, data = prod["header"], prod["data"]

    def col(name):
        if name not in header:
            raise KeyError(f"Column '{name}' not found. Available: {header}")
        return data[:, header.index(name)]

    step = col("Step")
    msdA = col("c_msdA[4]")
    msdB = col("c_msdB[4]")

    dt = 0.01  # must match `timestep 0.01` in the LAMMPS input script
    time = step * dt

    # ---- Linear regression on log-log data: MSD ~ t^alpha ----
    # A power law MSD(t) = C * t^alpha becomes a straight line in log-log
    # space: log(MSD) = alpha*log(t) + log(C), so np.polyfit(...,1) directly
    # returns the slope alpha (and intercept = log(C)).
    # alpha = 1 -> normal diffusion, < 1 -> subdiffusive, > 1 -> superdiffusive
    mask = time > 0
    if np.count_nonzero(mask) < 2:
        raise RuntimeError("Not enough positive-time samples to fit a slope.")
    logt = np.log(time[mask])

    def fit_power_law(msd):
        logmsd = np.log(np.clip(msd[mask], 1e-12, None))
        slope, intercept = np.polyfit(logt, logmsd, 1)
        fitted = np.exp(intercept) * time[mask] ** slope
        return slope, intercept, fitted

    alphaA, interceptA, fitA = fit_power_law(msdA)
    alphaB, interceptB, fitB = fit_power_law(msdB)

    print(f"Organelle A: MSD ~ t^{alphaA:.3f}  (linear-regression slope, log-log space)")
    print(f"Organelle B: MSD ~ t^{alphaB:.3f}  (linear-regression slope, log-log space)")

    # ---- Plot: log-log axes so the power-law fit appears as a straight line ----
    plt.figure(figsize=(6.5, 5))

    plt.loglog(time, msdA, "o", ms=3, alpha=0.5, color="tab:blue",
               label="Organelle A data")
    plt.loglog(time[mask], fitA, "--", color="tab:blue",
               label=fr"Organelle A fit: slope $\alpha$ = {alphaA:.2f}")

    plt.loglog(time, msdB, "o", ms=3, alpha=0.5, color="tab:orange",
               label="Organelle B data")
    plt.loglog(time[mask], fitB, "--", color="tab:orange",
               label=fr"Organelle B fit: slope $\alpha$ = {alphaB:.2f}")

    plt.xlabel("Time (reduced units)")
    plt.ylabel(r"MSD (reduced units$^2$)")
    plt.title(r"Tracer MSD with linear regression: MSD $\sim t^{\alpha}$")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("msd_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    logfile = sys.argv[1] if len(sys.argv) > 1 else "dpd_run.log"
    main(logfile)