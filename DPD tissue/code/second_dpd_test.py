import sys
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "lammps_data"
GRAPHS_DIR = BASE_DIR / "graphs"


# ============================================================
# Parse thermo blocks
# ============================================================

def parse_thermo_blocks(path):
    """
    Extract every thermo table in a LAMMPS log file.

    Each table starts with a line beginning with "Step" and ends
    when a non-numeric line or "Loop time" is encountered.

    Returns
    -------
    blocks : list of dict
        Each dictionary contains:
            {
                "header": [...],
                "data": np.ndarray
            }
    """

    blocks = []
    header = None
    rows = []
    in_table = False

    with open(path, "r") as f:

        for line in f:
            stripped = line.strip()

            # Start of a thermo table
            if stripped.startswith("Step"):

                # Save previous table if necessary
                if in_table and rows:
                    blocks.append({
                        "header": header,
                        "data": np.array(rows, dtype=float)
                    })

                header = stripped.split()
                rows = []
                in_table = True
                continue

            if in_table:

                # End of thermo table
                if stripped.startswith("Loop time") or stripped == "":
                    if rows:
                        blocks.append({
                            "header": header,
                            "data": np.array(rows, dtype=float)
                        })

                    header = None
                    rows = []
                    in_table = False
                    continue

                # Try to read numeric row
                try:
                    values = [float(p) for p in stripped.split()]

                    # Make sure the number of columns is correct
                    if len(values) == len(header):
                        rows.append(values)
                    else:
                        if rows:
                            blocks.append({
                                "header": header,
                                "data": np.array(rows, dtype=float)
                            })

                        header = None
                        rows = []
                        in_table = False

                except ValueError:

                    if rows:
                        blocks.append({
                            "header": header,
                            "data": np.array(rows, dtype=float)
                        })

                    header = None
                    rows = []
                    in_table = False

    # Save final block
    if in_table and rows:
        blocks.append({
            "header": header,
            "data": np.array(rows, dtype=float)
        })

    return blocks


# ============================================================
# Find MSD column
# ============================================================

def find_msd_column(header):
    """
    Find the MSD column automatically.

    Expected examples:

        c_msdA_1[4]
        c_msdA_2[4]
        c_msdA_3[4]

    Returns
    -------
    name : str
        Name of the MSD column.
    """

    for column in header:

        if column.startswith("c_msdA_") and column.endswith("[4]"):
            return column

    raise KeyError(
        f"Could not find an Organelle A MSD column.\n"
        f"Available columns: {header}"
    )


# ============================================================
# Fit MSD ~ t^alpha
# ============================================================

def fit_power_law(time, msd):
    """
    Fit:

        MSD = C * t^alpha

    using linear regression in log-log space.

    Returns
    -------
    alpha : float
        Power-law exponent.
    intercept : float
        log(C).
    fitted : ndarray
        Fitted MSD values.
    """

    # Only use positive time and positive MSD
    mask = (time > 0) & (msd > 0) & np.isfinite(time) & np.isfinite(msd)

    if np.count_nonzero(mask) < 2:
        raise RuntimeError(
            "Not enough positive MSD/time points to perform the fit."
        )

    logt = np.log(time[mask])
    logmsd = np.log(msd[mask])

    # Linear regression:
    #
    # log(MSD) = alpha*log(t) + log(C)
    slope, intercept = np.polyfit(logt, logmsd, 1)

    fitted = np.exp(intercept) * time[mask] ** slope

    return slope, intercept, fitted, mask


# ============================================================
# Main
# ============================================================

def main(logfile):

    # --------------------------------------------------------
    # Read all thermo blocks
    # --------------------------------------------------------

    blocks = parse_thermo_blocks(logfile)

    print(f"\nFound {len(blocks)} thermo block(s) in {logfile}\n")

    if len(blocks) == 0:
        raise RuntimeError("No thermo tables were found in the log file.")

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    N_values = []
    alpha_values = []
    msd_data = []

    dt = 0.01

    # --------------------------------------------------------
    # Process every simulation
    # --------------------------------------------------------

    for i, block in enumerate(blocks):

        header = block["header"]
        data = block["data"]

        # Find Step column
        if "Step" not in header:
            print(f"Skipping block {i}: no Step column.")
            continue

        step = data[:, header.index("Step")]

        # ----------------------------------------------------
        # Find MSD column
        # ----------------------------------------------------

        try:
            msd_column = find_msd_column(header)

        except KeyError as e:
            print(f"Skipping block {i}: {e}")
            continue

        msd = data[:, header.index(msd_column)]

        # ----------------------------------------------------
        # Determine Ntrace
        #
        # c_msdA_1[4] -> Ntrace = 1
        # c_msdA_2[4] -> Ntrace = 2
        # etc.
        # ----------------------------------------------------

        match = re.search(r"c_msdA_(\d+)\[4\]", msd_column)

        if match:
            Ntrace = int(match.group(1))
        else:
            # Fallback: use block number
            Ntrace = i + 1

        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

        time = step * dt

        # ----------------------------------------------------
        # Power-law fit
        # ----------------------------------------------------

        try:
            alpha, intercept, fitted, mask = fit_power_law(
                time,
                msd
            )

        except RuntimeError as e:
            print(
                f"Ntrace = {Ntrace}: fit failed: {e}"
            )
            continue

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        N_values.append(Ntrace)
        alpha_values.append(alpha)

        msd_data.append({
            "N": Ntrace,
            "time": time,
            "msd": msd,
            "fit_time": time[mask],
            "fit_msd": fitted,
            "alpha": alpha
        })

        print(
            f"Ntrace = {Ntrace:2d} | "
            f"alpha = {alpha:.4f} | "
            f"MSD points = {len(time)}"
        )

    # --------------------------------------------------------
    # Sort by Ntrace
    # --------------------------------------------------------

    order = np.argsort(N_values)

    N_values = np.array(N_values)[order]
    alpha_values = np.array(alpha_values)[order]

    msd_data = [
        msd_data[i]
        for i in order
    ]

    # ========================================================
    # GRAPH 1
    # MSD for all simulations
    # ========================================================

    figure = plt.figure(figsize=(8, 6))

    N_min = N_values.min()
    N_max = N_values.max()
    if N_min == N_max:
        N_max = N_min + 1

    color_map = plt.get_cmap("viridis")
    color_norm = Normalize(vmin=N_min, vmax=N_max)

    for result in msd_data:

        N = result["N"]

        plt.loglog(
            result["time"],
            result["msd"],
            color=color_map(color_norm(N)),
            alpha=0.35,
            linewidth=1
        )

    # Reference power laws, anchored at the first valid MSD point.
    reference = msd_data[0]
    valid = (
        (reference["time"] > 0)
        & (reference["msd"] > 0)
        & np.isfinite(reference["time"])
        & np.isfinite(reference["msd"])
    )

    reference_time = reference["time"][valid]
    reference_msd = reference["msd"][valid]

    if len(reference_time) > 0:
        guide_time = np.array([reference_time[0], reference_time[-1]])
        anchor_time = reference_time[0]
        anchor_msd = reference_msd[0]

        for alpha, label in (
            (1.0, r"$\alpha = 1$"),
            (0.5, r"$\alpha = 0.5$")
        ):
            guide_msd = anchor_msd * (guide_time / anchor_time) ** alpha
            plt.loglog(
                guide_time,
                guide_msd,
                "--",
                linewidth=1.5,
                label=label
            )

    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=color_norm, cmap=color_map),
        ax=plt.gca()
    )
    colorbar.set_label(r"$N_{\mathrm{trace}}$ (curve color)")

    plt.xlabel("Time (reduced units)")
    plt.ylabel(r"MSD (reduced units$^2$)")

    plt.title(
        r"Mean Squared Displacement for all $N_{\mathrm{trace}}$"
    )

    plt.grid(
        True,
        which="both",
        alpha=0.2
    )

    plt.legend()

    plt.tight_layout()

    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        GRAPHS_DIR / "msd_all_simulations.png",
        dpi=200
    )

    plt.show()

    # ========================================================
    # GRAPH 2
    # Alpha vs Ntrace
    # ========================================================

    plt.figure(figsize=(7, 5))

    plt.plot(
        N_values,
        alpha_values,
        "o-",
        markersize=4
    )

    # Normal diffusion reference
    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
        label=r"$\alpha = 1$ (normal diffusion)"
    )

    plt.xlabel(r"Number of tracer particles $N_{\mathrm{trace}}$")
    plt.ylabel(r"Diffusion exponent $\alpha$")

    plt.title(
        r"Diffusion exponent $\alpha$ vs $N_{\mathrm{trace}}$"
    )

    plt.grid(
        True,
        alpha=0.2
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        GRAPHS_DIR / "alpha_vs_Ntrace.png",
        dpi=200
    )

    plt.show()

    # ========================================================
    # Save numerical results
    # ========================================================

    results = np.column_stack(
        (N_values, alpha_values)
    )

    np.savetxt(
        BASE_DIR / "alpha_vs_Ntrace.txt",
        results,
        header="Ntrace alpha",
        fmt="%.8f"
    )

    print("\nAnalysis complete.")
    print("Generated:")
    print(f"  {GRAPHS_DIR / 'msd_all_simulations.png'}")
    print(f"  {GRAPHS_DIR / 'alpha_vs_Ntrace.png'}")
    print(f"  {BASE_DIR / 'alpha_vs_Ntrace.txt'}")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    logfile = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "log_5_200.lammps"
    )

    # A bare filename is looked up in lammps_data/
    if not logfile.exists():
        logfile = DATA_DIR / logfile

    main(logfile)
