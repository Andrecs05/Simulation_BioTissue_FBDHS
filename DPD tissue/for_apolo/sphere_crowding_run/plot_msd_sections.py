"""MSD of the spheres per crowding level, with a power-law fit (alpha) in log-spaced lag sections.

Same analysis as crowding_run/plot_msd_sections.py (sections from config_generator_tests.ipynb, cell 11),
for the sphere boxes: one organelle type, no nucleus panel. Run from sphere_crowding_run after compress:
    MPLBACKEND=Agg python plot_msd_sections.py
Writes results/msd_sections.png and prints alpha per section (mean ± spread over the boxes of each level).
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utilities import msd_time_averaged, fit_power_law
from make_runs import RC_UM, TAU_S, DT, steps_per_frame

FRAME_S = steps_per_frame * DT * TAU_S   # s per frame (83 steps x 0.02 tau x 3 s/tau = 4.98 s)
EQUIL_FRAMES = 60            # frames the hologram sampling throws away
N_SECTIONS = 5               # log-spaced lag windows, one alpha per window
MAX_LAG_FRAC = 0.5           # longest lag used, as a fraction of the post-equilibrium length

rows = list(csv.DictReader(open("runs/manifest.csv")))
levels = sorted({r["level"] for r in rows})
colors = dict(zip(levels, plt.cm.viridis(np.linspace(0, 0.9, len(levels)))))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
alpha_sections = {}          # level -> list of alpha arrays, one per box
D_long = {}                  # level -> list of MSD / (6 t) at the longest lag, one per box
for r in rows:
    d = np.load(f"{r['run_dir']}/traj.npz")

    # Drop the equilibration frames; com is unwrapped, relative to the box centre
    com = d["com"][EQUIL_FRAMES:]
    lags, msd, _ = msd_time_averaged(com)
    t = lags * FRAME_S
    keep = t <= MAX_LAG_FRAC * t[-1]
    t, msd = t[keep], msd[keep] * RC_UM**2               # s, um^2

    # Section edges, evenly spaced in log(lag)
    edges = np.logspace(np.log10(t[0]), np.log10(t[-1]), N_SECTIONS + 1)
    edges[[0, -1]] = t[0], t[-1]                         # exact ends, so the first and last lags are not lost to rounding
    sections = [(t >= a) & (t <= b) for a, b in zip(edges[:-1], edges[1:])]

    # Power-law fit inside each section, drawn on top of the data
    color = colors[r["level"]]
    alphas = []
    for m in sections:
        m = m & (msd > 0)
        if m.sum() < 3:
            alphas.append(np.nan)
            continue
        a, c = fit_power_law(t[m], msd[m])
        alphas.append(a)
        ax[0].loglog(t[m], c * t[m]**a, color=color, lw=3, alpha=0.6)
    alpha_sections.setdefault(r["level"], []).append(alphas)
    D_long.setdefault(r["level"], []).append(msd[-1] / (6 * t[-1]))
    first = r["run_dir"].endswith("box_0")
    ax[0].loglog(t, msd, "o", color=color, ms=1.5, label=r["level"] if first else None)
    ax[1].loglog(t, msd / (6 * t), "-", color=color, lw=1, label=r["level"] if first else None)
    ax[2].stairs(alphas, edges, baseline=None, color=color, lw=1.5, alpha=0.7)

for e in edges:
    ax[0].axvline(e, color="gray", lw=0.5, ls=":")
ax[2].axhline(1, color="k", ls="--", lw=1)
ax[2].set_xscale("log")
ax[0].set(xlabel="lag (s)", ylabel=r"MSD (µm$^2$)", title="Spheres, fits per section")
ax[1].set(xlabel="lag (s)", ylabel=r"MSD / 6t (µm$^2$/s)", title="Spheres, effective D")
ax[2].set(xlabel="lag (s)", ylabel=r"$\alpha$", title="Spheres, local exponent per section")
ax[0].legend(fontsize=8)
fig.tight_layout()
Path("results").mkdir(exist_ok=True)
fig.savefig("results/msd_sections.png", dpi=150)

# Mean +- spread over the boxes of each level
print("sections (s): " + "  ".join(f"[{a:.3g}, {b:.3g}]" for a, b in zip(edges[:-1], edges[1:])))
for level in levels:
    a = np.array(alpha_sections[level])
    D = np.array(D_long[level])
    print(f"{level:7s} sphere  alpha: " + "  ".join(f"{m:.2f}±{s:.2f}" for m, s in zip(a.mean(0), a.std(0)))
          + f"   D at {t[-1]:.0f} s: {D.mean():.2e}±{D.std():.1e} um^2/s")
