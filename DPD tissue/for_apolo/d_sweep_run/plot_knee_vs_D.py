"""Measured D (from the knee) against the input D of every level, with the instrument limits.

Run from d_sweep_run after gather:
    MPLBACKEND=Agg python plot_knee_vs_D.py
Reads results/summary.csv (columns D_in, knee_mHz, D_um2_s) and writes results/knee_vs_D.png.
"""
import csv

import numpy as np
import matplotlib.pyplot as plt

FPS = 0.2              # holograms per second
N_FRAMES = 1400        # holograms in the recording

rows = list(csv.DictReader(open("results/summary.csv")))
D_in = np.array([float(r["D_in"]) for r in rows])
fk = np.array([float(r["knee_mHz"]) for r in rows]) * 1e-3          # Hz
D_out = np.array([float(r["D_um2_s"]) for r in rows])

# Effective <|q|^2> used by gather: D_out = pi fk / q2  ->  the same for every row
q2 = np.median(np.pi * fk / D_out)
f_nyq, f_res = FPS / 2, FPS / N_FRAMES
D_nyq, D_res = np.pi * f_nyq / q2, np.pi * f_res / q2

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
x = np.logspace(np.log10(D_in.min()) - 0.3, np.log10(D_in.max()) + 0.3, 50)
for a in ax:
    a.axvspan(D_nyq, x[-1], color="tab:red", alpha=0.1, label="knee above Nyquist")
    a.axvspan(x[0], D_res, color="tab:gray", alpha=0.2, label="knee below 1/T")
    a.set_xscale("log"); a.set_xlabel(r"input D (µm$^2$/s)")

# Measured against input D, with the ideal line
ax[0].loglog(x, x, "k--", lw=1, label="measured = input")
ax[0].loglog(D_in, D_out, "o-", ms=5)
ax[0].set(ylabel=r"D from knee (µm$^2$/s)", title="D recovered by the instrument")
ax[0].legend(fontsize=8)

# Ratio, where the instrument stops reading D correctly
ax[1].semilogx(D_in, D_out / D_in, "o-", ms=5)
ax[1].axhline(1, color="k", ls="--", lw=1)
ax[1].set(ylabel="measured / input", title="Ratio")

fig.tight_layout()
fig.savefig("results/knee_vs_D.png", dpi=150)
print(f"<|q|^2> = {q2:.1f} um^-2; Nyquist {f_nyq * 1e3:.0f} mHz -> D = {D_nyq:.2e}; "
      f"resolution {f_res * 1e3:.2f} mHz -> D = {D_res:.2e} um^2/s")
for d, f, o in zip(D_in, fk, D_out):
    print(f"D_in {d:.0e}: knee {f * 1e3:7.2f} mHz, D_out {o:.3e}, ratio {o / d:.2f}")
