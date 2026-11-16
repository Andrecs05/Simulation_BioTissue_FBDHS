"""Knee and moving fraction of the signal against cage size for the DPD tethered spheres, against the
tethered-sphere prediction for the D of the free DPD spheres, and the Langevin tether_run points at D = 1e-4.

Run from tether_dpd_run after gather:
    MPLBACKEND=Agg python plot_knee_vs_cage.py
Reads results/summary.csv (columns cage_um, knee_mHz, D_um2_s) and results/<level>/S.npy. The free reference is
phi05 of ../sphere_crowding_run/results (same boxes without the tether; FREE_KNEE_MHZ if that folder is missing),
and the Langevin points come from ../tether_run/results if it is there.
Writes results/knee_vs_cage.png and prints measured against predicted values for every level.

Prediction: a sphere tied by a spring of rms size delta (per axis) has the field factor
    F(t) = exp(-q^2 delta^2 (1 - exp(-t / tau_c))),   tau_c = delta^2 / D,
and S+ (two independent spheres per term) decorrelates as F^2. Its moving part F^2(t) - F^2(inf) is sampled every
5 s, turned into a spectrum and fitted with the same knee_frequency as gather, so aliasing is included.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared_optics" / "optics_code"))
from spectrum import knee_frequency

FPS = 0.2              # holograms per second
N_FRAMES = 1400        # holograms in the recording
FREE_KNEE_MHZ = 12.2   # sphere_crowding_run phi05 (Apolo, 11 wavelengths)


def predicted(D, cage, q2):
    """
    Knee (Hz) and moving fraction of S+ for spheres with diffusion coefficient D in a cage of rms size cage.

    Parameters:
    D : float
        Free diffusion coefficient (um^2/s).
    cage : float
        Cage size (um), rms per axis; 0 for free spheres.
    q2 : float
        <|q|^2> of the instrument (um^-2).

    Returns:
    fk : float
        Knee from the same Lorentzian fit as gather (Hz).
    moving : float
        Fraction of the S+ variance that changes in time, 1 - F^2(inf).
    """
    t = np.arange(N_FRAMES) / FPS
    if cage:
        a = 2 * q2 * cage ** 2
        C = np.exp(-a * (1 - np.exp(-t * D / cage ** 2))) - np.exp(-a)
        moving = 1 - np.exp(-a)
    else:
        C, moving = np.exp(-2 * q2 * D * t), 1.0
    # Wiener-Khinchin on the sampled correlation (the sum folds the aliases in, as the camera does)
    f = np.fft.rfftfreq(2 * N_FRAMES, 1 / FPS)[1:N_FRAMES]
    w = np.ones(N_FRAMES); w[0] = 0.5
    P = 2 * (w * C * np.cos(2 * np.pi * f[:, None] * t)).sum(axis=1) / FPS
    return knee_frequency(f, np.maximum(P, 1e-30 * P.max()), FPS), moving


def moving_fraction(level, results="results"):
    """Measured fraction of the S+ variance that changes in time, |S - <S>_t|^2 / |S|^2 over all pixels."""
    S = np.load(Path(results) / level / "S.npy")
    dS = S - S.mean(axis=0)
    return (np.abs(dS) ** 2).mean() / (np.abs(S) ** 2).mean()


def read(results, keep=lambda r: True):
    """Rows of results/summary.csv with the knee in Hz and the moving fraction."""
    path = Path(results) / "summary.csv"
    if not path.exists():
        return []
    rows = [r for r in csv.DictReader(open(path)) if keep(r)]
    for r in rows:
        r["cage_um"] = float(r.get("cage_um", 0) or 0)
        r["fk"] = float(r["knee_mHz"]) * 1e-3
        r["moving"] = moving_fraction(r["level"], results)
    return rows


rows = sorted(read("results"), key=lambda r: r["cage_um"])
q2 = np.median([np.pi * r["fk"] / float(r["D_um2_s"]) for r in rows])     # <|q|^2> used by gather
free = read("../sphere_crowding_run/results", lambda r: r["level"] == "phi05")
fk_free = free[0]["fk"] if free else FREE_KNEE_MHZ * 1e-3
D_free = np.pi * fk_free / q2                                              # short-time D of the free DPD spheres
lang = read("../tether_run/results", lambda r: float(r["D_in"]) == 1e-4)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
cages = np.logspace(np.log10(0.003), np.log10(0.2), 60)
pred = np.array([predicted(D_free, x, q2) for x in cages])
x_dpd = [r["cage_um"] * 1e3 for r in rows]
ax[0].loglog(cages * 1e3, pred[:, 0] * 1e3, "k-", lw=1, label=f"prediction, D = {D_free:.2e} µm²/s (free DPD knee)")
ax[0].loglog(x_dpd, [r["fk"] * 1e3 for r in rows], "o", color="C0", label="DPD tethered")
ax[0].axhline(fk_free * 1e3, color="C0", ls=":", lw=1, label="DPD free (sphere_crowding_run phi05)")
ax[1].semilogx(cages * 1e3, pred[:, 1], "k-", lw=1, label=r"$1 - e^{-2 q^2 \delta^2}$")
ax[1].semilogx(x_dpd, [r["moving"] for r in rows], "o", color="C0", label="DPD tethered")
if free:
    ax[1].axhline(free[0]["moving"], color="C0", ls=":", lw=1, label="DPD free")
lt = [r for r in lang if r["cage_um"] > 0]
if lt:
    ax[0].loglog([r["cage_um"] * 1e3 for r in lt], [r["fk"] * 1e3 for r in lt], "s", mfc="none", color="C1",
                 label="Langevin tethered, D = 1e-4 (tether_run)")
    ax[1].semilogx([r["cage_um"] * 1e3 for r in lt], [r["moving"] for r in lt], "s", mfc="none", color="C1",
                   label="Langevin, D = 1e-4")

for r in rows:
    fk_p, mv_p = predicted(D_free, r["cage_um"], q2)
    print(f"{r['level']:8s} knee {r['fk'] * 1e3:7.2f} mHz (predicted {fk_p * 1e3:7.2f}), "
          f"moving fraction {r['moving']:.3f} (predicted {mv_p:.3f})")
print(f"free DPD knee {fk_free * 1e3:.2f} mHz -> D = {D_free:.3e} um^2/s" + ("" if free else " (FREE_KNEE_MHZ)"))

ax[0].axhline(FPS / 2 * 1e3, color="tab:red", lw=1, label="Nyquist")
ax[0].axvline(1e3 / np.sqrt(2 * q2), color="gray", ls="--", lw=1, label=r"$\delta = 1/(\sqrt{2}\,q)$")
ax[0].set(xlabel="cage size δ (nm, rms per axis)", ylabel="knee (mHz)", title="Knee")
ax[0].legend(fontsize=7)
ax[1].axvline(1e3 / np.sqrt(2 * q2), color="gray", ls="--", lw=1)
ax[1].set(xlabel="cage size δ (nm, rms per axis)", ylabel="moving / total power of S+",
          title="Fraction of the signal that moves")
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig("results/knee_vs_cage.png", dpi=150)
print(f"<|q|^2> = {q2:.1f} um^-2, 1/(sqrt(2) q) = {1e3 / np.sqrt(2 * q2):.0f} nm -> results/knee_vs_cage.png")
