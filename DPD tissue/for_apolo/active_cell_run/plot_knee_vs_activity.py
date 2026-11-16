"""Knee and moving fraction of the signal against swimming speed, free and caged, with the prediction for active
Brownian particles (no walls, no other mitochondria).

Run from active_cell_run after gather:
    MPLBACKEND=Agg python plot_knee_vs_activity.py
Reads results/summary.csv (columns v0_nm_s, cage_um, knee_mHz, D_um2_s) and results/<level>/S.npy,
writes results/knee_vs_activity.png and prints measured against predicted knees for every level.

Prediction: N_PRED independent active Brownian particles with the D0, TAU_RUN and cage of make_runs.py are moved
in numpy, their field factor F(t) = <exp(i q . (r(t) - r(0)))> is computed, and S+ decorrelates as |F|^2
(two independent mitochondria per term). The moving part |F(t)|^2 - |F(inf)|^2, sampled every 5 s, is turned
into a spectrum and fitted with the same knee_frequency as gather.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "shared_optics" / "optics_code"))
sys.path.insert(0, str(here))
from spectrum import knee_frequency
from make_runs import D0, TAU_RUN

FPS = 0.2              # holograms per second
N_FRAMES = 1400        # holograms in the recording
N_PRED = 2000          # particles in the prediction
SUB = 20               # prediction steps per frame (0.25 s)


def predicted(v_nm_s, cage_um, q2, seed=0):
    """
    Knee (Hz) of S+ for free active Brownian particles, from a direct simulation.

    Parameters:
    v_nm_s : float
        Swimming speed (nm/s).
    cage_um : float
        Cage size (um rms per axis); 0 for none.
    q2 : float
        <|q|^2> of the instrument (um^-2); q is taken along z.

    Returns:
    fk : float
        Knee from the same Lorentzian fit as gather (Hz).
    """
    rng = np.random.default_rng(seed)
    dt = 1 / FPS / SUB
    v, q = v_nm_s * 1e-3, np.sqrt(q2)
    # Start in the cage (or at 0), with random directions
    x = rng.normal(0, cage_um, (N_PRED, 3)) if cage_um else np.zeros((N_PRED, 3))
    u = rng.normal(size=(N_PRED, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    z = np.empty((N_FRAMES, N_PRED), complex)
    for n in range(N_FRAMES):
        for _ in range(SUB):
            drift = v * u - (D0 / cage_um ** 2 * x if cage_um else 0)
            x = x + drift * dt + rng.normal(0, np.sqrt(2 * D0 * dt), x.shape)
            # Rotational diffusion of the direction: <u(t).u(0)> = exp(-t / TAU_RUN)
            u = u + rng.normal(0, np.sqrt(dt / TAU_RUN), u.shape)
            u /= np.linalg.norm(u, axis=1)[:, None]
        z[n] = np.exp(1j * q * x[:, 2])
    # F(lag), time-averaged per particle with FFTs, then averaged over particles
    Z = np.fft.fft(z, 2 * N_FRAMES, axis=0)
    F = np.fft.ifft(np.abs(Z) ** 2, axis=0)[:N_FRAMES].real.mean(axis=1) / (N_FRAMES - np.arange(N_FRAMES))
    F_inf = (np.abs(z.mean(axis=0)) ** 2).mean()               # what never decorrelates (the cage)
    C = F ** 2 - F_inf ** 2                                    # moving part of |F|^2
    t = np.arange(N_FRAMES) / FPS
    f = np.fft.rfftfreq(2 * N_FRAMES, 1 / FPS)[1:N_FRAMES]
    w = np.ones(N_FRAMES); w[0] = 0.5
    P = 2 * (w * C * np.cos(2 * np.pi * f[:, None] * t)).sum(axis=1) / FPS
    return knee_frequency(f, np.maximum(P, 1e-6 * P.max()), FPS)


def moving_fraction(level):
    """Measured fraction of the S+ variance that changes in time, |S - <S>_t|^2 / |S|^2 over all pixels."""
    S = np.load(Path("results") / level / "S.npy")
    dS = S - S.mean(axis=0)
    return (np.abs(dS) ** 2).mean() / (np.abs(S) ** 2).mean()


rows = list(csv.DictReader(open("results/summary.csv")))
for r in rows:
    r["v"], r["cage_um"] = float(r["v0_nm_s"]), float(r["cage_um"])
    r["fk"] = float(r["knee_mHz"]) * 1e-3
    r["moving"] = moving_fraction(r["level"])
q2 = np.median([np.pi * r["fk"] / float(r["D_um2_s"]) for r in rows])     # <|q|^2> used by gather

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
vs = np.array([0, 1, 2, 3, 5, 7, 10, 15, 20, 30])
for j, cage in enumerate(sorted({r["cage_um"] for r in rows})):
    c = f"C{j}"
    lab = f"cage {cage * 1e3:.0f} nm" if cage else "no cage"
    sel = sorted((r for r in rows if r["cage_um"] == cage), key=lambda r: r["v"])
    pred = {v: predicted(v, cage, q2) for v in vs}              # every measured v0 is in vs
    ax[0].semilogy(vs, [pred[v] * 1e3 for v in vs], "-", color=c, lw=1, label=f"prediction, {lab}")
    ax[0].semilogy([r["v"] for r in sel], [r["fk"] * 1e3 for r in sel], "o", color=c, label=f"cell, {lab}")
    ax[1].plot([r["v"] for r in sel], [r["moving"] for r in sel], "o-", color=c, label=f"cell, {lab}")
    for r in sel:
        print(f"{r['level']:11s} knee {r['fk'] * 1e3:7.2f} mHz (predicted {pred[r['v']] * 1e3:7.2f}), "
              f"moving fraction {r['moving']:.3f}")

ax[0].axhline(FPS / 2 * 1e3, color="tab:red", lw=1, label="Nyquist")
ax[0].set(xlabel="swimming speed v0 (nm/s)", ylabel="knee (mHz)", title="Knee")
ax[0].legend(fontsize=7)
ax[1].set(xlabel="swimming speed v0 (nm/s)", ylabel="moving / total power of S+",
          title="Fraction of the signal that moves", ylim=(0, None))
ax[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig("results/knee_vs_activity.png", dpi=150)
print(f"<|q|^2> = {q2:.1f} um^-2 -> results/knee_vs_activity.png")
