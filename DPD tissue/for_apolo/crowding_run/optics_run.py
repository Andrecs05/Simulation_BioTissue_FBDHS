"""
FB-DHS optics for the crowding sweep, in parallel (Apolo).

For every crowding level in runs/manifest.csv it tiles a 1 mm sample with that level's runs
(tile_sample.build_layout), simulates the 1400 holograms of the EAFIT recording at n_k wavelengths
with the functions of DHS instrument/code, keeps the broadband sideband S+ = sum_j w_j S_j,
and finally fits the knee of its field PSD to get D.

Usage:
    python optics_run.py frames --task $SLURM_ARRAY_TASK_ID [--n-chunks C] [--workers W]
    python optics_run.py gather

    frames : array task t simulates chunk (t % C) of level (t // C), with W processes,
             and writes results/<level>/S_chunk_XX.npy
    gather : joins the chunks of every level into results/<level>/S.npy, computes the PSD and the knee,
             and writes results/summary.csv and results/spectra.png

The optics functions are imported from crowding_run/optics_code (copies of DHS instrument/code and
DPD tissue/code/tile_sample.py); --code points somewhere else, e.g. "DHS instrument/code" of a clone of the repo.
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")                 # one thread per process, the parallelism is over frames
import numpy as np

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- parameters (as in per_surface_analysis.ipynb)
OPT = dict(
    lam0=0.850, fwhm=0.060,        # source centre and FWHM (um)
    w0=2e3,                        # beam waist (um)
    z_M1=50e3, M1_r=3e3, M1_theta=30.0, z_sample=50e3,
    f1=75e3, R1=12.7e3,            # objective, 2f-2f
    R_IP=500.0,                    # field stop at the image plane
    f2=150e3,                      # Fourier lens
    beta_deg=5.0, ng_1=0.51,       # biprism half angle and (n_g - 1) of BK7
    d_B=140e3,                     # biprism to camera
    p=3.75, nx=1280, ny=960,       # camera
    n_m=1.36, th_in=30.0, th_out=15.0,
    fps=0.2, n_frames=1400,
)
# Mie spheres per organelle name: (radius um, index). Names must match the Masses comments of cell.data.
# make_runs.py: nucleus 8 um across; mitochondria 0.5 x 2 um capsules -> sphere of the same volume, 0.44 um
ORGANELLES = {"nucleus": (4.0, 1.39), "mitochondrion": (0.44, 1.41), "vesicle": (0.25, 1.40)}

# Field sampling (see README): 4 um with bilinear placement of the scatterers
DX = 4.0                           # object/image plane pixel (um)
HALF = 1024.0                      # half width of the object grid (um), holds the 1 mm sample
ILL_HALF = 4096.0                  # half width of the illumination grid (um), as the notebook's 4096 x 2 um
N_K = 11                           # wavelengths across +-3 sigma of the source


# ---------------------------------------------------------------- helpers
def read_manifest(path):
    """
    Read the run manifest and group its runs by crowding level.

    Parameters:
    path : Path
        CSV from make_runs.py, with at least the columns run_dir and level.

    Returns:
    levels : dict
        {level: list of row dicts}, in the order the levels first appear.
    """
    levels = {}
    for row in csv.DictReader(open(path)):
        row["path"] = run_path(path, row["run_dir"])
        levels.setdefault(row["level"], []).append(row)
    return levels


def box_rc(data_path):
    """
    Box length of a LAMMPS data file (in r_c), from its 'xlo xhi' line.

    Parameters:
    data_path : Path
        LAMMPS data file.

    Returns:
    L : float
        Box length along x.
    """
    for line in open(data_path):
        if "xlo xhi" in line:
            lo, hi = map(float, line.split()[:2])
            return hi - lo
    raise ValueError(f"no 'xlo xhi' line in {data_path}")


def make_frame_index(layout):
    """
    Group the tiles by (run, delay, orientation), so each group is placed with one array operation.

    Parameters:
    layout : dict
        Tile layout from build_layout.

    Returns:
    groups : list of tuples
        (run, delay, orient, tile indices) per group.
    """
    key = np.c_[layout["run"], layout["delay"], layout["orient"]]
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    return [(r, d, o, np.nonzero(inv.ravel() == g)[0]) for g, (r, d, o) in enumerate(uniq)]


def frame_positions_fast(layout, trajs, k, groups, n_org):
    """
    Same output as tile_sample.frame_positions(layout, trajs, k, "com"), looping over groups instead of tiles.

    Parameters:
    layout : dict
        Tile layout from build_layout.
    trajs : list of dicts
        One traj.npz per run, with at least "com".
    k : int
        Hologram index (0 to 1399).
    groups : list
        From make_frame_index.
    n_org : 1D int array
        Number of organelles in each run.

    Returns:
    r : 2D array
        Positions (in micrometers), shape (n_organelles_total, 3), in tile order.
    """
    start = np.r_[0, np.cumsum(n_org[layout["run"]])]
    out = np.empty((start[-1], 3))
    for run, delay, orient, tiles in groups:
        f = ts.EQUIL_FRAMES + delay * ts.DELAY_FRAMES + k
        p = trajs[run]["com"][f]                                       # (n_org, 3), r_c
        xy = p[:, :2] @ ts.ORIENT[orient].T
        rows = start[tiles][:, None] + np.arange(n_org[run])           # destination rows of these tiles
        out[rows, :2] = xy[None] * ts.RC_UM + layout["centre_um"][tiles][:, None]
        out[rows, 2] = p[:, 2] * ts.RC_UM
    return out


def field_after_sample_bilinear(r, U_ill, X, Y, amps, q):
    """
    Scattered field just after the sample, like doppler_scattering.field_after_sample but each point
    source is shared among its 4 nearest pixels (bilinear), so its image moves smoothly instead of
    jumping one pixel at a time.

    Parameters:
    r : 2D array
        Organelle positions (in micrometers), shape (n, 3).
    U_ill : 2D array
        Illumination field at the sample.
    X, Y : 2D arrays
        Coordinate grids of U_ill (in micrometers).
    amps : 1D array
        Mie amplitude of each organelle.
    q : 1D array
        Scattering vector (in 1/micrometers).

    Returns:
    U : 2D array
        Complex field just after the sample.
    """
    dx = X[0, 1] - X[0, 0]
    fx = (r[:, 0] - X[0, 0]) / dx
    fy = (r[:, 1] - Y[0, 0]) / dx
    ix = np.clip(np.floor(fx).astype(int), 0, X.shape[1] - 2)
    iy = np.clip(np.floor(fy).astype(int), 0, X.shape[0] - 2)
    wx, wy = np.clip(fx - ix, 0, 1), np.clip(fy - iy, 0, 1)

    # Point source with the Doppler phase q·r, lit by the illumination at its nearest pixel
    a = U_ill[np.rint(fy).astype(int).clip(0, X.shape[0] - 1), np.rint(fx).astype(int).clip(0, X.shape[1] - 1)]
    a = a * amps * np.exp(1j * (r @ q))

    U = np.zeros_like(U_ill)
    for ox, oy, w in ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)), (0, 1, (1 - wx) * wy), (1, 1, wx * wy)):
        np.add.at(U, (iy + oy, ix + ox), a * w)
    return U


def sideband_window(p, nx, ny, theta, lam_min, lam_max, radius, e_deg=45.0):
    """
    Stadium window that follows the +1 carrier from lam_max to lam_min (a circle if they are equal).

    Parameters:
    p : float
        Camera pixel (um).
    nx, ny : int
        Camera size (pixels).
    theta : float
        Biprism deflection (rad).
    lam_min, lam_max : float
        Wavelength range (um).
    radius : float
        Window radius (1/um).
    e_deg : float, optional
        Direction normal to the biprism apex line (deg). Default is 45.

    Returns:
    mask : 2D bool array
        Window on the fftshifted hologram spectrum.
    """
    e = np.array([np.cos(np.radians(e_deg)), np.sin(np.radians(e_deg))])
    FX, FY = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(nx, p)), np.fft.fftshift(np.fft.fftfreq(ny, p)))
    t = np.clip(FX * e[0] + FY * e[1], 2 * theta / lam_max, 2 * theta / lam_min)
    return np.hypot(FX - t * e[0], FY - t * e[1]) <= radius


# ---------------------------------------------------------------- set-up of one level (done once, before forking)
G = {}


def setup(rows, n_k, dx):
    """Fill G with everything a frame needs: layout, trajectories, and per wavelength the illumination,
    q, Mie amplitudes, grids, lens and field stop."""
    o = OPT
    trajs = [dict(np.load(r["path"] / "traj.npz")) for r in rows]
    trajs = [{"com": t["com"], "type_names": t["type_names"]} for t in trajs]   # drop the membrane
    need = ts.EQUIL_FRAMES + (ts.N_DELAYS - 1) * ts.DELAY_FRAMES + o["n_frames"]
    short = [str(r["path"]) for r, t in zip(rows, trajs) if len(t["com"]) < need]
    if short:
        sys.exit(f"these runs have fewer than {need} frames: {short}")

    ts.BOX_RC = box_rc(rows[0]["path"] / "cell.data")
    layout = ts.build_layout(len(trajs), seed=0)
    G.update(layout=layout, trajs=trajs, groups=make_frame_index(layout),
             n_org=np.array([len(t["type_names"]) for t in trajs]))

    # Source spectrum: n_k wavelengths and their power weights
    k_src, w_src = source_spectrum(o["lam0"], o["fwhm"], n_k=n_k)
    lam_src = 2 * np.pi / k_src if n_k > 1 else np.array([o["lam0"]])
    w_src = w_src if n_k > 1 else np.array([1.0])
    theta = o["ng_1"] * np.radians(o["beta_deg"])

    # Object grid and camera
    N = int(2 * round(HALF / dx))
    Xc, Yc = make_grid((N, N), dx, dx)
    Ni = int(2 * round(ILL_HALF / dx))
    X, Y = make_grid((Ni, Ni), dx, dx)
    s = slice(Ni // 2 - N // 2, Ni // 2 + N // 2)
    xc = (np.arange(o["nx"]) - o["nx"] // 2) * o["p"]
    yc = (np.arange(o["ny"]) - o["ny"] // 2) * o["p"]

    per_lam = []
    for lam in lam_src:
        # Illumination: source -> mirror M1 -> sample (cells 3-9 of the notebook)
        U = source_field(X, Y, lam, o["w0"])
        U = angular_spectrum(U, dx, dx, o["z_M1"], lam) * mirror_pupil(X, Y, o["M1_r"], o["M1_theta"])
        U = angular_spectrum(U, dx, dx, o["z_sample"], lam)

        q, th = scattering_vector(lam, o["n_m"], o["th_in"], o["th_out"])
        S_name = {n: mie_amplitude(a, n_p, o["n_m"], th, lam0=lam) for n, (a, n_p) in ORGANELLES.items()}

        # Grids fresnel_fft returns at this wavelength (lens plane, image plane)
        d1 = lam * 2 * o["f1"] / (N * dx)
        X1, Y1 = make_grid((N, N), d1, d1)
        di = lam * 2 * o["f1"] / (N * d1)
        Xi, Yi = make_grid((N, N), di, di)
        per_lam.append(dict(lam=lam, U_ill=U[s, s].copy(), q=q,
                            amps=object_amplitudes(layout, trajs, S_name),
                            X1=X1, Y1=Y1, L1=lens_transmitance(X1, Y1, o["f1"], lam, o["R1"]),
                            P_IP=pupil(Xi, Yi, o["R_IP"])))
        del U

    radius = 1.3 * 2 * o["R_IP"] / (lam_src.min() * o["f2"])
    mask = sideband_window(o["p"], o["nx"], o["ny"], theta, lam_src.min(), lam_src.max(), radius)
    G.update(lam_src=lam_src, w_src=w_src / w_src.sum(), theta=theta, Xc=Xc, Yc=Yc, xc=xc, yc=yc,
             per_lam=per_lam, mask=mask)


def simulate_frame(k):
    """
    Broadband sideband of hologram k: sum over wavelengths of w_j S_j.

    Parameters:
    k : int
        Hologram index.

    Returns:
    S : 1D complex64 array
        S+ inside the sideband window.
    """
    o, g = OPT, G
    r = frame_positions_fast(g["layout"], g["trajs"], k, g["groups"], g["n_org"])
    S = 0
    for wj, pl in zip(g["w_src"], g["per_lam"]):
        lam = pl["lam"]
        U = field_after_sample_bilinear(r, pl["U_ill"], g["Xc"], g["Yc"], pl["amps"], pl["q"])
        U, _, _ = fresnel_fft(U, g["Xc"], g["Yc"], lam, 2 * o["f1"])             # to the objective
        U, Xi, Yi = fresnel_fft(U * pl["L1"], pl["X1"], pl["Y1"], lam, 2 * o["f1"])  # to the image plane
        U, Xi, Yi = crop(U * pl["P_IP"], Xi, Yi, o["R_IP"] + 2 * (Xi[0, 1] - Xi[0, 0]))  # only the field stop
        I = camera_hologram(U, Xi, Yi, lam, o["f2"], g["xc"], g["yc"], g["theta"], o["d_B"])
        F = np.fft.fftshift(np.fft.fft2(I - I.mean()))
        S = S + wj * F[g["mask"]]
    return S.astype(np.complex64)


# ---------------------------------------------------------------- commands
def cmd_frames(a):
    levels = read_manifest(a.manifest)
    names = list(levels)
    level = names[a.task // a.n_chunks]
    chunk = a.task % a.n_chunks
    frames = np.array_split(np.arange(OPT["n_frames"]), a.n_chunks)[chunk]
    if a.max_frames:
        frames = frames[:a.max_frames]
    out = a.results / level
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"S_chunk_{chunk:02d}.npy"
    if path.exists():
        print(f"{path} exists, nothing to do")
        return

    t = time.time()
    setup(levels[level], a.n_k, a.dx)
    print(f"level {level}: {len(levels[level])} runs, {len(G['layout']['run'])} tiles, "
          f"{G['n_org'][G['layout']['run']].sum()} organelles, {len(G['lam_src'])} wavelengths, "
          f"{G['mask'].sum()} sideband px; set-up {time.time() - t:.0f} s", flush=True)

    t = time.time()
    with get_context("fork").Pool(a.workers) as pool:
        S = np.array(pool.map(simulate_frame, frames, chunksize=1))
    np.save(path, S)
    print(f"frames {frames[0]}-{frames[-1]} -> {path}: {time.time() - t:.0f} s", flush=True)


def cmd_gather(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = read_manifest(a.manifest)
    k_src, w_src = source_spectrum(OPT["lam0"], OPT["fwhm"], n_k=a.n_k)
    lam_src = 2 * np.pi / k_src if a.n_k > 1 else np.array([OPT["lam0"]])
    w_src = w_src / w_src.sum() if a.n_k > 1 else np.array([1.0])
    q2_eff = sum(w * np.sum(scattering_vector(l, OPT["n_m"], OPT["th_in"], OPT["th_out"])[0]**2)
                 for w, l in zip(w_src, lam_src))

    rows, fig = [], plt.figure(figsize=(7, 5))
    for level, runs in levels.items():
        chunks = sorted((a.results / level).glob("S_chunk_*.npy"))
        if not chunks:
            print(f"{level}: no chunks yet"); continue
        S = np.concatenate([np.load(c) for c in chunks])
        np.save(a.results / level / "S.npy", S)
        f, P = field_spectrum(S, OPT["fps"])
        fk = knee_frequency(f, P, OPT["fps"])
        D = np.pi * fk / q2_eff
        np.savez(a.results / level / "spectrum.npz", f=f, P=P, fk=fk, D=D)
        meta = {k: v for k, v in runs[0].items() if k not in ("run_dir", "path")}
        rows.append(dict(meta, n_runs=len(runs), n_frames=len(S), knee_mHz=fk * 1e3, D_um2_s=D))
        print(f"{level}: {len(S)} frames, knee {fk * 1e3:.2f} mHz, D = {D:.3e} um^2/s")
        plt.loglog(*log_bin(f, P), "o-", ms=3, label=f"{level}: {fk * 1e3:.1f} mHz")

    plt.xlabel("f (Hz)"); plt.ylabel("PSD of S+"); plt.legend(fontsize=8); plt.tight_layout()
    fig.savefig(a.results / "spectra.png", dpi=150)
    with open(a.results / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"-> {a.results / 'summary.csv'}, {a.results / 'spectra.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["frames", "gather"])
    ap.add_argument("--code", type=Path, default=HERE / "optics_code",
                    help="folder with utilities.py, ilumination.py, doppler_scattering.py, collection.py, "
                         "spectrum.py (and tile_sample.py unless doppler_scattering finds it in the repo)")
    ap.add_argument("--manifest", type=Path, default=HERE / "runs" / "manifest.csv")
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--task", type=int, default=0, help="SLURM_ARRAY_TASK_ID")
    ap.add_argument("--n-chunks", type=int, default=1, help="frame chunks per level")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
    ap.add_argument("--n-k", type=int, default=N_K, help="wavelengths (1 = monochromatic at lam0)")
    ap.add_argument("--dx", type=float, default=DX, help="object plane pixel (um)")
    ap.add_argument("--max-frames", type=int, default=0, help="only the first n frames of the chunk (tests)")
    a = ap.parse_args()

    sys.path.insert(0, str(a.code))
    from utilities import make_grid, angular_spectrum, pupil, fresnel_fft, crop
    from ilumination import source_field, source_spectrum, mirror_pupil
    from doppler_scattering import scattering_vector, mie_amplitude, object_amplitudes
    from collection import lens_transmitance, camera_hologram
    from spectrum import field_spectrum, log_bin, knee_frequency
    import tile_sample as ts
    sys.path.insert(0, str(HERE))
    from compress_dumps import run_path

    {"frames": cmd_frames, "gather": cmd_gather}[a.command](a)
