import sys
from pathlib import Path

import miepython
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "DPD tissue" / "code"))
from tile_sample import frame_positions

def scattering_vector(lam0=0.850, n_m=1.36, th_in_deg=30.0, th_out_deg=15.0):
    """
    Compute the scattering vector inside the sample for oblique illumination and collection.

    Parameters:
    lam0 : float
        Central wavelength in vacuum (in micrometers).
    n_m : float
        Refractive index of the medium around the scatterers.
    th_in_deg : float
        Illumination angle from the sample normal, in air (in degrees).
    th_out_deg : float
        Collection angle from the sample normal, in air, on the specular side (in degrees).

    Returns:
    q : 1D array
        Scattering vector (qx, qy, qz) in 1/micrometers, with z pointing out of the sample.
    theta_s : float
        Scattering angle inside the medium (in degrees).
    """
    k = 2 * np.pi / lam0

    # Refract both directions into the medium (Snell)
    ti = np.arcsin(np.sin(np.radians(th_in_deg)) / n_m)
    to = np.arcsin(np.sin(np.radians(th_out_deg)) / n_m)

    # Unit vectors of the incident and scattered waves
    s_in = np.array([np.sin(ti), 0.0, -np.cos(ti)])
    s_out = np.array([np.sin(to), 0.0, np.cos(to)])

    q = k * n_m * (s_in - s_out)
    theta_s = np.degrees(np.arccos(s_in @ s_out))

    return q, theta_s


def mie_amplitude(a, n_p, n_m, theta_deg, lam0=0.850, spread=0.05, n_avg=41):
    """
    Compute the unpolarized Mie scattering amplitude of a sphere at a given angle.

    Parameters:
    a : float
        Sphere radius (in micrometers).
    n_p : float
        Refractive index of the sphere.
    n_m : float
        Refractive index of the surrounding medium.
    theta_deg : float
        Scattering angle (in degrees).
    lam0 : float, optional
        Wavelength in vacuum (in micrometers). Default is 0.850.
    spread : float, optional
        Relative spread of radii to average over. Default is 0.05 (±5%).
    n_avg : int, optional
        Number of radii in the average. Default is 41.

    Returns:
    S : float
        Amplitude |S| = sqrt((|S1|^2 + |S2|^2) / 2), averaged over the radius spread.
    """
    mu = np.array([np.cos(np.radians(theta_deg))])

    # Average |S|^2 over a small spread of radii to smooth the Mie resonances
    S2 = []
    for r in a * np.linspace(1 - spread, 1 + spread, n_avg):
        x = 2 * np.pi * n_m / lam0 * r
        S1, S2_ = miepython.S1_S2(n_p / n_m, x, mu, norm='bohren')
        S2.append((abs(S1[0])**2 + abs(S2_[0])**2) / 2)

    return np.sqrt(np.mean(S2))

def object_amplitudes(layout, trajs, S_name, tiles=None):
    """
    Build the scattering amplitude of every organelle in the order frame_positions returns them.

    Parameters:
    layout : dict
        Tile layout from build_layout.
    trajs : list of dicts
        One traj.npz per run (from compress_dump).
    S_name : dict
        Amplitude |S| per organelle name, e.g. {"mitochondrion": 0.15}.
    tiles : 1D array, optional
        Tile indices to include. Default is all tiles.

    Returns:
    amps : 1D array
        Amplitude of each organelle.
    """
    tiles = np.arange(len(layout["run"])) if tiles is None else tiles
    return np.concatenate([[S_name[n] for n in trajs[layout["run"][t]]["type_names"]] for t in tiles])

def field_after_sample(k, U_ill, X, Y, layout, trajs, amps, q, tiles=None):
    """
    Compute the scattered field just after the sample for one hologram.

    Parameters:
    k : int
        Hologram index (0 to 1399).
    U_ill : 2D array
        Illumination field at the sample, without the tilt carrier.
    X, Y : 2D arrays
        Coordinate grids of U_ill (in micrometers).
    layout : dict
        Tile layout from build_layout.
    trajs : list of dicts
        One traj.npz per run (from compress_dump).
    amps : 1D array
        Amplitude of each organelle (from object_amplitudes, same tiles).
    q : 1D array
        Scattering vector (in 1/micrometers).
    tiles : 1D array, optional
        Tile indices to include. Default is all tiles.

    Returns:
    U : 2D array
        Complex field just after the sample.
    """
    # Organelle positions in micrometers (delay, orientation and tile centre already applied)
    r = frame_positions(layout, trajs, k, "com", tiles)

    # Nearest pixel of each organelle
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    ix = np.clip(np.rint((r[:, 0] - X[0, 0]) / dx).astype(int), 0, X.shape[1] - 1)
    iy = np.clip(np.rint((r[:, 1] - Y[0, 0]) / dy).astype(int), 0, X.shape[0] - 1)

    # Each organelle is a point source with phase q·r (this term carries the Doppler shift)
    U = np.zeros_like(U_ill)
    np.add.at(U, (iy, ix), U_ill[iy, ix] * amps * np.exp(1j * (r @ q)))

    return U