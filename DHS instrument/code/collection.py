import numpy as np
from utilities import pupil

def lens_transmitance(X, Y, f, lam, radius):
    """
    Create a lens transmittance function.

    Parameters:
    X, Y : 2D arrays
        Coordinate grids.
    f : float
        Focal length of the lens.
    lam : float
        Wavelength of the light.
    radius : float
        Radius of the lens (same units as X, Y).

    Returns:
    L : 2D array
        Lens transmittance function.
    """
    # Compute the phase delay introduced by the lens
    phase_delay = (np.pi / (lam * f)) * (X**2 + Y**2)

    pupil_mask = pupil(X, Y, radius)

    # Create lens transmittance function
    L = np.exp(-1j * phase_delay)
    L *= pupil_mask

    return L

def fourier_lens_mft(U, X, Y, lam, f2, xc, yc):
    """
    Field at the back focal plane of a lens, evaluated directly on chosen camera coordinates.

    Parameters:
    U : 2D array
        Field at the front focal plane.
    X, Y : 2D arrays
        Coordinate grids of U (in micrometers).
    lam : float
        Wavelength (in micrometers).
    f2 : float
        Focal length of the Fourier lens (in micrometers).
    xc, yc : 1D arrays
        Camera coordinates (in micrometers).

    Returns:
    U_F : 2D array
        Field at the camera, shape (len(yc), len(xc)).
    """
    x, y = X[0], Y[:, 0]
    dx, dy = x[1] - x[0], y[1] - y[0]

    # Fourier kernels: one matrix per axis
    Ex = np.exp(-2j * np.pi * np.outer(xc, x) / (lam * f2))
    Ey = np.exp(-2j * np.pi * np.outer(yc, y) / (lam * f2))

    return (Ey @ U @ Ex.T) * dx * dy / (1j * lam * f2)


def camera_hologram(U_i, X, Y, lam, f2, xc, yc, theta, d, e_deg=45.0):
    """
    Hologram on the camera for the Fresnel biprism after the Fourier lens.

    Parameters:
    U_i : 2D array
        Field at the image plane.
    X, Y : 2D arrays
        Coordinate grids of U_i (in micrometers).
    lam : float
        Wavelength (in micrometers).
    f2 : float
        Focal length of the Fourier lens (in micrometers).
    xc, yc : 1D arrays
        Camera pixel coordinates (in micrometers).
    theta : float
        Deflection of each biprism half, (n_g - 1) * beta (in radians).
    d : float
        Distance from the biprism to the camera (in micrometers).
    e_deg : float, optional
        Direction normal to the biprism apex line, from the pixel x axis (in degrees). Default is 45.

    Returns:
    I : 2D array
        Hologram intensity.
    """
    e = np.array([np.cos(np.radians(e_deg)), np.sin(np.radians(e_deg))])
    eta = theta * d                         # shear
    K = 2 * (2 * np.pi / lam) * theta       # carrier, fringe period lam / (2 theta)
    XC, YC = np.meshgrid(xc, yc)
    xb = XC * e[0] + YC * e[1]              # coordinate across the apex line

    # Each half: Fourier field at the shifted coordinates, restricted to its half of the aperture
    Up = fourier_lens_mft(U_i, X, Y, lam, f2, xc + eta * e[0], yc + eta * e[1])
    Um = fourier_lens_mft(U_i, X, Y, lam, f2, xc - eta * e[0], yc - eta * e[1])

    # Each half tilted by its prism
    Uc = (Up * (xb + eta > 0) * np.exp(-1j * K * xb / 2)
          + Um * (xb - eta < 0) * np.exp(1j * K * xb / 2))

    return np.abs(Uc)**2

def sideband(I, p, Lambda, e_deg=45.0, radius=None):
    """
    Extract the +1 sideband S+ from a hologram.

    Parameters:
    I : 2D array
        Hologram intensity.
    p : float
        Camera pixel size (in micrometers).
    Lambda : float
        Fringe period lam / (2 theta) (in micrometers).
    e_deg : float, optional
        Direction normal to the biprism apex line (in degrees). Default is 45.
    radius : float, optional
        Radius of the sideband window (in 1/micrometers). Default is 0.5 / Lambda.

    Returns:
    S : 1D array
        Complex S+ at every frequency pixel inside the window.
    """
    ny, nx = I.shape
    e = np.array([np.cos(np.radians(e_deg)), np.sin(np.radians(e_deg))])
    radius = 0.5 / Lambda if radius is None else radius

    # Spectrum of the hologram with the mean removed
    F = np.fft.fftshift(np.fft.fft2(I - I.mean()))
    FX, FY = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(nx, p)), np.fft.fftshift(np.fft.fftfreq(ny, p)))

    # Circular window around the carrier
    c = e / Lambda
    mask = np.hypot(FX - c[0], FY - c[1]) <= radius

    return F[mask]