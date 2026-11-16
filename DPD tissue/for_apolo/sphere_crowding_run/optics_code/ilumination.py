import numpy as np
from utilities import angular_spectrum, gaussian_beam

def source_field(X, Y, lam, w0, z=0, n=1.0):
    """
    Generate the illumination field at a given distance z.

    Parameters:
    X, Y : 2D arrays
        Coordinate grids (from make_grid or np.meshgrid).
    lam : float
        Wavelength of the light.
    w0 : float
        Beam waist radius.
    z : float, optional
        Propagation distance. Default is 0.
    n : float, optional
        Refractive index of the medium. Default is 1.0 (air).

    Returns:
    U : 2D array
        Complex amplitude of the illumination field.
    """
    # Pixel sizes from the grid spacing
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    # Generate a Gaussian beam at z=0
    k = 2 * np.pi / lam  # Wavenumber
    U0 = gaussian_beam(X, Y, 0, k, w0)

    # Propagate the field to distance z using angular spectrum method
    U = angular_spectrum(U0, dx, dy, z, lam, n)

    return U


def source_spectrum(lam0=0.850, dlam=0.060, n_k=24, n_sigma=3.0):
    """
    Create a source spectrum with a Gaussian profile.

    Parameters:
    lam0 : float
        Central wavelength (in micrometers).
    dlam : float
        Full width at half maximum (FWHM) of the spectrum (in micrometers).
    n_k : int
        Number of spectral samples.
    n_sigma : float
        Number of standard deviations to include in the spectrum.

    Returns:
    k : 1D array
        Wavenumber array corresponding to the spectral samples.
    S : 1D array
        Source spectrum (normalized).
    """
    # Convert wavelength to wavenumber
    k0 = 2 * np.pi / lam0
    dk = 2 * np.pi * dlam / (lam0**2)
    sigma_k = dk / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
    
    # Create wavenumber array centered around k0
    k = np.linspace(k0 - n_sigma * sigma_k, k0 + n_sigma * sigma_k, n_k)
    
    # Create Gaussian spectrum
    S = np.exp(-((k - k0)**2) / (2 * (sigma_k)**2))
    
    # Normalize the spectrum
    S /= np.sum(S)
    
    return k, S

def mirror_pupil(X, Y, radius, theta_deg):
    """
    Create a elliptical pupil function for a mirror with a given radius and orientation.

    Parameters:
    X, Y : 2D arrays
        Coordinate grids.
    radius : float
        Radius of the mirror.
    angle : float
        Angle of rotation of the mirror (in degrees).

    Returns:
    P : 2D array
        Pupil function of the mirror.
    """
    a = radius
    b = radius * np.cos(np.radians(theta_deg))

    # Create the elliptical pupil function
    P = ((X**2 / a**2) + (Y**2 / b**2) <= 1.0).astype(float)
    return P
