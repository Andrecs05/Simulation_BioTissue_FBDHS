import numpy as np
import matplotlib.pyplot as plt

def make_grid(n, dx, dy):
    """
    Create centred 2D coordinate grids.

    Parameters:
    n : tuple or array
        Number of pixels (ny, nx).
    dx, dy : float
        Pixel size in x and y.

    Returns:
    X, Y : 2D arrays
        Coordinate grids of shape (ny, nx).
    """
    ny, nx = n
    y = np.arange(-ny//2, ny//2) * dy
    x = np.arange(-nx//2, nx//2) * dx
    return np.meshgrid(x, y)


def angular_spectrum(U, dx, dy, z, lam, n=1.0):
    """
    Compute the angular spectrum of a 2D field U at a distance z.

    Parameters:
    U : 2D array
        Input field (complex amplitude).
    dx : float
        Sampling interval in x direction.
    dy : float
        Sampling interval in y direction.
    z : float
        Propagation distance.
    lam : float
        Wavelength of the light.
    n : float, optional
        Refractive index of the medium (default is 1.0 for air).

    Returns:
    U_z : 2D array
        Field at distance z.
    """
    # Get the size of the input field
    ny, nx = U.shape
    
    # Create meshgrid for frequency coordinates
    FX, FY = np.meshgrid(np.fft.fftfreq(nx, d=dx), np.fft.fftfreq(ny, d=dy))

    # Compute the transfer function for free-space propagation
    arg = 1 - (lam*FX/n)**2 - (lam*FY/n)**2
    H = np.exp(1j * 2 * np.pi * n * z/lam * np.sqrt(arg.astype(complex)))

    return np.fft.ifft2(np.fft.fft2(U)*H)

def pupil(X, Y, radius):
    """
    Create a circular pupil function.

    Parameters:
    X, Y : 2D arrays
        Coordinate grids.
    radius : float
        Radius of the circular pupil (same units as X, Y).

    Returns:
    P : 2D array
        Circular pupil function.
    """
    return ((X**2 + Y**2) <= radius**2).astype(float)

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


def gaussian_beam(x, y, z, k, w0, iris_radius=None):
    """
    Generate a Gaussian beam field.

    Parameters:
    x : 2D array
        X coordinates.
    y : 2D array
        Y coordinates.
    z : float
        Propagation distance.
    k : float
        Wavenumber (2 * pi / wavelength).
    w0 : float
        Beam waist radius.
    iris_radius : float, optional
        Radius of the iris to limit the beam. If None, no iris is applied.

    Returns:
    U : 2D array
        Complex amplitude of the Gaussian beam.
    """
    # Compute the Rayleigh range (z_R = pi*w0^2/lambda = k*w0^2/2)
    z_R = k * w0**2 / 2

    # Compute the beam radius at distance z
    w_z = w0 * np.sqrt(1 + (z / z_R)**2)

    # Compute the Gouy phase
    gouy_phase = np.arctan(z / z_R)

    # Wavefront curvature 1/R(z) = z / (z^2 + z_R^2); finite everywhere, 0 at the waist
    inv_R = z / (z**2 + z_R**2)

    # Compute the complex amplitude of the Gaussian beam
    U = (w0 / w_z) * np.exp(-(x**2 + y**2) / w_z**2) * np.exp(-1j * (k * z + k * (x**2 + y**2) * inv_R / 2 - gouy_phase))


    # Apply iris if specified
    if iris_radius is not None:
        iris_mask = pupil(x, y, iris_radius)
        U *= iris_mask

    return U

def plot_field(U, X, Y, title, z=0, lam=None, thresh=1e-2):
    """
    Plot amplitude and phase of a complex field side by side.

    Parameters:
    U : 2D array
        Complex field.
    X, Y : 2D arrays
        Coordinate grids.
    title : str
        Label for the plane (e.g. 'at Mirror').
    z : float, optional
        Total propagation distance from the source; its k*z piston is removed from the phase.
    lam : float, optional
        Wavelength. Required if z != 0.
    thresh : float, optional
        Phase is hidden where amplitude < thresh * max amplitude.
    """
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    amp = np.abs(U)

    # Remove the constant propagation phase, then hide phase where there is no light
    piston = np.exp(-1j * 2 * np.pi / lam * z) if z != 0 else 1
    phase = np.angle(U * piston)
    phase[amp < thresh * amp.max()] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(amp, extent=extent, origin='lower', cmap='gray')
    axes[0].set_title(f'Field Amplitude {title}')
    axes[0].set_xlabel('x (µm)'); axes[0].set_ylabel('y (µm)')
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(phase, extent=extent, origin='lower', cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title(f'Field Phase {title}')
    axes[1].set_xlabel('x (µm)'); axes[1].set_ylabel('y (µm)')
    fig.colorbar(im1, ax=axes[1], label='rad')

    plt.tight_layout()
    plt.show()