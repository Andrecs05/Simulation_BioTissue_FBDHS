import numpy as np

def angular_spectrum(U, dx, dy, z):
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

    Returns:
    U_z : 2D array
        Field at distance z.
    """
    # Get the size of the input field
    ny, nx = U.shape

    # Create frequency coordinates
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    
    # Create meshgrid for frequency coordinates
    FX, FY = np.meshgrid(fx, fy)

    # Compute the transfer function for free-space propagation
    H = np.exp(1j * 2 * np.pi * z * np.sqrt(1 - (FX**2 + FY**2)))

    # Compute the Fourier transform of the input field
    U_fft = np.fft.fft2(U)

    # Multiply by the transfer function
    U_z_fft = U_fft * H

    # Compute the inverse Fourier transform to get the propagated field
    U_z = np.fft.ifft2(U_z_fft)

    return U_z

def pupil(shape, radius):
    """
    Create a circular pupil function.

    Parameters:
    shape : tuple
        Shape of the output array (ny, nx).
    radius : float
        Radius of the circular pupil.

    Returns:
    P : 2D array
        Circular pupil function.
    """
    ny, nx = shape
    y = np.arange(-ny//2, ny//2)
    x = np.arange(-nx//2, nx//2)
    X, Y = np.meshgrid(x, y)
    
    # Create circular mask
    P = np.zeros((ny, nx))
    P[X**2 + Y**2 <= radius**2] = 1
    
    return P

def lens_transmitance(shape, f, wavelength, dx, dy, radius):
    """
    Create a lens transmittance function.

    Parameters:
    shape : tuple
        Shape of the output array (ny, nx).
    f : float
        Focal length of the lens.
    wavelength : float
        Wavelength of the light.
    dx : float
        Sampling interval in x direction.
    dy : float
        Sampling interval in y direction.
    radius : float
        Radius of the lens.

    Returns:
    L : 2D array
        Lens transmittance function.
    """
    ny, nx = shape
    y = np.arange(-ny//2, ny//2) * dy
    x = np.arange(-nx//2, nx//2) * dx
    X, Y = np.meshgrid(x, y)
    
    # Compute the phase delay introduced by the lens
    phase_delay = (np.pi / (wavelength * f)) * (X**2 + Y**2)

    pupil_mask = pupil(shape, radius)
    
    # Create lens transmittance function
    L = np.exp(-1j * phase_delay)
    L *= pupil_mask
    
    return L

def gaussian_beam(shape, w0, wavelength, dx, dy):
    """
    Create a Gaussian beam profile.

    Parameters:
    shape : tuple
        Shape of the output array (ny, nx).
    w0 : float
        Beam waist (radius at which the field amplitude drops to 1/e).
    wavelength : float
        Wavelength of the light.
    dx : float
        Sampling interval in x direction.
    dy : float
        Sampling interval in y direction.

    Returns:
    G : 2D array
        Gaussian beam profile.
    """
    ny, nx = shape
    y = np.arange(-ny//2, ny//2) * dy
    x = np.arange(-nx//2, nx//2) * dx
    X, Y = np.meshgrid(x, y)
    
    # Compute the Gaussian beam profile
    G = np.exp(-(X**2 + Y**2) / (w0**2))
    
    return G