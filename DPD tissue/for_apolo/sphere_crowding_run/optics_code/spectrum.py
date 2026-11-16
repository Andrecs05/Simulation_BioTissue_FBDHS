import numpy as np


def field_spectrum(S, fps=0.2):
    """
    Power spectrum of the complex sideband field (EAFIT processing), averaged over pixels.

    Parameters:
    S : 2D array
        Complex S+ with shape (n_frames, n_pixels).
    fps : float, optional
        Frame rate (in Hz). Default is 0.2 (one hologram every 5 s).

    Returns:
    f : 1D array
        Positive frequencies (in Hz).
    P : 1D array
        PSD with the +f and -f halves added, averaged over pixels.
    """
    # Remove the static part of each pixel, then window in time
    dS = S - S.mean(axis=0)
    w = np.hanning(len(S))[:, None]

    # PSD per pixel, averaged over pixels
    P = (np.abs(np.fft.fft(dS * w, axis=0))**2).mean(axis=1)
    f = np.fft.fftfreq(len(S), 1 / fps)

    # Fold negative frequencies onto positive ones (S+ is complex, so they differ)
    n = len(S) // 2
    return f[1:n], (P[1:n] + P[-1:-n:-1]) / (fps * (w**2).sum())


def log_bin(f, P, n=40):
    """
    Average a spectrum in logarithmically spaced frequency bins.

    Parameters:
    f : 1D array
        Frequencies (in Hz).
    P : 1D array
        Spectrum.
    n : int, optional
        Number of bins. Default is 40.

    Returns:
    fb, Pb : 1D arrays
        Bin-averaged frequencies and spectrum (empty bins are skipped).
    """
    edges = np.logspace(np.log10(f[0]), np.log10(f[-1]), n + 1)
    idx = np.digitize(f, edges) - 1
    keep = [k for k in range(n) if np.any(idx == k)]
    return (np.array([f[idx == k].mean() for k in keep]),
            np.array([P[idx == k].mean() for k in keep]))


def knee_frequency(f, P, fps=0.2):
    """
    Fit a Lorentzian A / (1 + (f/fk)^2) to the spectrum and return its knee.

    Parameters:
    f : 1D array
        Frequencies (in Hz).
    P : 1D array
        Spectrum from field_spectrum.
    fps : float, optional
        Frame rate (in Hz), used to include aliasing from above Nyquist. Default is 0.2.

    Returns:
    fk : float
        Knee frequency (in Hz).
    """
    fb, Pb = log_bin(f, P)

    # Aliased copies of the Lorentzian, since the camera samples a continuous motion
    images = np.arange(-20, 21)[:, None] * fps

    # Grid search on fk, least squares in log P with the amplitude solved at each fk
    best = (np.inf, np.nan)
    for fk in np.logspace(np.log10(f[0]), np.log10(10 * fps), 400):
        m = (1 / (1 + ((fb + images) / fk)**2)).sum(axis=0)
        logA = np.mean(np.log(Pb) - np.log(m))
        err = np.sum((np.log(Pb) - logA - np.log(m))**2)
        if err < best[0]:
            best = (err, fk)

    return best[1]