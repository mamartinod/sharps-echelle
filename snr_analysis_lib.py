import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def photon_number(t_exp, mag, throughput, S_tel, N_tel, dl, QE, phi_0):
    """Calculate the number of photo-events of a SINGLE beam during an exposure.

    Parameters
    ----------
    t_exp : float
        Exposure time in seconds.
    mag : float
        Magnitude of the target star in the J band.
    throughput : float
        Total throughput of the system, including telescope, fiber coupling, spectrograph, etc.
    S_tel : float
        Collecting area of a single telescope in m².
    N_tel : int
        Total number of telescopes used by the combiner.
    dl : float
        Spectral bin width in um.
    QE : float
        Quantum efficiency of the detector.
    phi_0 : float
        Reference photon flux for a zero-magnitude star, in photons/m²/um/s.
    Returns
    -------
    float
        Number of photo-events of a SINGLE beam during the exposure.
    """
    
    n_ph = throughput * S_tel * 1/(N_tel-1) * dl * QE * t_exp * phi_0 * 10**(-0.4 * mag)
    return n_ph

def photon_noise(n_phot, V, n_tel):
    """
    Return the variance of the photon noise in a fringe sampled on an ABCD system.


    Parameters
    ----------
    n_phot : float
        Number of photo-events of a SINGLE beam.
    V : float
        Fringe visibility (contrast), between 0 and 1.
    n_tel : int
        Number of telescopes contributing to the illumination of the fringe pattern.

    Returns
    -------
    float
        Variance of the photon noise [e-²], including both the Poisson
        shot noise term and the speckle noise contribution.

    Notes
    -----
    See main notes above for the derivation of this formula, inspired by Colavita (PASP, 1999).
    """
    n_ph_tot = n_tel * n_phot
    return 2 * n_ph_tot * (n_phot * V)**2 + n_ph_tot**2

def dark_current(dk_rate, t_exp, n_pix):
    """
    Return the variance of the dark current accumulated over an exposure.

    Parameters
    ----------
    dk_rate : float
        Dark current rate [e-/px/s]. For this detector, the nominal value
        is 0.1 e-/px/ksec (i.e. 1e-4 e-/px/s).
    t_exp : float
        Exposure time [s].
    n_pix : int
        Number of pixels over which the dark current is summed.

    Returns
    -------
    float
        Variance of the dark current [e-²] accumulated across all pixels during the
        exposure, computed as ``dk_rate**2 * t_exp * n_pix``.
    """
    return dk_rate**2 * t_exp * n_pix

def ron(BV, npix, glow, cds_mode):
    """
    Return the read noise as a function of bias voltage.

    Interpolates linearly from a table of measured read noise values.

    Parameters
    ----------
    BV : float or array_like
        Bias voltage [V]. Should be within the measured range [3, 14] V;
        values outside this range are clipped to the boundary values by
        ``numpy.interp``.
    npix : int
        Number of pixels over which the read noise is summed.
    glow : float
        Glow level [e-/px/frame] for the read noise calculation.
    cds_mode : bool, optional
        If True, the read noise is calculated for correlated double sampling (CDS) mode, which doubles the read noise variance.

    Returns
    -------
    float or ndarray
        Variance of the read noise [e-²] accumulated across all pixels at the
        requested bias voltage(s).

    """
    BV0 = np.arange(3, 15) # Bias voltage range to explore in Volt
    ron_values = np.array([13.3, 11.3, 9, 7, 5.5, 4, 3, 2, 1.5, 1, 0.8, 0.4])

    rn = np.interp(BV, BV0, ron_values)

    if cds_mode:
        rn = rn * 2**0.5 # CDS mode doubles the read noise variance
        
    return (rn**2 + glow**2) * npix

def total_noise(n_ph, visi, n_tel, dk_rate, t_exp, n_pix, BV, glow, nb_frames, cds_mode=False):
    """Calculate the total noise variance for a fringe measurement.
    
    Parameters
    ----------
    n_ph : float
        Number of photo-events of a SINGLE beam.
    visi : float
        Fringe visibility (contrast), between 0 and 1.
    n_tel : int
        Number of telescopes contributing to the illumination of the fringe pattern.
    dk_rate : float
        Dark current rate [e-/px/s].
    t_exp : float
        Exposure time [s].
    n_pix : int
        Number of pixels over which the noise is summed.
    BV : float
        Bias voltage [V] for the read noise calculation.
    glow : float
        Glow level [e-/px/frame] for the read noise calculation.
    nb_frames : int
        Number of frames in the integration time
    cds_mode : bool, optional
        If True, the read noise is calculated for correlated double sampling (CDS) mode, which doubles the read noise variance.

    Returns
    -------
    float
        Total noise standard deviation [e-] for the fringe measurement.
    """
    phot_noise = photon_noise(n_ph, visi, n_tel)
    dk_noise = dark_current(dk_rate, t_exp, n_pix)
    read_noise = ron(BV, n_pix, glow, cds_mode=cds_mode)

    return (phot_noise + dk_noise + read_noise)**0.5 / nb_frames**0.5

def coherent_energy(n_phot, visi):
    """Calculate the coherent energy of a fringe measurement.

    Parameters
    ----------
    n_phot : float
        Number of photo-events of a SINGLE beam.
    visi : float
        Fringe visibility (contrast), between 0 and 1.

    Returns
    -------
    float
        Coherent energy of the fringe measurement, computed as ``(n_phot * visi)**2``.

    Notes
    -----
    This formula is derived from the fact that the coherent energy scales as 
    ``E_coh ~ (I1 * I2) * V**2``. 
    For a balanced beam combination where both beams have the same intensity (i.e. ``I1 = I2 = n_phot``), 
    this simplifies to ``E_coh ~ n_phot**2 * V**2``.
    """
    return (n_phot * visi)**2

def calculate_snr(nrj, noise):
    """
    Calculate the signal-to-noise ratio (SNR) of a fringe measurement.

    Parameters
    ------------
    nrj : float
        Signal power.
    noise : float
        Noise standard deviation.

    Returns
    -------
    float
        Signal-to-noise ratio (SNR).
    """
    return nrj / noise
