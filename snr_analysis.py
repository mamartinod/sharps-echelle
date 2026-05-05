"""
SNR analysis for an échelle spectrograph to be installed in the VLTI/SHARPS instrument.
The camera is a 1kx1k LmAPD.

Main features of the camera:
- gain is propotional to bias voltage (BV)
- Read noise is function of bias voltage (BV), is expressed as after amplification
- amplification of signal is deterministic unlike EMCCD, so no excess noise factor
- Dark current: 0.1 e-/px/ksec
- Quantum efficiency: 0.8 if BV <= 12
- glow: 0.1 e-/px/s if BV <= 12. To be reduced in next detector prototype. It comes from reading electrons from the material itself
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.ion()

def photon_number(t_exp, mag, throughput, S_tel, N_tel, dl_j, QE, phi_0):
    n_ph = throughput * S_tel * N_tel * dl_j * QE * t_exp * phi_0 * 10**(-0.4 * mag)
    return n_ph

def photon_noise(n_phot, V, n_teltot):
    """
    Return the variance of the photon noise.


    Parameters
    ----------
    n_phot : float
        Total number of detected photons.
    V : float
        Fringe visibility (contrast), between 0 and 1.
    n_teltot : int
        Total number of telescopes contributing to the beam combination.

    Returns
    -------
    float
        Variance of the photon noise [e-²], including both the Poisson
        shot noise term and the speckle noise contribution.

    Notes
    -----
    The variance is computed as:

    .. math::

        \\sigma^2 = 2 N_{\\mathrm{phot}} \\left(\\frac{N_{\\mathrm{phot}} V}{N_{\\mathrm{tel}}}\\right)^2 + N_{\\mathrm{phot}}

    where the first term represents speckle (intensity) noise scaled by the
    squared fringe contrast per baseline, and the second term is the standard
    Poisson contribution.
    """
    return 2 * n_phot * (n_phot * V / n_teltot)**2 + n_phot

def dark_current(dk_rate, t_exp, n_pix):
    """
    Return the total dark current accumulated over an exposure.

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
        Total dark current [e-] accumulated across all pixels during the
        exposure, computed as ``dk_rate * t_exp * n_pix``.
    """
    return dk_rate * t_exp * n_pix

def ron(BV, npix, glow):
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

    Returns
    -------
    float or ndarray
        Total read noise [e-/frame] accumulated across all pixels at the
        requested bias voltage(s).

    """
    BV0 = np.arange(3, 15) # Bias voltage range to explore in Volt
    ron_values = np.array([13.3, 11.3, 9, 7, 5.5, 4, 3, 2, 1.5, 1, 0.8, 0.4])

    rn = np.interp(BV, BV0, ron_values)

    return (rn + glow) * npix

def total_noise(n_ph, visi, n_teltot, dk_rate, t_exp, n_pix, BV, glow):
    phot_noise = photon_noise(n_ph, visi, n_teltot)
    dk_noise = dark_current(dk_rate, t_exp, n_pix)
    read_noise = ron(BV, n_pix, glow)

    return (phot_noise + dk_noise + read_noise)**0.5

def coherent_energy(n_phot, visi, n_teltot):
    return (n_phot * visi / n_teltot)**2

def calculate_snr(nrj, noise):
    return nrj / noise

#-----------------------------------
# RON vs BV profile characterisation
#-----------------------------------
# BV = np.arange(3, 15) # Bias voltage range to explore in Volt
# ron_values = np.array([13.3, 11.3, 9, 7, 5.5, 4, 3, 2, 1.5, 1, 0.8, 0.4])
# ratios = ron_values[1:] / ron_values[:-1]

# decexpo = lambda x, A, lam: A * np.exp(-x / lam)
# decexpo2 = lambda x, A: A * np.exp(-x / ( -1/np.log(ratios.mean()) ))

# # Fit the decay exponent
# popt, pcov = curve_fit(decexpo, BV[-6:], ron_values[-6:])
# popt2, pcov2 = curve_fit(decexpo, BV, ron_values)

# res = ron_values - decexpo(BV, *popt)
# res2 = ron_values - decexpo(BV, *popt2)

# chi = np.sum(res**2)
# chi2 = np.sum(res2**2)

# print(chi, chi2)

# plt.figure()
# plt.plot(BV, ron_values, marker='o')
# plt.plot(BV, decexpo(BV, *popt), label='Exponential Fit', linestyle='--')
# plt.plot(BV, decexpo(BV, *popt2), label='Exponential Fit', linestyle=':')
# plt.xlabel('Bias Voltage [V]')
# plt.ylabel('Read Noise [e-]')
# plt.title('Read Noise vs Bias Voltage')
# plt.grid(True)

#----------------------------------
# SNR
#----------------------------------
use_ut = True

# Observatory and instrument parameters
if use_ut:
    t_vlti = 0.32 # np.mean([0.17, 0.18, 0.14, 0.18]) # VLTI UT transmission in J band, https://arxiv.org/pdf/1608.06752.pdf, Tab. 1
    m1_diam = 8.2 # Diameter of the primary mirror in m
    S_tel = np.pi * (m1_diam/2)**2 # Collecting area of a VLTI UT in m²
    tel_type = 'UT'
    m2_diam = 1.116 # Diameter of the secondary mirror in m
else:
    t_vlti = 0.22 # np.mean([0.11, 0.13, 0.12]) # VLTI AT transmission in J band, https://arxiv.org/pdf/1608.06752.pdf, Tab. 1
    S_tel = np.pi * (1.8/2)**2 # Collecting area of a VLTI AT in m²
    tel_type = 'AT'

N_tel = 4 # Number of telescopes contributing to the beam combination
rho_0 = 0.8 # Max coupling efficiency, Coupling efficiency of the single-mode fiber
rho_1 = 0.95 # Relative coupling efficiency for a M2 diameter being 13% of M1 diameter, like for the UT (Ruilier 1998)
rho0 = rho_0 * rho_1 # Coupling efficiency for the UT, taking into account central obscuration.
Sr = 0.5 # Strehl ratio
t_echelle = 0.25 # Throughput of the échelle spectrograph
t_sharps = 0.34/4 # Average throughput of GRAVITY in HR mode, as a proxy for SHARPS
throughput = t_vlti * rho_0 * t_echelle * t_sharps * Sr # Total throughput
n_pix = 9 # Number of pixels per spectral bin (assuming a 3x3 sampling of the PSF)

# Detector parameters
QE = 0.8 # Quantum efficiency of the detector
dk_rate = 1e-4 # Dark current rate in e-/px/s
t_exp = 0.005 # Exposure time in seconds
glow = 0.1 # Glow in e-/px/frame, to be reduced in next detector prototype

# Photometric parameters
phi_0 = 6.2e10 # Reference photon flux for a zero-magnitude star in J band, in photons/m²/um

wl_j = 1.25 # Central wavelength of J band in um
wl_i = 1.0 # Central wavelength of I band in um
R = 25000 # Spectral resolution of the spectrograph
dl_i = wl_i / R # Spectral bin width in um
dl_j = wl_j / R # Spectral bin width in um

mag = 0.
visi = 1. # Fringe visibility
BV = 3 # Bias voltage in Volt

n_ph = photon_number(t_exp, mag=mag, throughput=throughput, S_tel=S_tel, N_tel=2, dl_j=dl_j, QE=QE, phi_0=phi_0)

nrj = coherent_energy(n_ph, visi, 2)
noises = total_noise(n_ph, visi, 2, dk_rate, t_exp, n_pix, BV, glow)

snr = calculate_snr(nrj, noises)

