"""
SNR analysis for an échelle spectrograph to be installed in the VLTI/SHARPS instrument.
The camera is a 1kx1k LmAPD.

Main features of the camera:
- gain is propotional to bias voltage (BV)
- Read noise is function of bias voltage (BV), is expressed as after amplification, 
in e-/px/read
- amplification of signal is deterministic unlike EMCCD, so no excess noise factor
- Dark current: 0.1 e-/px/ksec
- Quantum efficiency: 0.8 if BV <= 12
- glow: 0.1 e-/px/read if BV <= 12. To be reduced in next detector prototype. It comes from reading electrons from the material itself

Combiner performs pair-wise combination and 
provide an ABCD sampling (4 outputs) of the fringes.

Consequently, throughput should be divvided by 3 to account for each beam being split to be
combined with the 3 other beams.

Each output is sampled over 4 pixels (2 in spectral direction and 2 in the orthogonal one), 
so a fringe is sampled over 16 pixels in total.

Noise model comes from Colavita (PASP, 1999): https://www.jstor.org/stable/10.1086/316302?seq=4.

The coherent energy scales as:
E_coh = n_ph**2 * V**2, n_ph being the number of photons of a SINGLE beam, and V the fringe visibility.
Demonstration:
E_coh ~ (I1 * I2) * V**2 = (n_ph1 * n_ph2) * V**2 = n_ph**2 * V**2, with n_ph1 = n_ph2 = n_ph for a balanced beam combination.
This expression slgihtly differs from Colavita because of their more accurate estimator for their own ABCD system.

The variance of the noise model includes the ones of:
- Photon noise: sigma_phot^2 = 2 * n_ph_tot * (n_ph * V)**2 + n_ph_tot**2, 
n_ph_tot being the total number of photons illuminating the pixels 
where the fringe pattern is projected (which explains why the SNR 
of an all-in-one combiner decreases as the number of telescopes increases).
- Dark current noise: sigma_dark^2 = dk_rate**2 * t_exp * n_pix
- Read noise: sigma_ron^2 = (ron(BV)**2 + glow**2) * n_pix

Axes of analysis:
- Bias voltage (BV)
- Target magnitude
- Exposure time (t_exp)
- Spectral band (Y, I, J and H)

Photometric zero points:
- Y: 5.71e-9 W/m2/um, 2026 Jy, wl=1.0305 um, https://ui.adsabs.harvard.edu/scan/manifest/2006MNRAS.367..454H
- I: 2550 Jy,wl=0.79e-6, https://www.astrouw.edu.pl/~simkoz/mags.html
- J: 3.129e-13 W/cm2/um, 1594 Jy, wl=1.235 um, https://ui.adsabs.harvard.edu/abs/2003AJ....126.1090C/abstract
- H: 1.133e-13 W/cm2/um, 1024 Jy, wl=1.662 um, https://ui.adsabs.harvard.edu/abs/2003AJ....126.1090C/abstract

"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import curve_fit

plt.ion()

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

def total_noise(n_ph, visi, n_tel, dk_rate, t_exp, n_pix, BV, glow, cds_mode=False):
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

    return (phot_noise + dk_noise + read_noise)**0.5

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
cds_mode = False

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
n_pix = 16 # Number of pixels to sample a fringe in a spectral chanel: spot of 2x2 pixels and 4 outputs

# Detector parameters
QE = 0.8 # Quantum efficiency of the detector
dk_rate = 1e-4 # Dark current rate in e-/px/s
t_exp = 1. # Exposure time in seconds
glow = 0.1 # Glow in e-/px/frame, to be reduced in next detector prototype

# Photometric parameters
# Reference photon flux for a zero-magnitude star in photons/m²/um/s
wl_labels = ['Y', 'I', 'J', 'H']
phi = [2.96215222e+10, 4.87143666e+10, 1.94534122e+10, 9.47947307e+09] # Reference photon flux for a zero-magnitude star in photons/m²/um/s for Y, I, J and H bands, respectively

wls = [0.79, 1.0305, 1.235, 1.662] # in um
wls = np.array(wls)

R = 25000 # Spectral resolution of the spectrograph

dls = wls / R # Spectral bin width in um
mag_range = np.linspace(0, 20, 11) # Range of target magnitudes to explore
visi_range = np.linspace(0., 1.0, 5) # Range of fringe visibilities to explore
voltage_range = np.arange(3, 13) # Range of bias voltages to explore

snr_results = np.zeros((len(dls), len(visi_range), len(mag_range), len(voltage_range))) # wl, V, mag, BV

for m, dl in enumerate(dls):
    for i, visi in enumerate(visi_range):
        for j, mag in enumerate(mag_range):
            for k, BV in enumerate(voltage_range):
                n_ph = photon_number(t_exp, mag=mag, throughput=throughput, S_tel=S_tel, N_tel=N_tel, dl=dl, QE=QE, phi_0=phi[m])

                nrj = coherent_energy(n_ph, visi)
                noises = total_noise(n_ph, visi, 2, dk_rate, t_exp, n_pix, BV, glow)
                snr = calculate_snr(nrj, noises)
                snr_results[m, i, j, k] = snr

#------------------------------
# Plotting SNR
#------------------------------

def plot_all_1d_cuts(data_cube, axis_values, represented_axes, frozen_values=None,
                     axis_names=None, quantity_label='SNR', ax=None):
    """Plot all possible 1D cuts for selected axes of an N-D data cube.

    Parameters
    ----------
    data_cube : ndarray
        N-D array containing the data to plot.
    axis_values : list of array_like
        Coordinate values for each axis of ``data_cube``.
        ``axis_values[i]`` contains the physical values along axis ``i``.
    represented_axes : list or tuple
        Axes to represent on the plot. The first axis is used as x-axis.
        Any additional axes are fully expanded as one line per value
        combination (all 1D cuts are shown).
    frozen_values : dict, optional
        Frozen values for non-represented axes.
        Keys are axis indices (int) or axis names (if ``axis_names`` is given).
        Values are physical axis values; nearest grid point is used.
    axis_names : list of str, optional
        Labels for each axis of ``data_cube``.
    quantity_label : str, optional
        Label of plotted quantity on the y-axis.
    ax : matplotlib.axes.Axes, optional
        Existing axes where the plot is drawn.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes containing the plot.

    Notes
    -----
    For every axis not in ``represented_axes``, a value must be provided in
    ``frozen_values`` so the corresponding slice can be extracted.
    """
    data_cube = np.asarray(data_cube)
    ndim = data_cube.ndim

    if len(axis_values) != ndim:
        raise ValueError("axis_values must contain one coordinate array per data cube axis.")

    if axis_names is None:
        axis_names = [f"axis_{i}" for i in range(ndim)]
    if len(axis_names) != ndim:
        raise ValueError("axis_names must have the same length as data_cube.ndim.")

    name_to_index = {name: i for i, name in enumerate(axis_names)}

    def _to_axis_index(axis_id):
        if isinstance(axis_id, str):
            if axis_id not in name_to_index:
                raise ValueError(f"Unknown axis name: {axis_id}")
            return name_to_index[axis_id]
        return int(axis_id)

    rep_axes = [_to_axis_index(axis_id) for axis_id in represented_axes]
    if len(rep_axes) == 0:
        raise ValueError("represented_axes must contain at least one axis.")
    if len(set(rep_axes)) != len(rep_axes):
        raise ValueError("represented_axes contains duplicated axes.")
    if any(ax_id < 0 or ax_id >= ndim for ax_id in rep_axes):
        raise ValueError("represented_axes contains an out-of-range axis index.")

    frozen_values = {} if frozen_values is None else dict(frozen_values)

    # Normalize frozen-values keys to axis indices.
    frozen_by_axis = {}
    for key, value in frozen_values.items():
        axis_id = _to_axis_index(key)
        frozen_by_axis[axis_id] = value

    non_rep_axes = [i for i in range(ndim) if i not in rep_axes]
    missing_frozen_axes = [i for i in non_rep_axes if i not in frozen_by_axis]
    if missing_frozen_axes:
        missing_names = ", ".join(axis_names[i] for i in missing_frozen_axes)
        raise ValueError(f"Missing frozen value(s) for axis/axes: {missing_names}")

    def _format_axis_value(axis_id, value, idx=None):
        """Return a readable value label, with special handling for spectral band labels."""
        axis_name = axis_names[axis_id].strip().lower()

        if axis_name in ("wl band", "wl_band", "band"):
            wl_global = globals().get('wl_labels', None)
            if idx is not None and wl_global is not None and len(wl_global) == data_cube.shape[axis_id]:
                return str(wl_global[idx])

        if isinstance(value, (str, np.str_)):
            return str(value)

        try:
            return f"{float(value):g}"
        except (TypeError, ValueError):
            return str(value)

    def _find_axis_index(arr, value):
        """Return axis index from value: nearest for numeric axes, exact match for text axes."""
        arr = np.asarray(arr)

        if np.issubdtype(arr.dtype, np.number):
            return int(np.argmin(np.abs(arr - value)))

        exact = np.where(arr == value)[0]
        if exact.size:
            return int(exact[0])

        # Tolerate case-only mismatch for string-like axes.
        arr_str = np.char.lower(arr.astype(str))
        value_str = str(value).lower()
        exact_ci = np.where(arr_str == value_str)[0]
        if exact_ci.size:
            return int(exact_ci[0])

        raise ValueError(f"Value {value} not found in non-numeric axis values: {arr.tolist()}")

    # Select nearest indices for frozen axes.
    frozen_idx = {}
    frozen_selected_values = {}
    for axis_id in non_rep_axes:
        arr = np.asarray(axis_values[axis_id])
        val = frozen_by_axis[axis_id]
        idx = _find_axis_index(arr, val)
        frozen_idx[axis_id] = idx
        frozen_selected_values[axis_id] = arr[idx]

    if ax is None:
        golden_ratio = (1 + 5**0.5) / 2
        fig_width = 10
        fig, ax = plt.subplots(figsize=(fig_width, fig_width / golden_ratio))
    else:
        fig = ax.figure

    x_axis = rep_axes[0]
    x = np.asarray(axis_values[x_axis])
    varying_axes = rep_axes[1:]

    if len(varying_axes) == 0:
        idxer = [slice(None)] * ndim
        for axis_id, idx in frozen_idx.items():
            idxer[axis_id] = idx
        y = data_cube[tuple(idxer)]
        ax.plot(x, y, marker='o', markersize=8, linestyle='-', linewidth=2.5)
    else:
        varying_values = [np.asarray(axis_values[axis_id]) for axis_id in varying_axes]
        for varying_combination in product(*varying_values):
            idxer = [slice(None)] * ndim
            label_parts = []

            for axis_id, idx in frozen_idx.items():
                idxer[axis_id] = idx

            for axis_id, v in zip(varying_axes, varying_combination):
                arr = np.asarray(axis_values[axis_id])
                idx = _find_axis_index(arr, v)
                idxer[axis_id] = idx
                label_value = _format_axis_value(axis_id, arr[idx], idx=idx)
                label_parts.append(f"{axis_names[axis_id]}={label_value}")

            y = data_cube[tuple(idxer)]
            ax.plot(x, y, marker='o', markersize=8, linestyle='-', linewidth=2.5, label=", ".join(label_parts))

        ax.legend(title="Slice 1D", fontsize=13, title_fontsize=14)

    ax.set_xlabel(axis_names[x_axis], fontsize=16)
    ax.set_ylabel(quantity_label, fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_yscale('log')
    ax.set_ylim(1e-9, 1e3)
    ax.axhline(3, color='k', linestyle='--', linewidth=1.2, label='Threshold SNR=3')
    ax.grid(True, alpha=0.3)

    frozen_title = ", ".join(
        f"{axis_names[axis_id]}={_format_axis_value(axis_id, value, idx=frozen_idx[axis_id])}"
        for axis_id, value in frozen_selected_values.items()
    )
    if frozen_title:
        ax.set_title(f"{quantity_label} vs {axis_names[x_axis]} (for {frozen_title})\n R={R}", fontsize=16)
    else:
        ax.set_title(f"{quantity_label} vs {axis_names[x_axis]}", fontsize=16)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(title="1D slice", fontsize=13, title_fontsize=14)

    fig.tight_layout()
    return fig, ax

froz_values = {'visibility': 1, 'Bias voltage (V)': 12.0}
fig, ax = plot_all_1d_cuts(
    data_cube=snr_results,
    axis_values=[wl_labels, visi_range, mag_range, voltage_range],
    represented_axes=['Magnitude', 'Band'],  # x-axis then all line cuts
    frozen_values=froz_values,
    axis_names=['Band', 'visibility', 'Magnitude', 'Bias voltage (V)'],
    quantity_label=r'SNR ($V^2$)'
)
fig.savefig(f'snr_vs_mag_visibility{froz_values["visibility"]}_BV{froz_values["Bias voltage (V)"]:04.1f}_R{R}_CDS{cds_mode}_{tel_type}.png', dpi=150)

froz_values = {'visibility': 0.5, 'Bias voltage (V)': 12.0}
fig, ax = plot_all_1d_cuts(
    data_cube=snr_results,
    axis_values=[wl_labels, visi_range, mag_range, voltage_range],
    represented_axes=['Magnitude', 'Band'],  # x-axis then all line cuts
    frozen_values=froz_values,
    axis_names=['Band', 'visibility', 'Magnitude', 'Bias voltage (V)'],
    quantity_label=r'SNR ($V^2$)'
)
fig.savefig(f'snr_vs_mag_visibility{froz_values["visibility"]}_BV{froz_values["Bias voltage (V)"]:04.1f}_R{R}_CDS{cds_mode}_{tel_type}.png', dpi=150)
