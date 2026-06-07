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
- K: 4.66e9 ph/s/m2/um, 666.7 Jy, wl=2.159 um, https://ui.adsabs.harvard.edu/abs/2003AJ....126.1090C/abstract
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from snr_analysis_lib import *

plt.ion()

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
        fig_width = 12
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
    ax.set_ylim(1e-8, 5e5)
    ax.axhline(3, color='k', linestyle='--', linewidth=1.2, label='Threshold SNR=3')
    ax.grid(True, alpha=0.3)

    frozen_title = ", ".join(
        f"{axis_names[axis_id]}={_format_axis_value(axis_id, value, idx=frozen_idx[axis_id])}"
        for axis_id, value in frozen_selected_values.items()
    )
    if frozen_title:
        ax.set_title(f"{quantity_label} vs {axis_names[x_axis]} \nwith {frozen_title}\nand R={R}, DIT={t_dit} s, exp time={t_exp/60:.1f} min", fontsize=16)
    else:
        ax.set_title(f"{quantity_label} vs {axis_names[x_axis]}", fontsize=16)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(title="1D slice", fontsize=13, title_fontsize=14)

    fig.tight_layout()
    return fig, ax

#----------------------------------
# SNR
#----------------------------------
use_ut = True
cds_mode = False

# Observatory and instrument parameters
if use_ut:
    t_vlti = [np.mean([0.11, 0.13, 0.12, 0.14]), 
              np.mean([0.17, 0.23, 0.21, 0.10]), 
              np.mean([0.26, 0.29, 0.25, 0.15])] # VLTI UT transmission in J, H and K bands, https://arxiv.org/pdf/1608.06752.pdf, Tab. 1
    m1_diam = 8.2 # Diameter of the primary mirror in m
    S_tel = np.pi * (m1_diam/2)**2 # Collecting area of a VLTI UT in m²
    tel_type = 'UT'
    m2_diam = 1.116 # Diameter of the secondary mirror in m
else:
    t_vlti = [np.mean([0.17, 0.18, 0.14, 0.18]), 
              np.mean([0.33, 0.37, 0.28, 0.33]), 
              np.mean([0.36, 0.39, 0.32, 0.33])] # VLTI AT transmission in J, H and K bands, https://arxiv.org/pdf/1608.06752.pdf, Tab. 1
    S_tel = np.pi * (1.8/2)**2 # Collecting area of a VLTI AT in m²
    tel_type = 'AT'

N_tel = 4 # Number of telescopes contributing to the beam combination
rho_0 = 0.8 # Max coupling efficiency, Coupling efficiency of the single-mode fiber
rho_1 = 0.95 # Relative coupling efficiency for a M2 diameter being 13% of M1 diameter, like for the UT (Ruilier 1998)
rho0 = rho_0 * rho_1 # Coupling efficiency for the UT, taking into account central obscuration.
t_echelle = 0.25 # Throughput of the échelle spectrograph
t_sharps = 0.20 # Average throughput of GRAVITY in HR mode, as a proxy for SHARPS
t_corrector = 0.26
"""
Corrective througphput so limiting magnitude match GRAVITY's K=22 on UTs 
for 60 min integration time with 100s exposure time per frame 
(source: Very Large Telescope Paranal Science Operations GRAVITY User Manual + 
advertised limiting magnitude by GPAO + DIT of 100s for "deep integration" and exp time of 60 min (assumed)) 
"""
n_pix = 16 # Number of pixels to sample a fringe in a spectral chanel: spot of 2x2 pixels and 4 outputs

# Detector parameters
QE = 0.8 # Quantum efficiency of the detector
dk_rate = 1e-4 # Dark current rate in e-/px/s
t_exp= 30. # Total exposure time in minutes
t_exp *= 60. # Total exposure time in seconds

for t_dit in [1. ,10., 100.]:
    nb_frames = t_exp // t_dit # Number of frames in the integration time
    glow = 0.1 # Glow in e-/px/frame, to be reduced in next detector prototype

    # Photometric parameters
    # Reference photon flux for a zero-magnitude star in photons/m²/um/s
    wl_labels = ['Y', 'I', 'J', 'H', 'K']
    phi = [2.96215222e+10, 4.87143666e+10, 1.94534122e+10, 9.47947307e+09, 4.66e9] # Reference photon flux for a zero-magnitude star in photons/m²/um/s for Y, I, J, H and K bands, respectively

    wls = [0.79, 1.0305, 1.235, 1.662, 2.159] # in um
    wls = np.array(wls)
    Sr = [0.5, 0.5, 0.5, 0.5, 0.8] # Strehl ratio set at 0.5 for Y, I, J and H bands, and 0.8 for K band, as a proxy for the expected performance of the AO system in each band (per GPAO expectations)


    # R = 25000 # Spectral resolution of the spectrograph

    for R in [22, 500, 4500, 25000]:

        dls = wls / R # Spectral bin width in um
        mag_range = np.linspace(0, 23, 24) # Range of target magnitudes to explore
        visi_range = np.array([0.1, 0.5, 1.0]) # Range of fringe visibilities to explore
        voltage_range = np.arange(3, 13) # Range of bias voltages to explore

        snr_results = np.zeros((len(dls), len(visi_range), len(mag_range), len(voltage_range))) # wl, V, mag, BV

        for m, dl in enumerate(dls):
            for i, visi in enumerate(visi_range):
                for j, mag in enumerate(mag_range):
                    for k, BV in enumerate(voltage_range):
                        t_vlti_interp = interp1d(wls[2:], t_vlti, kind='linear', fill_value='extrapolate')
                        throughput = t_vlti_interp(wls[m]) * rho_0 * t_echelle * t_sharps * Sr[m] * t_corrector # Total throughput
                        # throughput = t_vlti[-1] * rho_0 * t_echelle * t_sharps * Sr[m] * t_corrector # Total throughput

                        n_ph = photon_number(t_dit, mag=mag, throughput=throughput, S_tel=S_tel, N_tel=N_tel, dl=dl, QE=QE, phi_0=phi[m])

                        nrj = coherent_energy(n_ph, visi)
                        noises = total_noise(n_ph, visi, 2, dk_rate, t_dit, n_pix, BV, glow, nb_frames)
                        snr = calculate_snr(nrj, noises)
                        snr_results[m, i, j, k] = snr

        #------------------------------
        # Plotting SNR
        #------------------------------

        for v in visi_range:
            froz_values = {'visibility': v, 'Bias voltage (Volt)': 12.0}
            fig, ax = plot_all_1d_cuts(
                data_cube=snr_results,
                axis_values=[wl_labels, visi_range, mag_range, voltage_range],
                represented_axes=['Magnitude', 'Band'],  # x-axis then all line cuts
                frozen_values=froz_values,
            axis_names=['Band', 'visibility', 'Magnitude', 'Bias voltage (Volt)'],
            quantity_label=r'SNR ($V^2$)'
            )
            fig.savefig(f'figures/snr_vs_mag_visibility{froz_values["visibility"]}_BV{froz_values["Bias voltage (Volt)"]:04.1f}_R{R}_CDS{cds_mode}_{tel_type}_tdit{t_dit}_texp{t_exp/60:.1f}.png', dpi=150)
            plt.close('all')
