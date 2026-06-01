# SHARPS echelle - SNR analysis

This repository contains a Python script to estimate and visualize the fringe SNR for a SHARPS-like echelle spectrograph concept at VLTI.

Main script:
- `snr_analysis.py`

The model computes SNR as a function of:
- spectral band
- fringe visibility
- target magnitude
- detector bias voltage

## Scientific overview

The script evaluates:
- photo-events from a zero-point flux and instrument throughput
- coherent energy: `(n_ph * V)^2`
- noise terms (photon, dark current, read noise)
- final SNR: `SNR = E_coh / sigma_total`

Default setup includes:
- 4 telescopes
- UT transmission mode (`use_ut = True`)
- detector and throughput assumptions documented in the script header
- Y, I, J, H, K bands

## Requirements

Python 3.9+ is recommended.

Install dependencies:

```bash
pip install numpy matplotlib scipy
```

## Run

From the project root:

```bash
python snr_analysis.py
```

The script:
- computes a 4D SNR cube
- produces 1D cuts with `plot_all_1d_cuts(...)`
- saves output figures as PNG files in the project root

## Outputs

Typical output filenames follow this pattern:

`snr_vs_mag_visibility{visibility}_BV{bias_voltage}_R{R}_CDS{cds_mode}_{tel_type}.png`

Existing examples are already present in the repository.

## Plot helper

`plot_all_1d_cuts(...)` is a generic N-D slicing utility that:
- picks one axis for x
- expands optional additional represented axes into multiple plotted series
- freezes all other axes using nearest grid values
- supports axis names as strings

## Configuration notes

Edit parameters near the middle and end of `snr_analysis.py` to explore different scenarios:
- `use_ut`, `cds_mode`
- `R`, `t_exp`, `glow`, `dk_rate`
- `mag_range`, `visi_range`, `voltage_range`
- `froz_values` in the final plotting blocks

## Limitations

This is currently a script-oriented workflow (not packaged as a Python module), with no automated test suite.

## License

No explicit license file is currently provided in this repository.
