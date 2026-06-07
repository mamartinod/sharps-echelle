import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from snr_analysis_lib import *

plt.ion()
#-----------------------------------
# RON vs BV profile characterisation
#-----------------------------------
BV = np.arange(3, 15) # Bias voltage range to explore in Volt
ron_values = np.array([13.3, 11.3, 9, 7, 5.5, 4, 3, 2, 1.5, 1, 0.8, 0.4])
ratios = ron_values[1:] / ron_values[:-1]

decexpo = lambda x, A, lam: A * np.exp(-x / lam)
decexpo2 = lambda x, A: A * np.exp(-x / ( -1/np.log(ratios.mean()) ))

# Fit the decay exponent
popt, pcov = curve_fit(decexpo, BV[-6:], ron_values[-6:])
popt2, pcov2 = curve_fit(decexpo, BV, ron_values)

res = ron_values - decexpo(BV, *popt)
res2 = ron_values - decexpo(BV, *popt2)

chi = np.sum(res**2)
chi2 = np.sum(res2**2)

print(chi, chi2)

plt.figure()
plt.plot(BV, ron_values, marker='o')
plt.plot(BV, decexpo(BV, *popt), label='Exponential Fit', linestyle='--')
plt.plot(BV, decexpo(BV, *popt2), label='Exponential Fit', linestyle=':')
plt.xlabel('Bias Voltage [V]')
plt.ylabel('Read Noise [e-]')
plt.title('Read Noise vs Bias Voltage')
plt.grid(True)