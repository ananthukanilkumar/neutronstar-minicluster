import numpy as np
from scipy.interpolate import interp1d

dE, dM, fej, fub = np.loadtxt('/suphys/aani0116/Work/minicluster_tidal/my_modules/Perturbations_NFW.txt', unpack=True)

_log_interp = interp1d(np.log10(dE), np.log10(np.clip(dM, 1e-30, 1)),
                       bounds_error=False, fill_value=(-30, 0))

def mass_loss_frac(dE_over_Ebind):
    return 10**_log_interp(np.log10(dE_over_Ebind))

_fej_interp = interp1d(np.log10(dE), np.log10(np.clip(fej, 1e-30, 1)),
                       bounds_error=False, fill_value=(-30, 0))

def fej_frac(dE_over_Ebind):
    return 10**_fej_interp(np.log10(dE_over_Ebind))

_fub_interp = interp1d(np.log10(dE), np.log10(np.clip(fub, 1e-30, 1)),
                       bounds_error=False, fill_value=(-30, 0))

def fub_frac(dE_over_Ebind):
    return 10**_fub_interp(np.log10(dE_over_Ebind))
