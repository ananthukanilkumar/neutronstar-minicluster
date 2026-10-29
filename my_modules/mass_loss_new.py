import numpy as np
from scipy.interpolate import interp1d


dE, dM, fej, fub = np.loadtxt('/suphys/aani0116/Work/minicluster_tidal/my_modules/Perturbations_NFW.txt', unpack=True) # Columns: deltaE/Ebind, deltaM/M, f_ej, E_i_unbound/E_i_total

dm_interp=interp1d(np.log10(dE[dM!=0]),np.log10(dM[dM!=0]),bounds_error=False,fill_value=(-40, 0))
get_dm_frac=lambda dE_dummy:10**dm_interp(np.log10(dE_dummy))

fej_interp=interp1d(np.log10(dE),fej,bounds_error=False, fill_value=(0, 1),kind='linear')
get_fej=lambda dE_dummy:fej_interp(np.log10(dE_dummy))

mask= fub>0 # to remove a kink in the data around 1e-3
mask[50:52]=0

fub_interp=interp1d(np.log10(dE[mask]),fub[mask],bounds_error=False, fill_value=(0, 1),kind='linear')
get_fub=lambda dE_dummy:fub_interp(np.log10(dE_dummy))
