import numpy as np
import agama
agama.setUnits(length=1, velocity=1, mass=1)  # kpc, km/s, Solar mass

def dynamical_time(mc_mass,mc_rvir,density=None):
    rho= ((3*mc_mass)/(4*np.pi*mc_rvir**3)) 
    if density is not None:
        print('Density of the bound object is:', rho)
    return np.sqrt((3*np.pi)/(16*agama.G*rho))