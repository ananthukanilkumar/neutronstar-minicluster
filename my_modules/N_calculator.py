import numpy as np
import matplotlib.pyplot as plt


def rho_bulge(x,y,z, rho0=99.5*1e9, r0=0.075, rcut=2.1):
    r       = np.sqrt(x**2 + y**2)
    r_prime = np.sqrt(r**2 + 4*z**2)
    return rho0 * np.exp(-(r_prime/rcut)**2) / (1 + (r_prime/r0)**1.8)

def rho_thin_disk(x, y, z, Sigma=816.6*1e6, r_s=2.90, z_s=0.3):
    R = np.sqrt(x**2 + y**2)
    return (Sigma / (2*z_s)) * np.exp(-R/r_s - np.abs(z)/z_s)

def rho_thick_disk(x, y, z, Sigma=209.5*1e6, r_s=3.31, z_s=0.9):
    R = np.sqrt(x**2 + y**2)
    return (Sigma / (2*z_s)) * np.exp(-R/r_s - np.abs(z)/z_s)  #Check if 2 is required

def stellar_rho_total(x, y, z):
    return rho_bulge(x, y, z) + rho_thin_disk(x, y, z) + rho_thick_disk(x, y, z)

def N_encounter_new(t,xv, b_max=1e-3): #stellar_rho_total use the better model
    return np.trapezoid(stellar_rho_total(xv[:,0], xv[:,1], xv[:,2]) * np.linalg.norm(xv[:, 3:],axis=1) * np.pi  * b_max**2,t)
