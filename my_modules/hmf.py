from scipy.integrate import quad
import numpy as np
def dndlog10Mpow(power):
    M=10**power
    return M**(-0.5)*np.exp(-M/1e-2)*np.exp(-1e-14/M) #cutoff set to 1e-2 and 1e-14 M_sun and \gamma =0.5

def prob(power):
    return dndlog10Mpow(power)/quad(dndlog10Mpow, -100,100)[0]

M=np.linspace(-14, -1, 1000)
def get_mass():
    while True:
        x = np.random.uniform(-17, 1)
        u = np.random.uniform(0, 1)
        if u < prob(x):
            return 10**x   