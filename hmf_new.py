import numpy as np
import agama
agama.setUnits(length=1, velocity=1, mass=1) 
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

def dndm(M):
    return (M**-1.8)*(np.exp(-M/1e-6))*(np.exp(-1e-13/M))
def dndlogm(pow):
    M=10**pow
    return (M**-0.8)*(np.exp(-M/1e-6))*(np.exp(-1e-13/M))
logm=np.linspace(-16,-3,1000)
def get_halo_weighted(N):
    
    get_n_vec=np.vectorize(lambda a, b: quad(dndlogm,  b,a)[0])
    upper_m,lower_m=logm[1:],logm[:-1]
    logm_mid=(upper_m+lower_m)/2
    n=get_n_vec(upper_m,lower_m)
    M_total=(10**logm_mid)*n
    prob=M_total/M_total.sum()
    prob_extrapolate = interp1d(logm_mid, prob, kind='linear', fill_value='extrapolate')
    #generate dummy array from which we draw
    m_dummy=np.linspace(-18,-1,10000000)
    p=prob_extrapolate(m_dummy)/prob_extrapolate(m_dummy).sum()
    mc_mass=10**np.random.choice(m_dummy,N,p=p)
    return mc_mass

def get_halo(N):
    get_n_vec=np.vectorize(lambda a, b: quad(dndlogm,  b,a)[0])
    upper_m,lower_m=logm[1:],logm[:-1]
    logm_mid=(upper_m+lower_m)/2
    n=get_n_vec(upper_m,lower_m)
    
    prob=n/n.sum()
    prob_extrapolate = interp1d(logm_mid, prob, kind='linear', fill_value='extrapolate')
    #generate dummy array from which we draw
    m_dummy=np.linspace(-18,-1,100000)
    p=prob_extrapolate(m_dummy)/prob_extrapolate(m_dummy).sum()
    mc_mass=10**np.random.choice(m_dummy,N,p=p)
    return mc_mass