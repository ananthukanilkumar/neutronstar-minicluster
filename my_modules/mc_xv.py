
import numpy as np
from scipy.stats import uniform_direction
import agama
agama.setUnits(length=1, velocity=1, mass=1)
pot_mw = agama.Potential('/suphys/aani0116/minicluster/minicluster_tidal/Agama/data/McMillan17.ini')

def generate_xv_at(position , N ):

    coords = np.tile(position, (N, 1))   # shape (N, 3), every row = position
    R=np.linalg.norm(coords,axis=1)  # shape (N,)

    v_c=np.sqrt(agama.G*pot_mw.enclosedMass(R)/R)  # circular speed
    sigma=v_c/np.sqrt(2)  # 1D dispersion, MB speed distribution peaks at v_c
    vel=np.random.normal(0, 1, (N, 3)) * sigma[:, None]  # each component ~ N(0, sigma)
    return np.column_stack([coords, vel])

