
import numpy as np
from scipy.stats import uniform_direction
import agama
pot_mw = agama.Potential('/suphys/aani0116/minicluster/minicluster_tidal/Agama/data/McMillan17.ini')

def generate_xv_at(position , N ):

    coords = np.tile(position, (N, 1))   # shape (N, 3), every row = position
    R=np.linalg.norm(coords,axis=1)  # shape (N,)
    vel_unit = uniform_direction(3).rvs(N)  # shape (N, 3)

    v_ideal=np.sqrt(agama.G*pot_mw.enclosedMass(R)/R)
    vel=v_ideal[:, None] * vel_unit
    return np.column_stack([coords, vel])

def generate_xv(N,Rmax=100):

    R=np.random.uniform(0,Rmax,N)
    theta = np.random.uniform(0, 2 * np.pi,N)
    phi = np.random.uniform(0, 2 * np.pi,N)

    x=R*np.sin(theta)*np.cos(phi)
    y=R*np.sin(theta)*np.sin(phi)
    z=R*np.cos(theta)
    coords = np.column_stack([x, y, z])  # shape (N, 3)

    vel_unit = uniform_direction(3).rvs(N)  # shape (N, 3)

    v_ideal=np.sqrt(agama.G*pot_mw.enclosedMass(R)/R)
    vel=v_ideal[:, None] * vel_unit
    return np.column_stack([coords, vel])

