# initial_conditions.py
import numpy as np


def sod(x):
    """
    Sod shock tube on [0,1]:
      (rho, u ,p) = (1,0,1)       for x < 0.5
                  = (0.125,0,0.1) for x >= 0.5
    """
    V = np.zeros((3, x.size))
    V[0] = np.where(x < 0.5, 1.0, 0.125)  # density
    V[1] = 0.0  # velocity
    V[2] = np.where(x < 0.5, 1.0, 0.1)  # pressure

    tmax = 0.2

    return V, tmax


def rarefaction(x):
    V = np.zeros((3, x.size))
    V[0] = 1.0  # density
    V[1] = np.where(x < 0.5, -2.0, 2.0)  # velocity
    V[2] = 0.4  # pressure

    tmax = 0.15

    return V, tmax


def blast2(x):
    """
    blast2 ([0.0, 1.0]):
      left region   (x < 0.1):   (1.0, 0.0, 1000)
      middle region (x < 0.9):   (1.0, 0.0, 0.01)
      right region  (x > 0.9):   (1.0, 0.0, 100)
    """
    V = np.zeros((3, x.size))
    V[0] = 1.0
    V[1] = 0.0
    V[2] = np.where(x < 0.1, 1000.0,
                    np.where(x < 0.9, 0.01, 100.0))

    tmax = 0.038

    return V, tmax


def shu_osher(x):
    """
    Shu–Osher problem on [-4.5, 4.5]:
      left of shock (x < -4):   (3.857143, 2.629369, 10.33333)
      else: rho = 1 + 0.2sin(5x), u=0, p=1
    """
    V = np.zeros((3, x.size))
    left_mask = x < -4.0
    V[0] = np.where(left_mask, 3.857143, 1.0 + (0.2 * np.sin(5.0 * x)))
    V[1] = np.where(left_mask, 2.629369, 0.0)
    V[2] = np.where(left_mask, 10.33333, 1.0)

    tmax = 1.8

    return V, tmax


# Map string keys to functions
ic_map = {
    "sod": sod,
    "rarefaction": rarefaction,
    "blast2": blast2,
    "shu-osher": shu_osher,
}
