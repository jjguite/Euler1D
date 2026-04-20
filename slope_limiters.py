import time
import warnings
from warnings import catch_warnings

import numpy as np

# calculates slope limiter value (delta w) at a single grid location
# forward, backward, centered = pre-calculated characteristic difference formulas
def calc_slope(forward, backward, centered, limiter):
    # TVD slope limiter options
    if limiter == 'minmod':
        return minmod(forward, backward)
    elif limiter == 'MC':
        return minmod3(2 * forward, centered, 2 * backward)
    elif limiter == 'vanLeer':
        return vanLeer(backward, forward)

def minmod3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Three-argument minmod for MC limiter
    """
    prod = (a*b) * (a*c)
    m = np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c)))
    return np.where(prod > 0, np.sign(a) * m, 0.0)

def minmod(a, b):
    prod = a * b
    return np.where(prod > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

def vanLeer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    prod = a * b

    with catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # ensures all warnings are caught

        result = np.zeros(3)

        for i in range(len(prod)):
            if prod[i] > 0:
                result[i] = (2 * prod[i]) / (a[i] + b[i])

        if w:
            print(f"{w[0].category.__name__}: {w[0].message} (from {w[0].filename}, line {w[0].lineno})")

            time.sleep(1000)
        return result


# limiting procedure (primitive vars)
# calculation for ONE grid cell; V = [rho_i, u_i, p_i]
def calc_limiter(V, L, params):
    # V: 3x3 (3 variables, [i-1:i+1] for each)
    # L, R: 3x3
    dx = params.dx
    lim = params.limiter

    dV_forward = np.zeros(V.shape[0])
    dV_backward = np.zeros(V.shape[0])
    dV_centered = np.zeros(V.shape[0])

    for i in range(V.shape[0]):                         # loop over all variables
        dV_forward[i] = (V[i, 2] - V[i, 1])           # forward difference
        dV_backward[i] = (V[i, 1] - V[i, 0])          # backward difference
        dV_centered[i] = (0.5 * (V[i, 2] - V[i, 0]))   # centered difference


    dW_forward = np.zeros(3)
    dW_backward = np.zeros(3)
    dW_centered = np.zeros(3)

    for i in range(3):
        dW_forward[i]  = L[i] @ dV_forward    # convert to characteristic slope limiting (dot product)
        dW_backward[i] = L[i] @ dV_backward
        dW_centered[i] = L[i] @ dV_centered

    dW_lim   = calc_slope(dW_forward, dW_backward, dW_centered, lim)   # calculate slope using limiter of choice

    return dW_lim

