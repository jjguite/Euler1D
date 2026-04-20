from HLL import HLL
from Roe import Roe
from conversions import *

# top level function for Riemann solvers
def riemann(UL, UR, solver, params):

    if solver == 'HLL':
        VL = cons_to_prim(UL, params)
        VR = cons_to_prim(UR, params)

        aL = np.sqrt(params.gamma * VL[2] / VL[0])
        aR = np.sqrt(params.gamma * VR[2] / VR[0])
        uL, uR = VL[1], VR[1]

        SL = min(uL - aL, uR - aR)
        SR = max(uL + aL, uR + aR)

        return HLL(UL, UR, SL, SR, params)
    elif solver == 'Roe':
        return Roe(UL, UR, params)
    else:
        raise ValueError(f"unknown solver {solver}")
