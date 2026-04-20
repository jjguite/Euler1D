from conversions import *
from flux_helper import cons_flux

# HLL Riemann Solver
def HLL(UL, UR, SL, SR, params):
    FL = np.array(cons_flux(UL, params))
    FR = np.array(cons_flux(UR, params))

    F_hll = ((SR * FL) - (SL * FR) + (SL * SR * (UR - UL))) / (SR - SL)

    if 0.0 <= SL:
        return FL
    elif SL < 0.0 < SR:
        return F_hll
    elif 0.0 >= SR:
        return FR

