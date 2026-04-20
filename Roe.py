from flux_helper import cons_flux
from eigensystem import *

# Roe Riemann Solver
def Roe(UL: np.ndarray, UR: np.ndarray, params) -> np.ndarray:
    # UL, UR : (3,) arrays of conservative variables [rho, rho*u, E]

    # physical fluxes at UL and UR
    FL = np.array(cons_flux(UL, params))
    FR = np.array(cons_flux(UR, params))

    # Roe-average state for eigenvalues/vectors: not implemented

    # arithmetic avg
    U_avg = 0.5 * (UL + UR)

    lam, L, R = get_eigensystem_conservative(U_avg, params)

    # jump in conservative variables
    dU = UR - UL                            # shape (3,)

    roe_sum = np.zeros(3)

    for i in range(3):
        li = L[i, :]
        ri = R[:, i]
        partial = (li @ dU) * abs(lam[i]) * ri
        roe_sum += partial

    # final Roe flux
    return (0.5 * (FL + FR)) - (0.5 * roe_sum)
