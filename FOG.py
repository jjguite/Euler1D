# First-Order Godunov method definition

from conversions import *
from riemann import riemann
import boundary_conditions as bc

# FOG: one complete update step
def FOG(U, dt, params):
    # 1) BCs
    U = bc.update_BCs(U, params)

    # 2) build UL/UR at interfaces
    ib, ie = params.ib, params.ie
    UL = U[:, (ib - 1):(ie + 1)]    # left state at interfaces (0 ... nFlux-1)
    UR = U[:, ib:(ie + 2)]          # right state

    # 3) Riemann solver
    nFlux = params.nFlux
    F = np.zeros((3, nFlux))
    for j in range(nFlux):
        F[:, j] = riemann(UL[:, j], UR[:, j],
                          params.solver, params)

    # 4) update interior
    U[:, ib:ie+1] -= (dt/params.dx) * (F[:, 1:] - F[:, :-1])

    return U
