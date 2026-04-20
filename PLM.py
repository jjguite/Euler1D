from conversions import *
from slope_limiters import calc_limiter
from eigensystem import get_eigensystem
import boundary_conditions as bc
from riemann import riemann

# Piecewise Linear Method: one-step update
def PLM(U, dt, params):
    """
    One complete step of the second‑order piecewise linear method
    (characteristic‐limited, half‑time predictor) with a Riemann solver.

    * Ghost cells are updated first -> no div‑by‑zero in cons->prim.
    * We build UL/UR directly at the N+1 interfaces (ib‑1:ie) and
      never allocate partial slices that can go out of bounds.
    """

    # update boundary conditions (must be valid before cons_to_prim)
    U = bc.update_BCs(U, params)

    ib, ie   = params.ib, params.ie
    dx       = params.dx
    solver   = params.solver
    nFlux    = params.nFlux

    # full primitive var array
    V = cons_to_prim(U, params)              # shape (3, Nx)

    # half‑time predictor in cell range ib‑1 to ie+1  (N+2 cells)
    # store their left/right edge values in VL_cell, VR_cell
    VL_cell = np.zeros_like(V[:, ib-1:ie+2])  # (3, N+2)
    VR_cell = np.zeros_like(V[:, ib-1:ie+2])

    for j in range(ib-1, ie+2):                 # loop over N+2 cells
        lam, Lp, Rp = get_eigensystem(V[:, j], params)

        # characteristic slopes (MC/minmod/vanLeer)
        dW = calc_limiter(V[:, j-1:j+2], Lp, params)   # shape (3,)
        # dW = 0

        # right & left predictor sums
        left_sum = np.zeros(3)
        right_sum = np.zeros(3)

        for i in range(3):
            c1 = 1.0 - lam[i] * dt / dx
            c2 = -1.0 - lam[i] * dt / dx
            right_sum += (c1 * Rp[:, i]) * dW[i]
            left_sum  += (c2 * Rp[:, i]) * dW[i]

        VR_cell[:, j-(ib-1)] = V[:, j] + 0.5 * right_sum
        VL_cell[:, j-(ib-1)] = V[:, j] + 0.5 * left_sum


    # Assemble interface states  (UL, UR)  for [ib-1, ie]
    #     UL = left state of interface, from cell j
    #     UR = right state of interface, from cell j
    UL = prim_to_cons(VR_cell[:, 0:-1], params)       # shape (3, nFlux)
    UR = prim_to_cons(VL_cell[:, 1:], params)         # shape (3, nFlux)

    # Calculate fluxes at interfaces via chosen Riemann solver
    F = np.zeros((3, nFlux))
    for k in range(nFlux):
        F[:, k] = riemann(UL[:, k], UR[:, k], solver, params)

    # Conservative update on interior cells
    U[:, ib:ie+1] -= (dt/dx) * (F[:, 1:] - F[:, :-1])

    return U         # BCs will be filled at the top of the next step
