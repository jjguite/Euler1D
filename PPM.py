from conversions import *
from slope_limiters import *
from eigensystem import get_eigensystem
import boundary_conditions as bc
from riemann import riemann

# Piecewise Parabolic Method: one-step update
def PPM(U, dt, params):
    """
    One explicit step of the third order piecewise parabolic method
    (characteristic‐limited, half‑time predictor) with a Riemann solver.

    Notes
    -----
    * Ghost cells are updated first → no div‑by‑zero in cons→prim.
    * We build UL/UR directly at the N+1 interfaces (ib‑1 ... ie) and
      never allocate partial slices that can go out of bounds.
    """
    # ------------------------------------------------------------
    # 1.  Boundary conditions  (must be valid before cons_to_prim)
    # ------------------------------------------------------------
    U = bc.update_BCs(U, params)

    ib, ie   = params.ib, params.ie
    dx       = params.dx
    solver   = params.solver
    nFlux    = params.nFlux

    # ------------------------------------------------------------
    # 2.  Convert to primitive vars
    # ------------------------------------------------------------
    V = cons_to_prim(U, params)              # shape (3, Nx)

    # ------------------------------------------------------------
    # 3. Half‑time predictor in cells ib ... ie  (N interior cells)
    #    VL_cell / VR_cell also need TWO ghost slots on each side
    # ------------------------------------------------------------
    Nx_int = ie - ib + 1  # = N interior
    VL_cell = np.zeros((3, Nx_int + 2))  # two extra slots: left & right ghost
    VR_cell = np.zeros_like(VL_cell)

    # helper to map cell index -> column in VL/VR
    col = lambda i: i - ib + 1  # ib → 1, …, ie → N

    for i in range(ib, ie + 1):  # only real cells
        lam, Lp, Rp = get_eigensystem(V[:, i], params)

        # initial interface estimates based on 3rd order polynomial reconstruction (unused)
        # vL_i = (1.0 / 12.0) * (-V[:, i - 2] + (7.0 * V[:, i - 1]) + (7.0 * V[:, i]) - V[:, i + 1])
        # vR_i = (1.0 / 12.0) * (-V[:, i - 1] + (7.0 * V[:, i]) + (7.0 * V[:, i + 1]) - V[:, i + 2])

        # --- storage ----------------------------------------------------------
        dW = np.zeros((3, 3))  # characteristic slopes for i-1, i, i+1
        dV = np.zeros((3, 3))  # primitive   slopes   for i-1, i, i+1

        # --- gather slopes for each neighbour --------------------------------
        for j in range(3):  # j = 0 -> i-1, 1 -> i, 2 -> i+1
            cell_idx = i + j - 1

            # three-cell stencil centred on cell_idx
            V_stencil = V[:, cell_idx - 1: cell_idx + 2]

            lam_j, Lp_j, Rp_j = get_eigensystem(V[:, cell_idx], params)

            # characteristic slope for jth cell
            dW[j] = calc_limiter(V_stencil, Lp_j, params)

            # primitive slope for jth cell
            dV[j] = Rp_j @ dW[j]

        # --- new characteristic limited interface values ---------------------
        vL_i = 0.5 * (V[:, i - 1] + V[:, i]) - (1.0 / 6.0) * (dV[1] - dV[0])
        vR_i = 0.5 * (V[:, i] + V[:, i + 1]) - (1.0 / 6.0) * (dV[2] - dV[1])

        vbar = V[:, i]

        # -----------------------------------------------------------
        # Condition 1(a): Detect local extrema and reduce to FOG
        # -----------------------------------------------------------
        # vL_i, vR_i, vbar are 1-D arrays of length 3 (ρ, u, p)
        prod = (vR_i - vbar) * (vbar - vL_i)  # element-wise product
        mask = prod < 0  # True where the inequality holds

        # apply the flattening where needed
        vL_i = np.where(mask, vbar, vL_i)
        vR_i = np.where(mask, vbar, vR_i)

        # -----------------------------------------------------------
        # Condition 1(b): Abscissa shifting
        # -----------------------------------------------------------
        # test (a): shift right edge to 3*vbar - 2*v_L
        cond_a = (-((vR_i - vL_i) ** 2) >
                  6.0 * (vR_i - vL_i) * (vbar - 0.5 * (vR_i + vL_i)))

        vR_new = 3.0 * vbar - 2.0 * vL_i
        vR_i = np.where(cond_a, vR_new, vR_i)

        # test (b): shift left edge in the opposite direction
        cond_b = ((vR_i - vL_i) ** 2 <
                  6.0 * (vR_i - vL_i) * (vbar - 0.5 * (vR_i + vL_i)))

        vL_new = 3.0 * vbar - 2.0 * vR_i
        vL_i = np.where(cond_b, vL_new, vL_i)

        # ---------- C‑coefficients --------------------
        C2 = (6.0 / dx ** 2) * (0.5 * (vR_i + vL_i) - vbar)
        C1 = (1.0 / dx) * (vR_i - vL_i)
        C0 = vbar - (dx * dx / 12.0) * C2

        dc1 = Lp @ (C1 * dx)
        dc2 = Lp @ (C2 * dx * dx)

        # ---------- half‑time right / left sum storage -----------------------
        right_sum_c1 = np.zeros(3)
        right_sum_c2 = np.zeros(3)
        left_sum_c1 = np.zeros(3)
        left_sum_c2 = np.zeros(3)

        for k in range(3):  # explicit loop
            tau = lam[k] * dt / dx
            if lam[k] > 0.0:  # waves contributing to RIGHT state
                c1 = 1.0 - tau
                c2 = 1.0 - (2.0 * tau) + (4.0 / 3.0) * (tau ** 2)
                right_sum_c1 += c1 * Rp[:, k] * dc1[k]
                right_sum_c2 += c2 * Rp[:, k] * dc2[k]

            if lam[k] < 0.0:  # waves contributing to LEFT state
                c1 = -1.0 - tau
                c2 = 1.0 + (2.0 * tau) + (4.0 / 3.0) * (tau ** 2)
                left_sum_c1 += c1 * Rp[:, k] * dc1[k]
                left_sum_c2 += c2 * Rp[:, k] * dc2[k]

        # save into column col(i)
        VR_cell[:, col(i)] = C0 + 0.5 * right_sum_c1 + 0.25 * right_sum_c2
        VL_cell[:, col(i)] = C0 + 0.5 * left_sum_c1 + 0.25 * left_sum_c2

    # -----------------------------------------------------------------
    # 4. fill the ghost columns (col 0 and col −1) with constant states
    # -----------------------------------------------------------------
    VR_cell[:, 0] = V[:, ib - 1]  # right edge of left ghost cell
    VL_cell[:, -1] = V[:, ie + 1]  # left  edge of right ghost cell

    # -----------------------------------------------------------------
    # 5. assemble interface states (UL, UR)  j = ib-1 ... ie
    # -----------------------------------------------------------------
    UL = prim_to_cons(VR_cell[:, :-1], params)  # VR from cell j
    UR = prim_to_cons(VL_cell[:, 1:], params)  # VL from cell j+1

    # ------------------------------------------------------------
    # 6.  Fluxes at interfaces via chosen Riemann solver
    # ------------------------------------------------------------
    F = np.zeros((3, nFlux))
    for k in range(nFlux):
        F[:, k] = riemann(UL[:, k], UR[:, k], solver, params)

    # ------------------------------------------------------------
    # 7.  Conservative update on interior cells
    # ------------------------------------------------------------
    U[:, ib:ie+1] -= (dt/dx) * (F[:, 1:] - F[:, :-1])

    return U         # BCs will be filled at the top of the next step
