# boundary condition definitions

from conversions import *

def update_BCs(U, params):
    ib, ie, ngc, bc_type = params.ib, params.ie, params.ngc, params.bc_type

    if bc_type == "outflow":  # outflow BC update
        # outflow: copy boundary cell into all ghosts
        # left ghosts
        for i in range(ngc):
            U[:, i] = U[:, ib]
        # right ghosts
        N_tot = U.shape[1]
        for i in range(ngc):
            U[:, N_tot-1-i] = U[:, ie]

    elif bc_type == "reflecting":
        # reflecting: mirror rho and p, flip u
        V = cons_to_prim(U, params)
        # left ghosts: mirror about the first interior (ib)
        #  j = 2*ib - 1 - i
        for i in range(ngc):
            j = 2 * ib - 1 - i
            V[0, i] = V[0, j]  # rho
            V[1, i] = -V[1, j]  # rho*u
            V[2, i] = V[2, j]  # rho*E

        # right ghosts: mirror about the last interior (ie)
        #  ghost index gi = N_tot-ngc + i
        #  mirror index j = 2*ie + 1 - gi
        N_tot = V.shape[1]
        for i in range(ngc):
            gi = N_tot - ngc + i
            j = 2 * ie + 1 - gi
            V[0, gi] = V[0, j]
            V[1, gi] = -V[1, j]
            V[2, gi] = V[2, j]

        U = prim_to_cons(V, params)

    elif bc_type == "Shu-Osher":
        # keep ghosts at their initial primitive values
        # must have params.V0 = initial V for entire domain
        V = cons_to_prim(U, params)

        # left ghosts
        V[:, :ngc] = params.V0[:, :ngc]
        # right ghosts
        V[:, -ngc:] = params.V0[:, -ngc:]

        U = prim_to_cons(V, params)

    return U
