# conversions between conservative vars <--> primitive vars

import numpy as np

# U = [rho, rho_u, rho_E]
# V = [rho, u, p]

# Note: positivity safeguard is done here in the conversions
#   --> Although it is probably not the most efficient solution,
#       it is the safest and easiest way I found
#       to avoid nan values/unphysical solutions.

def cons_to_prim(U, params):
    GAMMA = params.gamma if params else 1.4

    rho, rho_u, E = U
    u = rho_u / rho
    p = (GAMMA - 1.0) * (E - 0.5 * rho * u*u)

    if params:
        rho_floor = params.rho_floor
        p_floor = params.p_floor

        # positivity safeguard
        if np.isscalar(rho):
            rho = max(rho, rho_floor)
        else:
            for i in range(len(rho)):
                rho[i] = max(rho[i], rho_floor)

        if np.isscalar(p):
            p = max(p, p_floor)
        else:
            for j in range(len(p)):
                p[j] = max(p[j], p_floor)

    # {params == None} happens only during initialization and testing,
    # so no safeguard is needed in this case

    return np.array([rho, u, p])

def prim_to_cons(V, params):
    GAMMA = params.gamma if params else 1.4

    if params:
        rho_floor = params.rho_floor
        p_floor = params.p_floor

        rho, u, p = V

        # positivity safeguard
        if np.isscalar(rho):
            rho = max(rho, rho_floor)
        else:
            for i in range(len(rho)):
                rho[i] = max(rho[i], rho_floor)

        if np.isscalar(p):
            p = max(p, p_floor)
        else:
            for j in range(len(p)):
                p[j] = max(p[j], p_floor)

    else:
        # {params == None} happens only during initialization and testing, so no safeguard is needed
        rho, u, p = V

    # total energy density
    E = (p / (rho * (GAMMA - 1.0))) + (0.5 * u * u)

    return np.array([rho, rho*u, rho*E])
