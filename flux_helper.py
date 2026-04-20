# helper to compute conservative flux

from conversions import *

def cons_flux(U: np.ndarray, params) -> np.ndarray:
    """
    Compute physical flux F(U) = [rho*u, rho*(u^2) + p, u(E+p)].
    U : (3,) = [rho, rho*u, E]
    returns F : (3,)
    """
    rho_floor = params.rho_floor
    p_floor = params.p_floor

    rho, mom, E = U
    rho, u, p = cons_to_prim(U, params)   # returns [rho, u, p]

    # positivity safeguard
    rho = max(rho, rho_floor)
    p   = max(p, p_floor)

    return np.array([mom,
                     rho * u*u + p,
                     u * (E + p)])
