# eigenvalue and eigenvector ("eigensystem") calculations in primitive and conservative vars.

from conversions import *

# eigensystem formulation with PRIMITIVE state variables at one grid cell location
def get_eigensystem(V, params):
    gamma = params.gamma
    rho, u, p = V          # primitive variables

    cs  = np.sqrt((gamma * p) / rho)

    scale1 = 0.5 * (rho / cs)      # scaling factor (lambda1)
    scale3 = 0.5 * rho * cs        # scaling factor (lambda3 - DIFFERENT FROM LAMBDA1) <- this is where the bug was

    Rp = np.array([[-scale1,  1.0,  scale1],
                   [0.5,     0.0,  0.5],
                   [-scale3,  0.0,  scale3]])

    Lp = np.linalg.inv(Rp)

    eigenvalues = np.array([u - cs, u, u + cs])

    return eigenvalues, Lp, Rp


# eigensystem formulation with conservative state variables at one grid cell location
#   --> only used in Roe solver
def get_eigensystem_conservative(U, params):
    gamma = params.gamma

    # convert to prim
    rho, u, p = cons_to_prim(U, params)

    # sound speed
    a = np.sqrt(gamma * p / rho)

    # get eigenvalues
    lam = np.array([u - a, u, u + a])

    # scaling factor
    sc = rho / (2 * a)

    # build the conservative right eigenvectors Rc
    Rc = np.array([
        [-sc, 1.0, sc],
        [-sc * (u - a), u, sc * (u + a)],
        [-sc * ((0.5 * u * u) + (a * a / (gamma - 1)) - (u * a)),
         0.5 * u * u,
         sc * ((0.5 * u * u) + (a * a / (gamma - 1)) + (u * a))]
    ])

    # left eigenvectors
    Lc = np.linalg.inv(Rc)

    return lam, Lc, Rc
