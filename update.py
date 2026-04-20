from FOG import FOG
from PLM import PLM
from PPM import PPM

# main update method
def update(U, dt, params):
    method = params.method

    if method == 'FOG':
        U = FOG(U, dt, params)
    if method == 'PLM':
        U = PLM(U, dt, params)
    elif method == 'PPM':
        U = PPM(U, dt, params)

    return U
