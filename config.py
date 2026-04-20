# Config class for simple parameter setup
from dataclasses import dataclass, field
from initial_conditions import ic_map
from conversions import *  # or wherever you put your EOS

@dataclass
class Config:
    # manually set parameters
    method:     str    # 'FOG', 'PLM', or 'PPM'
    limiter:    str    # 'minmod', 'vanLeer', 'MC'
    solver:     str    # 'HLL' or 'Roe'
    problem:    str    # key in ic_map: 'sod', 'blast2', 'shu-osher', or 'rarefaction'
    N:          int    # number of INTERIOR cells
    Ca:         float  # CFL number
    gamma:      float  = 1.4

    # positivity clamps
    rho_floor:  float  = 1e-12
    p_floor:    float  = 1e-12

    # derived
    xa:         float  = field(init=False)   # left physical boundary
    xb:         float  = field(init=False)   # right physical boundary
    tmax:       float  = field(init=False)   # final time

    dx:         float  = field(init=False)
    ngc:        int    = field(init=False)
    ib:         int    = field(init=False)
    ie:         int    = field(init=False)
    nFlux:      int    = field(init=False)
    bc_type:    str    = field(init=False)

    # initial state arrays
    x:          np.ndarray = field(init=False)  # spatial grid
    V0:         np.ndarray = field(init=False)  # primitive at t=0
    U0:         np.ndarray = field(init=False)  # conservative at t=0

    def __post_init__(self):
        self.xa    = -4.5 if self.problem == 'shu-osher' else 0.0
        self.xb    = 4.5 if self.problem == 'shu-osher' else 1.0

        # 1) spatial resolution
        self.dx    = (self.xb - self.xa) / self.N

        # 2) ghost and total cell counts
        self.ngc   = 2
        self.nx    = self.N + 2*self.ngc

        # 3) interior index beginning (ib) and end (ie)
        self.ib    = self.ngc
        self.ie    = self.ngc + self.N - 1

        # 4) number of flux interfaces = N + 1
        self.nFlux = self.N + 1

        # 5) build cell‐center grid (including ghost cells)
        #    x[i] = xa + (i-ngc+0.5)*dx for i=0...nx-1
        self.x = (self.xa + (np.arange(self.nx) - self.ngc + 0.5) * self.dx)

        # 6) get primitive IC function
        assert self.problem in ic_map, \
            f"Unknown problem '{self.problem}'. Choose one of {list(ic_map)}"

        # 7) use IC mapping to get initial conditions and tmax
        self.V0, self.tmax = ic_map[self.problem](self.x)

        # 8) convert to conservative vars
        self.U0 = prim_to_cons(self.V0, None)

        # 8) select BC type based on test problem
        if self.problem == "shu-osher":
            self.bc_type = "Shu-Osher"
        elif self.problem == "blast2":
            self.bc_type = "reflecting"
        else:
            self.bc_type = "outflow"
