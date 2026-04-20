# Euler1D: Finite Volume Methods for 1D Euler Equations

This repository implements finite volume shock-capturing methods for the 1D compressible Euler equations. The goal is to study how different numerical choices affect the resolution of shocks, discontinuities, and smooth flow features.

The system solved is the standard conservation law:
U_t + F(U)_x = 0, where U = (ρ, ρu, ρE) with an ideal gas equation of state (γ = 1.4).

The goal of this project is to understand trade-offs between accuracy, stability, and numerical diffusion, and to reproduce benchmark-quality solutions consistent with standard CFD references. 

## Methods

The code includes first-, second-, and third-order spatial discretizations using:
- First-order Godunov (FOG)
- Piecewise Linear Method (PLM)
- Piecewise Parabolic Method (PPM)

Both HLL and Roe Riemann solvers are implemented, along with several slope limiters (minmod, van Leer, and MC). Boundary conditions include outflow, reflecting, and problem-specific variants.

## Test Problems

Four standard benchmark problems are used for verification and comparison:

- **Sod shock tube**: a classic discontinuous problem that generates a shock, contact discontinuity, and rarefaction wave.
- **Rarefaction wave**: a smooth problem used to assess convergence behavior.
- **Interacting blast waves (Blast2)**: a strong shock problem with reflecting boundaries and no closed-form solution.
- **Shu–Osher problem**: a shock interacting with oscillatory density perturbations, testing resolution of small-scale structure.

## Comparison Studies

The project compares methods across several dimensions:

- Reconstruction methods are compared on the Sod problem using FOG, PLM, and PPM.  
- Riemann solvers (HLL vs Roe) are compared on the rarefaction problem.  
- Slope limiters are compared on the Blast2 problem using PPM with Roe.  
- Grid resolution effects are studied on the Shu–Osher problem (Nx = 32, 64, 128).  
- CFL sensitivity is also tested on Shu–Osher across a range of time step sizes.

