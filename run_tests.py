import os
import pickle
import numpy as np
from config import Config
from main import main_without_plots

# testing functions. saves test data to .pkl files.

def save_simulation_data(
        filename: str,
        U_final: np.ndarray,
        N: int,
        CFL: float,
        method: str,
        limiter: str,
        solver: str,
        problem: str
) -> None:
    """
    Append one simulation record (metadata + arrays) to `filename` using pickle.

    Each record is a Python dict with keys containing parameters and final state U_final.

    If `filename` does not exist yet, it will be created.  We always open in append mode,
    so multiple calls to save_simulation_data(...) with the same `filename` will stack
    up multiple records in that one file.
    """
    record = {
        'N': N,
        'CFL': CFL,
        'method': method,
        'limiter': limiter,
        'U_final': U_final,
        'solver': solver,
        'problem': problem
    }

    # Ensure the directory exists (if the user passed in a path with subfolders)
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    # Open in append‐binary mode and pickle the record
    with open(filename, "ab") as f:
        pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)

    # That’s it. On subsequent calls, new records are just tacked on to the same file.


##########################################################################
# TEST FUNCTIONS

# Each test:
#   - Runs a series of tests based on specified parameters
#   - Creates sequence of pickled records and stores them in a .pkl file
#       --> Unpickle contents to recover parameters and final state U
##########################################################################


def _test1_reconstruction_methods():
    """
    Sod shock tube, N = 128, HLL flux.
    Compare FOG, PLM+minmod, and PPM+minmod.
    Each figure overlays primitive variables (rho, u, p) at t = 0.2.
    """
    filename = "test1.pkl"
    if os.path.exists(filename):
        os.remove(filename)

    # reconstruction suites to test
    suites = [
        ("FOG", ""),
        ("PLM", "minmod"),
        ("PPM", "minmod"),
    ]

    for method, limiter in suites:
        # configuration
        cfg = Config(
            method=method,
            limiter=limiter,
            solver="HLL",
            problem="sod",
            N=128,
            Ca=0.8,
        )

        # run simulation
        U_final = main_without_plots(cfg)

        # save simulation data to file
        save_simulation_data(
            filename=filename,
            U_final=U_final,
            method=method,
            limiter=limiter,
            CFL=cfg.Ca,
            problem=cfg.problem,
            N=cfg.N,
            solver=cfg.solver,
        )


def _test2_riemann_solvers():
    """Smooth rarefaction, N=128, PLM+minmod: HLL vs Roe"""
    filename = "test2.pkl"
    if os.path.exists(filename):
        os.remove(filename)

    # Loop over the two solvers
    for solver in ["HLL", "Roe"]:
        cfg = Config(
            method="PLM",
            limiter="minmod",
            solver=solver,
            problem="rarefaction",
            N=128,
            Ca=0.8,
        )

        # Run the simulation and save data
        U_final = main_without_plots(cfg)
        save_simulation_data(
            filename=filename,
            U_final=U_final,
            method=cfg.method,
            limiter=cfg.limiter,
            CFL=cfg.Ca,
            problem=cfg.problem,
            N=cfg.N,
            solver=solver
        )

def _test3_slope_limiters_blast2():
    """Blast2, N=128, PPM+Roe: minmod / vanLeer / MC vs FOG+Roe"""
    filename = "test3.pkl"
    if os.path.exists(filename):
        os.remove(filename)
    # PPM + Roe with each limiter
    for limiter in ["minmod", "vanLeer", "MC"]:
        cfg = Config(
            method="PPM",
            limiter=limiter,
            solver="Roe",
            problem="blast2",
            N=128,
            Ca=0.8,
        )
        U_final = main_without_plots(cfg)
        save_simulation_data(
            filename=filename,
            U_final=U_final,
            method=cfg.method,
            limiter=limiter,
            CFL=cfg.Ca,
            problem=cfg.problem,
            N=cfg.N,
            solver=cfg.solver
        )
    # baseline FOG + Roe
    cfg = Config(
        method="FOG",
        limiter="",
        solver="Roe",
        problem="blast2",
        N=128,
        Ca=0.8,
    )
    U_final = main_without_plots(cfg)
    save_simulation_data(
        filename=filename,
        U_final=U_final,
        method=cfg.method,
        limiter=cfg.limiter,
        CFL=cfg.Ca,
        problem=cfg.problem,
        N=cfg.N,
        solver=cfg.solver
    )


def _test4_grid_resolution_shu_osher():
    """Shu–Osher, PLM+MC+HLL: N=32,64,128 vs FOG+HLL (N=128)"""
    filename = "test4.pkl"
    if os.path.exists(filename):
        os.remove(filename)
    for N in [32, 64, 128]:
        # run PLM+MC+HLL for N=32,64,128
        cfg = Config(
            method="PLM",
            limiter="MC",
            solver="HLL",
            problem="shu-osher",
            N=N,
            Ca=0.8,
        )
        U_final = main_without_plots(cfg)
        save_simulation_data(
            filename=filename,
            U_final=U_final,
            method=cfg.method,
            limiter=cfg.limiter,
            CFL=cfg.Ca,
            problem=cfg.problem,
            N=cfg.N,
            solver=cfg.solver
        )
        # run FOG for N=32,64,128
        cfg = Config(
            method="FOG",
            limiter="",
            solver="HLL",
            problem="shu-osher",
            N=N,
            Ca=0.8,
        )
        U_final = main_without_plots(cfg)
        save_simulation_data(
            filename=filename,
            U_final=U_final,
            method=cfg.method,
            limiter=cfg.limiter,
            CFL=cfg.Ca,
            problem=cfg.problem,
            N=cfg.N,
            solver=cfg.solver
        )


def _test5_cfl_study_shu_osher():
    """Shu–Osher, PLM+vanLeer+Roe: CFL = 0.2, 0.4, 0.6, 0.8, 1.0"""
    """     -> Note: CFL=1.4 is omitted due to instability"""

    filename = "test5.pkl"
    if os.path.exists(filename):
        os.remove(filename)
    for Ca in [0.2, 0.4, 0.6, 0.8, 1.0]:
        cfg = Config(
            method="PLM",
            limiter="vanLeer",
            solver="Roe",
            problem="shu-osher",
            N=128,
            Ca=Ca,
        )
        U_final = main_without_plots(cfg)
        save_simulation_data(
            filename=filename,
            U_final=U_final,
            method=cfg.method,
            limiter=cfg.limiter,
            CFL=cfg.Ca,
            problem=cfg.problem,
            N=cfg.N,
            solver=cfg.solver
        )


def run_all_tests():
    _test1_reconstruction_methods()
    _test2_riemann_solvers()
    _test3_slope_limiters_blast2()
    _test4_grid_resolution_shu_osher()
    _test5_cfl_study_shu_osher()


# if we want to run all tests at once
# if __name__ == "__main__":
#     run_all_tests()

_test2_riemann_solvers()
