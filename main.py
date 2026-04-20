import math
import matplotlib.pyplot as plt
import time

from config import Config
import boundary_conditions as bc
from conversions import *

from update import update
from dynamic_plots import plot_init, plot_update
import warnings
from warnings import catch_warnings

go = True
should_quit = False

def compute_dt(U, params):
    gamma = params.gamma
    dx    = params.dx
    Ca    = params.Ca

    V = cons_to_prim(U, params)     # shape (3, N)
    rho, u, p = V

    with catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # ensures all warnings are caught

        # sound speed
        a = np.sqrt(gamma * p / rho)

        if w:
            print(f"{w[0].category.__name__}: {w[0].message} (from {w[0].filename}, line {w[0].lineno})")
            print('sound speed: ', a)

            for i in range(len(a)):
                if math.isnan(a[i]):
                    print('nan index: ', i)

            print()
            time.sleep(1000)

    # maximum signal speed
    max_speed = np.max(np.abs(u) + a)

    # CFL time step
    dt = Ca * dx / max_speed

    return dt

# run the simulation in real-time with a live updating plot display
def main_with_plots():
    global go, should_quit
    go = True
    should_quit = False

    def on_key(event):
        global go, should_quit
        if event.key == ' ':
            go = not go
        elif event.key == 'escape':
            should_quit = True

    # set configuration
    params = Config(
        method="PPM",
        limiter="MC",
        solver="Roe",
        problem="blast2",
        N=128,
        Ca=0.8,
    )

    U = params.U0.copy()                  # initial conservative state
    U = bc.update_BCs(U, params)          # sets ghost cells appropriately

    U_init = U.copy()
    V = cons_to_prim(U, params)
    t = 0.0

    # ─── PLOT SETUP ──────────────────────────────────────────────────────────────
    fig, ax, line, buffer_frac, eps = plot_init(
        params.x,
        V[0],
        params,
        on_key,
        buffer_frac=0.10,   # optional override
        eps=1e-8            # optional override
    )

    # ─── MAIN UPDATE LOOP ───────────────────────────────────────────────────────
    while t <= params.tmax and not should_quit:
        # pause loop
        if not go:
            plt.pause(0.1)
            continue

        dt = compute_dt(U, params)

        # update solution
        U = update(U, dt, params)
        V = cons_to_prim(U, params)

        # update plot with smooth expansion
        plot_update(V[0], ax, line, buffer_frac, eps, params)

        # adjust to slow down or speed up simulation
        plt.pause(0.01)

        t += dt

    U_final = U.copy()

    # ─── CLEANUP ────────────────────────────────────────────────────────────────
    plt.ioff()
    plt.show()

    return U_init, U_final, params

# used for testing or if plotting display is not wanted
def main_without_plots(cfg):
    U = cfg.U0.copy()  # initial conservative state
    U = bc.update_BCs(U, cfg)  # sets ghost cells appropriately using cfg.V0 if needed

    t = 0.0

    # ─── Main Update Loop ───────────────────────────────────────────────────────
    while t <= cfg.tmax:
        dt = compute_dt(U, cfg)

        # update data
        U = update(U, dt, cfg)

        t += dt

    U_final = U.copy()

    return U_final


# run with live plots
ui, uf, params = main_with_plots()
