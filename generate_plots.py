# generate and save plots from test data

import matplotlib.pyplot as plt
import pickle
import os

from conversions import *

all_records = []

# open test data (replace "1" by the test number)
with open("test1.pkl", "rb") as f:
    while True:
        try:
            rec = pickle.load(f)
            all_records.append(rec)
        except EOFError:
            break


# Map problem to physical domain
DOMAIN = {
    'sod':         (0.0,  1.0),
    'rarefaction': (0.0,  1.0),
    'blast2':      (0.0,  1.0),
    'shu-osher':   (-4.5, 4.5),
}

# Plot styles
STYLE3 = [
    {"color": "orange", "linestyle": "--", "marker": "o", "markersize": 4, "linewidth": 0.7, "mfc": 'none', "mew": 0.8},
    {"color": "blue",   "linestyle": "--", "marker": "s", "markersize": 3, "linewidth": 0.7, "mfc": 'none', "mew": 0.8},
    {"color": "green",  "linestyle": "--", "marker": "D", "markersize": 3, "linewidth": 0.7, "mfc": 'none', "mew": 0.8},
]
STYLE2 = [
    {"color": "orange", "linestyle": "--", "marker": "o", "markersize": 4, "linewidth": 0.7, "mfc": 'none', "mew": 0.8},
    {"color": "green",  "linestyle": "--", "marker": "s", "markersize": 3, "linewidth": 0.7, "mfc": 'none', "mew": 0.8},
]

def _get_x(problem, N):
    xa, xb = DOMAIN[problem]
    dx = (xb - xa)/N
    # interior cells only
    return xa + (np.arange(N)+0.5)*dx

def save_plots_1(records):
    """Test1: Sod, N=128, HLL: 3 plots (FOG, PLM, PPM) with overlay (rho, u, p)."""
    N = records[0]['N']
    x = _get_x(records[0]['problem'], N)

    # labels for primitive variables
    var_labels = [r'$\rho$', r'$u$', r'$p$']

    for rec in records:
        method = rec['method']
        U = rec['U_final']
        V = cons_to_prim(U, None)

        ngc = 2

        # one figure per reconstruction method
        fig, ax = plt.subplots(figsize=(6, 4))
        for k, label in enumerate(var_labels):
            ax.plot(
                x,
                V[k, ngc:-ngc],
                label=label,
                **STYLE3[k]
            )

        ax.set_title(f"{method} (N={N})")
        ax.set_xlabel('x')
        ax.set_ylabel('primitive variables')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8, loc='best')

        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/test1_{method}.png", dpi=600)
        plt.close(fig)

def save_plots_2(records):
    """Test2: Rarefaction, N=128, PLM+minmod: 2 plots (HLL, Roe) overlay (rho, u, p).
        Omitting Roe plot.
    """

    ngc = 2

    for solver in ["HLL"]:
        # select the record matching this solver
        rec = next(r for r in records if r['solver'] == solver)
        N = rec['N']
        x = _get_x(rec['problem'], N)

        V = cons_to_prim(rec['U_final'], None)

        rho = V[0, ngc:-ngc]
        u   = V[1, ngc:-ngc]
        p   = V[2, ngc:-ngc]

        # create overlay plot for this solver
        plt.figure(figsize=(6, 4))
        plt.plot(x, rho, **STYLE3[0], label=r'$\rho$')
        plt.plot(x, u,   **STYLE3[1], label=r'$u$')
        plt.plot(x, p,   **STYLE3[2], label=r'$p$')

        plt.title(f"{solver} (N={N}, PLM+minmod)")
        plt.xlabel('x')
        plt.ylabel('primitive variables')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=8)

        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/test2_{solver}.png", dpi=600)
        plt.close()

def save_plots_3(records):
    """Test3: Blast2, N=128, PPM+Roe w/ 3 limiters, then FOG baseline."""
    # PPM+Roe overlay limiters
    recs_ppm = records[:3]

    N = recs_ppm[0]['N']
    x = _get_x(recs_ppm[0]['problem'], N)
    ngc = 2

    plt.figure()
    for i, rec in enumerate(recs_ppm):
        V = cons_to_prim(rec['U_final'], None)
        plt.plot(x, V[0, ngc:-ngc],
                 label=rec['limiter'],
                 **STYLE3[i])
    plt.xlabel('x')
    plt.ylabel('density')
    plt.legend(fontsize=8, loc='best')
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/test3_ppm_limiters.png", dpi=600)
    plt.close()

    # FOG+Roe baseline
    rec0 = records[3]
    N = rec0['N']
    x = _get_x(rec0['problem'], N)
    plt.figure()
    V = cons_to_prim(rec0['U_final'], None)
    plt.plot(x, V[0, ngc:-ngc],
             label='FOG',
             **STYLE2[0])
    plt.xlabel('x')
    plt.ylabel('density')
    plt.legend(fontsize=8, loc='best')
    plt.savefig("plots/test3_fog_baseline.png", dpi=600)
    plt.close()

def save_plots_4(records):
    """Test4: Shu-Osher, N=32,64,128: 3 plots overlay FOG vs PLM."""
    # Determine grid resolutions
    Ns = sorted({rec['N'] for rec in records})

    for N in Ns:
        # Find a record to supply problem and method for _get_x
        first_rec = next(rec for rec in records if rec['N'] == N)
        x = _get_x(first_rec['problem'], N)

        plt.figure()
        # Plot both methods for this N
        for rec in records:
            if rec['N'] != N:
                continue
            method = rec['method']
            V = cons_to_prim(rec['U_final'], None)
            ngc = 2
            style = STYLE2[0] if method == 'FOG' else STYLE2[1]

            plt.plot(
                x,
                V[0, ngc:-ngc],
                label=method,
                **style
            )

        plt.xlabel('x')
        plt.ylabel('density')
        plt.legend(fontsize=8, loc='best')

        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/test4_N{N}.png", dpi=600)
        plt.close()

def save_plots_5(records):
    """Test5: Shu-Osher, PLM+vanLeer+Roe: 5 plots, one per CFL (excluding CFL=1.4)."""
    for rec in records:
        N = rec['N']
        x = _get_x(rec['problem'], N)
        plt.figure()
        V = cons_to_prim(rec['U_final'], None)
        ngc = 2
        plt.plot(x, V[0, ngc:-ngc],
                 label=f"CFL={rec['CFL']}",
                 **STYLE3[0])
        plt.xlabel('x')
        plt.ylabel('density')
        plt.legend(fontsize=8, loc='best')
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/test5_CFL{rec['CFL']}.png", dpi=600)
        plt.close()


if __name__ == "__main__":
    save_plots_1(all_records)
