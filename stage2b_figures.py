"""
Stage 2b figures. Run AFTER transient_growth.py (same notebook, or keep both files together).
Fig 7: maximum amplification over the (alpha, beta) plane, Poiseuille and Couette, Re = 1000.
Fig 8: peak growth vs Re for both flows, against Trefethen et al. (1993) Table 1.
Fig 9: the optimal perturbation at t = 0 and at t = t_opt (vortices turning into streaks).
"""
import numpy as np
import matplotlib.pyplot as plt

import inspect

try:                                  # running as a script from the project folder
    from transient_growth import gmax, optimal_perturbation, clencurt
except ModuleNotFoundError:           # running as a notebook cell: the functions must already exist
    pass

_missing = [n for n in ("gmax", "optimal_perturbation", "clencurt") if n not in globals()]
if _missing or "flow" not in inspect.signature(gmax).parameters:
    raise RuntimeError(
        "This script needs the LATEST transient_growth.py (with Couette support and "
        "optimal_perturbation). Paste and run the latest version of that file first. "
        f"Missing names: {_missing}")

# ------------------------------------------------------------------ Fig 7
Re = 1000
alphas = np.arange(0.0, 2.01, 0.1)
betas = np.arange(0.0, 4.01, 0.2)
fig7, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, flow in zip(axes, ("poiseuille", "couette")):
    G = np.full((len(betas), len(alphas)), np.nan)
    for i, b in enumerate(betas):
        for j, a in enumerate(alphas):
            if a == 0 and b == 0:
                continue                      # k = 0 is degenerate
            G[i, j] = gmax(Re, a, b, N=40, tmax=0.3 * Re, nt=40, flow=flow)[0]
    k = np.nanargmax(G)
    ib, ia = np.unravel_index(k, G.shape)
    print(f"{flow:10s}: largest G on grid = {G[ib, ia]:.1f} at alpha={alphas[ia]:.1f}, beta={betas[ib]:.1f}"
          f"   | G at beta=0 (2D waves) max = {np.nanmax(G[0, :]):.1f}")
    cs = ax.contourf(alphas, betas, G, levels=20)
    fig7.colorbar(cs, ax=ax, label=r"$G_{max}$")
    ax.plot(alphas[ia], betas[ib], "k*", ms=12)
    ax.set_xlabel(r"streamwise wavenumber $\alpha$")
    ax.set_ylabel(r"spanwise wavenumber $\beta$")
    ax.set_title(f"{flow.capitalize()}, Re = {Re}")
fig7.tight_layout()
fig7.savefig("fig7_alpha_beta_map.png", dpi=200)

# ------------------------------------------------------------------ Fig 8
Res = np.array([100, 200, 400, 800, 1600, 3200])
GP = np.array([gmax(R, 0.0, 2.04, N=64)[0] for R in Res])
GC = np.array([gmax(R, 0.0, 1.66, N=64, tmax=0.3 * R, flow="couette")[0] for R in Res])
print("Re   Poiseuille  (Re/71.5)^2   Couette  (Re/29.3)^2")
for R, gp, gc in zip(Res, GP, GC):
    print(f"{R:5d} {gp:10.1f} {(R/71.5)**2:10.1f} {gc:10.1f} {(R/29.3)**2:10.1f}")
fig8, ax8 = plt.subplots(figsize=(6, 4.5))
ax8.loglog(Res, GP, "o", label="Poiseuille (this code)")
ax8.loglog(Res, (Res / 71.5) ** 2, "-", label=r"$(Re/71.5)^2$")
ax8.loglog(Res, GC, "s", label="Couette (this code)")
ax8.loglog(Res, (Res / 29.3) ** 2, "--", label=r"$(Re/29.3)^2$")
ax8.set_xlabel("Reynolds number")
ax8.set_ylabel(r"$G_{max}$ ($\alpha = 0$)")
ax8.set_title("Peak transient growth vs Re")
ax8.legend()
fig8.tight_layout()
fig8.savefig("fig8_poiseuille_vs_couette.png", dpi=200)

# ------------------------------------------------------------------ Fig 9
Re, beta = 1000, 2.04
Gm, topt = gmax(Re, 0.0, beta, N=64)
op = optimal_perturbation(Re, 0.0, beta, topt, N=64)
y = op["y"]
z = np.linspace(0, 2 * np.pi / beta, 49)
ph = np.exp(1j * beta * z)

def fields(v, Dv, eta):
    u_hat = -1j * eta / beta           # from eta = i*beta*u when alpha = 0
    w_hat = 1j * Dv / beta             # from continuity: i*beta*w + Dv = 0
    pad = lambda f: np.concatenate([[0], f, [0]])      # zeros at the two walls
    yy = np.concatenate([[1.0], y, [-1.0]])
    u = np.real(np.outer(pad(u_hat), ph))
    v_ = np.real(np.outer(pad(v), ph))
    w = np.real(np.outer(pad(w_hat), ph))
    return yy, u, v_, w

yy, u0, v0, w0 = fields(op["v0"], op["Dv0"], op["eta0"])
_, uT, vT, wT = fields(op["vT"], op["DvT"], op["etaT"])
umax = np.abs(uT).max()
vec0 = np.hypot(v0, w0).max()
fig9, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (u, v_, w, title) in zip(axs, ((u0, v0, w0, "t = 0: streamwise vortices"),
                                       (uT, vT, wT, f"t = {topt:.0f}: streaks"))):
    cs = ax.contourf(z, yy, u, levels=np.linspace(-umax, umax, 21), cmap="RdBu_r")
    ax.quiver(z[::4], yy[::3], w[::3, ::4], v_[::3, ::4], scale=vec0 * 8, color="k")
    ax.set_xlabel("z (one spanwise wavelength)")
    ax.set_title(title)
axs[0].set_ylabel("y")
fig9.colorbar(cs, ax=axs, label="streamwise velocity u (same scale in both panels)")
fig9.savefig("fig9_optimal_perturbation.png", dpi=200, bbox_inches="tight")
plt.show()
