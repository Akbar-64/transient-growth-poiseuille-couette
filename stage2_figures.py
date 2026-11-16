"""
Stage 2 figures. Run AFTER transient_growth.py in the same notebook (or keep both files together).
Fig 4: G(t) curves.  Fig 5: peak growth vs Re vs Trefethen et al. (1993).  Fig 6: scan over beta.
"""
import numpy as np
import matplotlib.pyplot as plt

try:
    gmax
except NameError:
    from transient_growth import growth_curve, gmax

N = 64

# Fig 4: energy amplification G(t), alpha = 0, beta = 2.04
fig4, ax4 = plt.subplots(figsize=(6, 4.5))
for Re in (250, 500, 1000, 2000):
    t = np.linspace(0.5, 0.35 * Re, 80)
    ax4.plot(t / Re, growth_curve(Re, 0.0, 2.04, t, N), label=f"Re = {Re}")
ax4.set_xlabel("t / Re")
ax4.set_ylabel("energy amplification G(t)")
ax4.set_title("Transient growth, plane Poiseuille, alpha = 0, beta = 2.04")
ax4.legend()
fig4.tight_layout()
fig4.savefig("fig4_growth_curves.png", dpi=200)

# Fig 5: peak growth vs Re, compared with (Re/71.5)^2 from Trefethen et al. (1993), Table 1
Res = np.array([100, 200, 400, 800, 1600, 3200])
Gs = np.array([gmax(R, 0.0, 2.04, N)[0] for R in Res])
big = Res >= 400                                  # asymptotic law only holds at larger Re
slope = np.polyfit(np.log(Res[big]), np.log(Gs[big]), 1)[0]
print("Fitted slope of log Gmax vs log Re (Re >= 400):", round(slope, 3), "(theory: 2)")
fig5, ax5 = plt.subplots(figsize=(6, 4.5))
ax5.loglog(Res, Gs, "o", label="this code")
ax5.loglog(Res, (Res / 71.5) ** 2, "-", label=r"$(Re/71.5)^2$, Table 1")
ax5.set_xlabel("Reynolds number")
ax5.set_ylabel(r"$G_{max}$")
ax5.set_title(f"Peak growth vs Re (fitted slope {slope:.2f})")
ax5.legend()
fig5.tight_layout()
fig5.savefig("fig5_Gmax_vs_Re.png", dpi=200)

# Fig 6: which spanwise wavenumber grows most? Re = 1000, alpha = 0
betas = np.linspace(1.0, 3.5, 26)
Gb = np.array([gmax(1000, 0.0, b, N)[0] for b in betas])
print("Best beta on this grid:", betas[np.argmax(Gb)], " (Table 1: 2.04)")
fig6, ax6 = plt.subplots(figsize=(6, 4.5))
ax6.plot(betas, Gb, "-o", ms=3)
ax6.set_xlabel(r"spanwise wavenumber $\beta$")
ax6.set_ylabel(r"$G_{max}$")
ax6.set_title("Optimal spanwise wavenumber, Re = 1000, alpha = 0")
fig6.tight_layout()
fig6.savefig("fig6_beta_scan.png", dpi=200)
plt.show()
