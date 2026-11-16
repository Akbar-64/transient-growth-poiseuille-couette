"""
Figures 1 and 2 for the Orr-Sommerfeld solver.
Run AFTER orr_sommerfeld.py (same notebook, or keep both files in one folder).
"""
import numpy as np
import matplotlib.pyplot as plt

try:
    os_eigenvalues
except NameError:
    from orr_sommerfeld import os_eigenvalues, least_stable, max_growth

# ---------- Figure 1: eigenvalue spectrum in the wave-speed plane ----------
Re, alpha = 10000, 1.0
lam = os_eigenvalues(Re, alpha, N=96)
c = 1j * lam / alpha                       # wave speed c = i*lambda/alpha

keep = (c.real > -0.1) & (c.real < 1.1) & (c.imag > -1.0) & (c.imag < 0.1)
fig1, ax1 = plt.subplots(figsize=(6, 4.5))
ax1.plot(c.real[keep], c.imag[keep], "o", ms=4)
ax1.axhline(0, color="k", lw=0.6)
ax1.set_xlabel(r"$c_r$ (wave speed)")
ax1.set_ylabel(r"$c_i$ (growth if > 0)")
ax1.set_title(f"Orr-Sommerfeld spectrum, Poiseuille, Re={Re}, alpha={alpha:g}", fontsize=11)
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-1.0, 0.05)
fig1.tight_layout()
fig1.savefig("fig1_spectrum.png", dpi=200)

# ---------- Figure 2: growth rate vs alpha at Re = 8000 ----------
Re2 = 8000
alphas = np.linspace(0.5, 1.5, 41)
growth = np.array([max_growth(Re2, a, N=96) for a in alphas])

fig2, ax2 = plt.subplots(figsize=(6, 4.5))
ax2.plot(alphas, growth, "-o", ms=3)
ax2.axhline(0, color="k", lw=0.6)
ax2.set_xlabel(r"streamwise wavenumber $\alpha$")
ax2.set_ylabel(r"largest growth rate Re($\lambda$)")
ax2.set_title(f"Unstable band at Re={Re2}")
fig2.tight_layout()
fig2.savefig("fig2_growth_vs_alpha.png", dpi=200)

unstable = alphas[growth > 0]
if unstable.size:
    print(f"Unstable for alpha between {unstable.min():.3f} and {unstable.max():.3f} (grid resolution 0.025)")
else:
    print("No unstable alpha found at this Re")
print("Max growth rate:", growth.max(), "at alpha =", alphas[np.argmax(growth)])
plt.show()
