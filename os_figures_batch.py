"""
Batch figures: one growth-rate plot per Reynolds number, plus the neutral stability curve.
Run AFTER orr_sommerfeld.py (same notebook). Saves PNGs and one zip file.
"""
import numpy as np
import matplotlib.pyplot as plt

try:
    max_growth
except NameError:
    from orr_sommerfeld import max_growth

N = 96

# ---------- A. one growth-rate figure per Reynolds number ----------
alphas = np.linspace(0.5, 1.5, 41)
for Re in (5000, 5700, 5800, 6000, 8000):
    g = np.array([max_growth(Re, a, N) for a in alphas])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(alphas, g, "-o", ms=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"streamwise wavenumber $\alpha$")
    ax.set_ylabel(r"largest growth rate Re($\lambda$)")
    ax.set_title(f"Growth rate vs alpha, Re = {Re}")
    fig.tight_layout()
    fig.savefig(f"fig2_Re{Re}.png", dpi=200)
    plt.close(fig)
    print("saved", f"fig2_Re{Re}.png")

# ---------- B. neutral stability curve ----------
Re_list = np.array([5800, 6000, 6500, 7000, 8000, 10000, 12000, 15000, 20000])
alpha_grid = np.linspace(0.6, 1.3, 71)       # spacing 0.01
lower, upper = [], []
for Re in Re_list:
    g = np.array([max_growth(Re, a, N) for a in alpha_grid])
    idx = np.where(g > 0)[0]
    lower.append(alpha_grid[idx.min()] if idx.size else np.nan)
    upper.append(alpha_grid[idx.max()] if idx.size else np.nan)
    print(Re, lower[-1], upper[-1])

figN, axN = plt.subplots(figsize=(6, 4.5))
axN.plot(Re_list, lower, "-o", ms=4, label="lower edge")
axN.plot(Re_list, upper, "-o", ms=4, label="upper edge")
axN.axvline(5772.22, color="gray", ls="--", lw=0.8, label="Re = 5772.22")
axN.set_xscale("log")
axN.set_xlabel("Reynolds number")
axN.set_ylabel(r"streamwise wavenumber $\alpha$")
axN.set_title("Neutral stability curve, plane Poiseuille (grid-resolved)")
axN.legend()
figN.tight_layout()
figN.savefig("fig3_neutral_curve.png", dpi=200)

# ---------- C. zip everything for one download ----------
import glob, zipfile
with zipfile.ZipFile("figures.zip", "w") as z:
    for f in glob.glob("fig*.png"):
        z.write(f)
print("zipped:", sorted(glob.glob("fig*.png")))
plt.show()
