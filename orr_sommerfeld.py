"""
Step 2 starter: Orr-Sommerfeld eigenvalues for plane Poiseuille flow.

Flow between walls at y = -1 and y = +1, base flow U(y) = 1 - y^2.
Perturbation: v(x, y, t) = vhat(y) * exp(i*alpha*x) * exp(lam*t)
Equation:     lam * (D2 - k2) vhat = L vhat
with L = -i*alpha*U*(D2 - k2) + i*alpha*U'' + (1/Re)*(D2 - k2)^2
Wall conditions: vhat = 0 and D vhat = 0 at y = +1 and y = -1.

Wave speed:  c = i*lam/alpha.  Growth rate: Re(lam) = alpha * Im(c).
If any eigenvalue has Re(lam) > 0, the flow is linearly unstable.
"""
import numpy as np
from scipy.linalg import eig
from scipy.optimize import brentq


def cheb(N):
    """Chebyshev points x (from +1 to -1) and differentiation matrix D.
    Follows Trefethen, Spectral Methods in MATLAB, program cheb."""
    if N == 0:
        return np.array([1.0]), np.zeros((1, 1))
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2, np.ones(N - 1), 2]) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return x, D


def os_eigenvalues(Re, alpha, N=96):
    """All finite eigenvalues lam of the Orr-Sommerfeld problem (Poiseuille)."""
    y, D = cheb(N)
    D2 = D @ D
    D4 = D2 @ D2
    I = np.eye(N + 1)
    k2 = alpha**2

    U = 1 - y**2
    Upp = -2 * np.ones_like(y)

    M = D2 - k2 * I
    L = (-1j * alpha * (np.diag(U) @ M)
         + 1j * alpha * np.diag(Upp)
         + (D4 - 2 * k2 * D2 + k2**2 * I) / Re)

    # Impose the four wall conditions by replacing four rows.
    # Zero rows in M turn those rows into algebraic constraints.
    L = L.astype(complex)
    M = M.astype(complex)
    for row in (0, 1, N - 1, N):
        L[row, :] = 0
        M[row, :] = 0
    L[0, 0] = 1                 # v(+1) = 0
    L[N, N] = 1                 # v(-1) = 0
    L[1, :] = D[0, :]           # Dv(+1) = 0
    L[N - 1, :] = D[N, :]       # Dv(-1) = 0

    lam = eig(L, M, right=False)
    lam = lam[np.isfinite(lam)]
    lam = lam[np.abs(lam) < 1e4]  # drop huge spurious values
    return lam


def least_stable(Re, alpha, N=96):
    """Eigenvalue with the largest growth rate, plus its wave speed c."""
    lam = os_eigenvalues(Re, alpha, N)
    lam0 = lam[np.argmax(lam.real)]
    c = 1j * lam0 / alpha
    return lam0, c


def max_growth(Re, alpha, N=96):
    return least_stable(Re, alpha, N)[0].real


if __name__ == "__main__":
    # Benchmark 1: Orszag (1971), Re = 10000, alpha = 1
    lam0, c = least_stable(10000, 1.0, N=96)
    print("Re=10000, alpha=1")
    print("  computed c  =", c)
    print("  published c = 0.23752649 + 0.00373967j")

    # Benchmark 2: convergence in N
    print("\nConvergence of Im(c):")
    for N in (48, 64, 96, 128):
        print(f"  N={N:3d}  Im(c) = {least_stable(10000, 1.0, N)[1].imag:.8f}")

    # Benchmark 3: critical Reynolds number at alpha = 1.02056
    alpha_c = 1.02056
    Re_c = brentq(lambda Re: max_growth(Re, alpha_c), 5000, 6500, xtol=1e-3)
    print(f"\nCritical Re at alpha={alpha_c}: {Re_c:.2f}  (published ~5772.22)")
