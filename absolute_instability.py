"""
Stage 3A: local absolute-instability analysis (Briggs-Bers saddle point) for the inviscid Rayleigh equation.

Rayleigh equation for the wall-normal velocity amplitude phi(y), disturbance ~ exp(i*alpha*x - i*omega*t):
    (U - c) (phi'' - alpha^2 phi) - U'' phi = 0,      omega = alpha * c
Dirichlet conditions phi = 0 at both ends of the domain [y0, y1]
(for a free shear layer take a wide domain; for a wall-bounded profile put the wall at y0).

A flow is ABSOLUTELY unstable if the saddle point alpha_s of omega(alpha) in the complex alpha-plane,
d omega / d alpha = 0, has Im(omega_s) > 0. The threshold is Im(omega_s) = 0.

Benchmark: tanh mixing layer U = 1 + R tanh(y). Huerre and Monkewitz (1985) give the threshold R = 1.315,
where R = (U1 - U2)/(U1 + U2).

NOT implemented: the pinch check (that the two spatial branches meeting at the saddle come from opposite
half-planes). The solver is validated instead by reproducing the published threshold.
"""
import numpy as np
from scipy.linalg import eig
from scipy.optimize import brentq


def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2, np.ones(N - 1), 2]) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return x, D


class Rayleigh:
    """Rayleigh eigenproblem on [y0, y1] with a given base flow U(y) and its second derivative."""

    def __init__(self, U_func, Upp_func, y0, y1, N=200, stretch=0.0):
        """stretch = 0: plain Chebyshev grid mapped to [y0, y1] (clusters points at BOTH ends, e.g. a wall at y0).
        stretch = b > 0: sinh map y = ym + (h) sinh(b s)/sinh(b), which puts more points in the MIDDLE of a
        symmetric domain (use it for free shear layers centred at y = 0)."""
        s, D = cheb(N)
        ym, h = (y0 + y1) / 2, (y1 - y0) / 2
        if stretch == 0:
            yp, ypp = h * np.ones_like(s), np.zeros_like(s)
            self.y = ym + h * s
        else:
            b = stretch
            self.y = ym + h * np.sinh(b * s) / np.sinh(b)
            yp = h * b * np.cosh(b * s) / np.sinh(b)
            ypp = h * b**2 * np.sinh(b * s) / np.sinh(b)
        Dy = np.diag(1 / yp) @ D
        self.D2 = np.diag(1 / yp**2) @ (D @ D) - np.diag(ypp / yp**3) @ D
        self.U = U_func(self.y)
        self.Upp = Upp_func(self.y)
        self.n = N + 1

    def c_eigenvalues(self, alpha):
        """All finite wave speeds c for (possibly complex) alpha."""
        I = np.eye(self.n)
        B = (self.D2 - alpha**2 * I).astype(complex)
        A = (np.diag(self.U) @ B - np.diag(self.Upp)).astype(complex)
        for r in (0, self.n - 1):               # phi = 0 at both ends
            A[r, :] = 0
            B[r, :] = 0
            A[r, r] = 1
        c = eig(A, B, right=False)
        c = c[np.isfinite(c)]
        return c[np.abs(c) < 50]

    def most_unstable_real(self, alphas):
        """Scan real alpha, return (alpha, c) with the largest temporal growth rate alpha*Im(c)."""
        best = None
        for a in alphas:
            c = self.c_eigenvalues(a)
            cm = c[np.argmax(c.imag)]
            g = a * cm.imag
            if best is None or g > best[0]:
                best = (g, a, cm)
        return best[1], best[2]

    def omega(self, alpha, c_guess):
        """Follow one eigenvalue branch: pick the eigenvalue closest to the previous c."""
        c = self.c_eigenvalues(alpha)
        c = c[np.argmin(np.abs(c - c_guess))]
        return alpha * c, c

    def saddle(self, alpha0, c0, h=2e-3, tol=1e-8, maxit=60):
        """Newton iteration for d omega / d alpha = 0 in the complex alpha-plane."""
        a, c = complex(alpha0), c0
        for _ in range(maxit):
            w0, c = self.omega(a, c)
            wp, _ = self.omega(a + h, c)
            wm, _ = self.omega(a - h, c)
            dw = (wp - wm) / (2 * h)
            d2w = (wp - 2 * w0 + wm) / h**2
            step = dw / d2w
            step = step if abs(step) < 0.2 else 0.2 * step / abs(step)     # damp large steps
            a = a - step
            if abs(step) < tol:
                break
        w, c = self.omega(a, c)
        return a, w, c


def tanh_layer(R, L=40.0, N=200, stretch=3.0):
    return Rayleigh(lambda y: 1 + R * np.tanh(y),
                    lambda y: -2 * R * np.tanh(y) / np.cosh(y) ** 2, -L, L, N, stretch)


def saddle_tanh(R, L=40.0, N=200, stretch=3.0, start=None):
    """Dominant saddle of the tanh layer: returns (alpha_s, omega_s, c_s). `start`=(alpha, c) to continue from."""
    prob = tanh_layer(R, L, N, stretch)
    if start is None:
        a0, c0 = prob.most_unstable_real(np.linspace(0.1, 1.0, 19))
        start = (a0 + 0.0j, c0)
    a, w, c = prob.saddle(*start)
    return a, w, c


def threshold_tanh(L=40.0, N=200, stretch=3.0, h=2e-3, Rlo=1.25, Rhi=1.45):
    """R at which Im(omega_s) = 0 for the tanh layer, found by root finding with continuation."""
    a, w, c = saddle_tanh(1.4, L, N, stretch)
    state = {"s": (a, c)}

    def f(R):
        prob = tanh_layer(R, L, N, stretch)
        a, w, c = prob.saddle(*state["s"], h=h)
        state["s"] = (a, c)
        return w.imag

    return brentq(f, Rlo, Rhi, xtol=1e-6)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1) Im(omega_s) against R
    Rs = np.arange(1.0, 1.61, 0.05)
    start, rows = None, []
    print("   R     alpha_s (complex)           omega_s (complex)        Im(omega_s)")
    for R in Rs:
        a, w, c = saddle_tanh(R, start=start)
        start = (a, c)                                   # continuation in R
        rows.append((R, w.imag))
        print(f"{R:5.2f}   {a.real:8.4f}{a.imag:+8.4f}i      {w.real:8.4f}{w.imag:+8.4f}i     {w.imag:+.5f}")

    # 2) threshold and convergence with resolution
    print("\nThreshold R_c (Im omega_s = 0) for different resolutions:")
    for N in (160, 240, 320):
        print(f"   N = {N}: R_c = {threshold_tanh(N=N):.4f}")
    Rc = threshold_tanh()
    print(f"\nR_c = {Rc:.4f}   (Huerre and Monkewitz 1985: 1.315)")

    # 3) figure
    rr, wi = np.array(rows).T
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(rr, wi, "-o", ms=4)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(1.315, color="gray", ls="--", lw=0.8, label="R = 1.315 (published)")
    ax.set_xlabel(r"$R = (U_1 - U_2)/(U_1 + U_2)$")
    ax.set_ylabel(r"Im($\omega_s$) at the saddle point")
    ax.set_title("Absolute-instability threshold, tanh mixing layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig10_absolute_instability_benchmark.png", dpi=200)
    plt.show()
