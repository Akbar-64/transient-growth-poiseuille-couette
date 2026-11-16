"""
Stage 2: transient energy growth in plane Poiseuille flow (velocity-vorticity form).
Benchmarks come from Trefethen, Trefethen, Reddy and Driscoll (1993), Science 261, Table 1.

State: v = wall-normal velocity, eta = wall-normal vorticity, for one Fourier mode
exp(i*alpha*x + i*beta*z).  Lengths scaled by half-height, speeds by centreline speed.

    d/dt (D2 - k2) v = L_OS v
    d/dt eta         = -i*beta*U' v + L_SQ eta
with  k2 = alpha^2 + beta^2,
    L_OS = -i*alpha*U*(D2 - k2) + i*alpha*U'' + (1/Re)*(D2 - k2)^2
    L_SQ = -i*alpha*U + (1/Re)*(D2 - k2)
Wall conditions: v = Dv = eta = 0 at y = +-1.

Wall conditions are built into the basis (no row replacement):
    v = (1 - y^2)^2 * q(y),   eta = interior values only.
Energy norm: E = integral( |Dv|^2 + k2*|v|^2 + |eta|^2 ) dy   (constant factors cancel in G).
Amplification:  G(t) = max E(t)/E(0) = || F expm(A t) F^-1 ||_2^2,  Q = F^H F.
"""
import numpy as np
from scipy.linalg import expm, cholesky, solve, eigvals, eig, qr, solve_triangular


def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2, np.ones(N - 1), 2]) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return x, D


def clencurt(N):
    """Clenshaw-Curtis quadrature weights on the Chebyshev points (Trefethen 2000, program 30)."""
    theta = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    ii = np.arange(1, N)
    v = np.ones(N - 1)
    if N % 2 == 0:
        w[0] = w[N] = 1 / (N**2 - 1)
        for k in range(1, N // 2):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1)
        v -= np.cos(N * theta[ii]) / (N**2 - 1)
    else:
        w[0] = w[N] = 1 / N**2
        for k in range(1, (N - 1) // 2 + 1):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1)
    w[ii] = 2 * v / N
    return w


def diffmat_nodes(x):
    """Differentiation matrix for polynomial interpolation on arbitrary nodes (barycentric form)."""
    n = len(x)
    diff = x[:, None] - x[None, :]
    np.fill_diagonal(diff, 1.0)
    logb = -np.sum(np.log(np.abs(diff)), axis=1)
    sgn = np.prod(np.sign(diff), axis=1)          # diagonal entries are +1, ignored
    b = sgn * np.exp(logb - logb.max())           # common scale cancels in ratios
    D = (b[None, :] / b[:, None]) / diff
    np.fill_diagonal(D, 0.0)
    D -= np.diag(D.sum(axis=1))
    return D


def build(Re, alpha, beta, N=64, flow="poiseuille", extras=False):
    """Return A (dx/dt = A x) and energy matrix Q for the state x = [v_int, eta_int].
    flow = 'poiseuille' (U = 1 - y^2) or 'couette' (U = y, walls at y = -1, +1)."""
    y, Dfull = cheb(N)
    s = slice(1, N)
    xi = y[s]                      # interior nodes, n = N-1 of them
    n = N - 1
    I = np.eye(n)
    k2 = alpha**2 + beta**2

    # v = (1-y^2)^2 q, with q a polynomial of degree N-2 fixed by its n interior values.
    # This builds v = Dv = 0 at both walls into the basis (same idea as Weideman & Reddy's cheb4c).
    Dn = diffmat_nodes(xi)
    Dq = [I, Dn, Dn @ Dn, Dn @ Dn @ Dn, Dn @ Dn @ Dn @ Dn]
    ws = [(1 - xi**2) ** 2, 4 * xi**3 - 4 * xi, 12 * xi**2 - 4, 24 * xi, 24 * np.ones_like(xi)]
    binom = [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]

    def Mder(m):
        out = np.zeros((n, n))
        for k in range(m + 1):
            out += binom[m][k] * np.diag(ws[k]) @ Dq[m - k]
        return out

    M0, M1, M2, M4 = Mder(0), Mder(1), Mder(2), Mder(4)
    if flow == "poiseuille":
        U, Up, Upp = 1 - xi**2, -2 * xi, -2 * np.ones_like(xi)
    elif flow == "couette":
        U, Up, Upp = xi.copy(), np.ones_like(xi), np.zeros_like(xi)
    else:
        raise ValueError("flow must be 'poiseuille' or 'couette'")

    Lap = M2 - k2 * M0
    LOS = (-1j * alpha * np.diag(U) @ Lap + 1j * alpha * np.diag(Upp) @ M0
           + (M4 - 2 * k2 * M2 + k2**2 * M0) / Re)
    D2full = Dfull @ Dfull
    LSQ = -1j * alpha * np.diag(U) + (D2full[s, s] - k2 * I) / Re   # Dirichlet: interior block
    Lc = -1j * beta * np.diag(Up) @ M0

    A = np.zeros((2 * n, 2 * n), dtype=complex)
    A[:n, :n] = solve(Lap, LOS)
    A[n:, :n] = Lc
    A[n:, n:] = LSQ

    W = np.diag(clencurt(N)[s])
    Q = np.zeros((2 * n, 2 * n), dtype=complex)
    Q[:n, :n] = M1.T @ W @ M1 + k2 * M0.T @ W @ M0
    Q[n:, n:] = W
    # Change state variable from q to v = (1-y^2)^2 q. Eigenvalues are unchanged, but the
    # energy matrix becomes far better conditioned (q-coordinates are badly scaled near the walls).
    sc = np.concatenate([ws[0], np.ones(n)])
    A = (sc[:, None] * A) / sc[None, :]
    Q = Q / sc[:, None] / sc[None, :]
    if extras:
        return A, Q, {"y": xi, "Dv_from_v": M1 / ws[0][None, :], "n": n}
    return A, Q


def reduced_operator(Re, alpha, beta, N=64, m=None, flow="poiseuille"):
    """Project the problem onto the m least-damped eigenmodes, orthonormalised in the energy norm.

    Why: the collocation matrix also contains mesh-scale modes that are numerical junk. In the
    energy norm they cause fake growth at very short times. Keeping only the m least-damped modes
    (m = N by default) removes them. This is the same idea as Reddy and Henningson (1993).
    Returns Atilde such that G(t) = || expm(Atilde * t) ||_2 ^ 2.
    """
    if m is None:
        m = N
    A, Q = build(Re, alpha, beta, N, flow)
    lam, X = eig(A)
    X = X[:, np.argsort(-lam.real)[:m]]
    F = cholesky(Q, lower=False)             # Q = F^H F
    Finv = solve_triangular(F, np.eye(F.shape[0]), lower=False)
    Qr, _ = qr(F @ X, mode="economic")       # orthonormal basis in the energy norm
    return Qr.conj().T @ (F @ A @ Finv) @ Qr


def growth_curve(Re, alpha, beta, times, N=64, m=None, flow="poiseuille"):
    At = reduced_operator(Re, alpha, beta, N, m, flow)
    return np.array([np.linalg.norm(expm(At * t), 2) ** 2 for t in times])


def gmax(Re, alpha, beta, N=64, tmax=None, nt=60, m=None, flow="poiseuille"):
    """Maximum over time of G(t) and the time where it occurs (coarse scan, then refinement)."""
    if tmax is None:
        tmax = 0.5 * Re
    At = reduced_operator(Re, alpha, beta, N, m, flow)
    Gof = lambda ts: np.array([np.linalg.norm(expm(At * t), 2) ** 2 for t in ts])
    times = np.linspace(0.5, tmax, nt)
    G = Gof(times)
    i = int(np.argmax(G))
    fine = np.linspace(times[max(i - 1, 0)], times[min(i + 1, nt - 1)], 25)
    Gf = Gof(fine)
    j = int(np.argmax(Gf))
    return Gf[j], fine[j]


def optimal_perturbation(Re, alpha, beta, t_opt, N=64, m=None, flow="poiseuille"):
    """Optimal initial condition for the target time t_opt, and what it has become by then.

    The optimal initial condition is the first right singular vector of the propagator,
    mapped back to physical variables (v, eta). Returns nodes and v, Dv, eta at t=0 and t.
    """
    if m is None:
        m = N
    A, Q, ex = build(Re, alpha, beta, N, flow, extras=True)
    lam, X = eig(A)
    X = X[:, np.argsort(-lam.real)[:m]]
    F = cholesky(Q, lower=False)
    Finv = solve_triangular(F, np.eye(F.shape[0]), lower=False)
    Qr, _ = qr(F @ X, mode="economic")
    At = Qr.conj().T @ (F @ A @ Finv) @ Qr
    E = expm(At * t_opt)
    _, S, Vh = np.linalg.svd(E)
    r = Vh[0].conj()                         # optimal initial condition (reduced coordinates)
    x0 = Finv @ (Qr @ r)                     # physical state at t = 0
    xT = Finv @ (Qr @ (E @ r))               # physical state at time t
    n = ex["n"]
    out = {"y": ex["y"], "G": S[0] ** 2, "Gmax": S[0] ** 2, "Q": Q, "x0": x0, "xT": xT}
    for tag, x in (("0", x0), ("T", xT)):
        out["v" + tag] = x[:n]
        out["Dv" + tag] = ex["Dv_from_v"] @ x[:n]
        out["eta" + tag] = x[n:]
    return out


if __name__ == "__main__":
    # Check 1: the TS eigenvalue must still match Orszag (Re=10000, alpha=1, beta=0)
    A, Q = build(10000, 1.0, 0.0, N=96)
    ev = eigvals(A)
    lam = ev[np.argmax(ev.real)]
    c = 1j * lam / 1.0
    print("Check 1 (Orszag): c =", c, " published 0.23752649+0.00373967j")

    # Check 2: no growth blow-up. All eigenvalues of A at Re=1000, alpha=0 should decay.
    A, Q = build(1000, 0.0, 2.04, N=64)
    print("Check 2: max Re(eigenvalue) at Re=1000, alpha=0, beta=2.04 =", eigvals(A).real.max())

    # Check 2b: growth must start from G = 1 at t = 0
    print('Check 2b: G(t=0.001) =', growth_curve(1000, 0.0, 2.04, [1e-3], N=64)[0], '(must be about 1)')

    # Check 3: Trefethen et al. (1993), Table 1. Poiseuille, alpha=0, beta=2.04:
    # amplitude growth ~ Re/71.5 -> energy G ~ (Re/71.5)^2, at t_opt ~ Re/13.2
    for Re in (500, 1000, 2000):
        G, t = gmax(Re, 0.0, 2.04, N=64)
        print(f"Check 3: Re={Re:5d}  Gmax={G:8.2f} at t={t:7.2f}   |  Table 1: {(Re/71.5)**2:8.2f} at t={Re/13.2:7.2f}")
