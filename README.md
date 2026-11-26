# Transient energy growth in plane Poiseuille and Couette flow (Python)

A small, validated Python implementation of linear stability and transient (non-modal) growth
for plane channel flows, written to reproduce known results from the literature.

**Status: reproduction and validation. This repository contains no new physical results.**
Its purpose is to show a working, tested pipeline that reproduces published benchmarks, and to serve
as the base for further work on transition to turbulence.

Author: Muhammad Akbar. The code was written with the assistance of an AI model (Claude, Anthropic)
and then run and checked by the author against published benchmark values (see below).

## What it computes

1. **Modal stability** of plane Poiseuille flow: Orr-Sommerfeld eigenvalues, the neutral stability curve,
   the critical Reynolds number.
2. **Transient energy growth** G(t) in the velocity-vorticity (Orr-Sommerfeld and Squire) formulation,
   for Poiseuille and Couette flow, including the optimal initial perturbation (streamwise vortices that
   turn into streaks).

Lengths are scaled by the channel half-height. Poiseuille flow: U = 1 - y^2, Re uses the centreline speed.
Couette flow: U = y, walls at y = -1 and +1.

## Validation

| Check | This code | Reference |
|---|---|---|
| Least-stable eigenvalue, Poiseuille, Re = 10000, alpha = 1 | c = 0.23752649 + 0.00373967i | 0.23752649 + 0.00373967i (Orszag 1971) |
| Critical Reynolds number | 5772.22 at alpha = 1.02056 | 5772.22 (Orszag 1971) |
| Peak energy growth, Poiseuille, Re = 1000, alpha = 0, beta = 2.04 | 196.17 at t = 76.0 | (Re/71.5)^2 = 195.6 at t = Re/13.2 = 75.8 (Trefethen et al. 1993, Table 1) |
| Peak growth, Poiseuille, Re = 2000 | 783.8 | 782.4 |
| Scaling of peak growth with Re (Re >= 400) | slope 2.00 (log-log) | 2 |
| Peak growth, Couette, Re = 3200, alpha = 0, beta = 1.66 | 11936 | (Re/29.3)^2 = 11928 |
| Absolute-instability threshold, tanh mixing layer, inviscid | R = 1.3157 | 1.315 (Huerre and Monkewitz 1985) |

The numbers above were reproduced in independent runs (the author's Google Colab session and a separate
local Python 3.12 run); the runs agree to the digits shown. Exact trailing digits can differ slightly between
machines and library versions.

Table 1 of Trefethen et al. (1993) gives asymptotic values, stated accurate to about 1% for Re > 100.
It reports amplitude growth; this code reports energy growth, which is its square.

## How to run

Locally:

```
pip install -r requirements.txt
python orr_sommerfeld.py        # Stage 1: eigenvalue benchmarks
python os_figures.py            # figures 1 and 2
python os_figures_batch.py      # per-Re growth plots, neutral curve (figure 3), and figures.zip
python transient_growth.py      # Stage 2 checks against Trefethen et al. Table 1
python stage2_figures.py        # figures 4 to 6
python stage2b_figures.py       # figures 7 to 9: (alpha, beta) maps, Couette, optimal perturbation
python absolute_instability.py  # stage 3A: absolute-instability benchmark (tanh mixing layer)
```

Figures are written to the folder you run the scripts from.

In Google Colab: upload the zip (for example with `from google.colab import files; files.upload()`),
then run in notebook cells

```
!unzip -o transient-growth-poiseuille-couette.zip
%cd transient-growth-poiseuille-couette
!mkdir -p results/figures results/colab_outputs
!python orr_sommerfeld.py   > results/colab_outputs/01_orr_sommerfeld.txt
!python os_figures.py       > results/colab_outputs/02_os_figures.txt
!python os_figures_batch.py > results/colab_outputs/03_os_figures_batch.txt
!python transient_growth.py > results/colab_outputs/04_transient_growth.txt
!python stage2_figures.py   > results/colab_outputs/05_stage2_figures.txt
!python stage2b_figures.py  > results/colab_outputs/06_stage2b_figures.txt
!cp fig*.png results/figures/
```

This regenerates every figure and every output file in `results/`.

To record your own library versions: `python -c "import sys, numpy, scipy, matplotlib; print(sys.version, numpy.__version__, scipy.__version__, matplotlib.__version__)"`

Tested with Python 3.12 on two library sets: numpy 1.26.4 / scipy 1.11.4 / matplotlib 3.8.4, and
numpy 2.5.4 / scipy 1.18.1 / matplotlib 3.11.2.

## Results (figures)

The figures in `results/figures/` were produced by the scripts above, run by the author in Google Colab.
`results/colab_outputs/` holds the printed output of each script from the same run, as text files
(`01_orr_sommerfeld.txt` to `06_stage2b_figures.txt`).

| File | Shows | Made by |
|---|---|---|
| `fig1_spectrum.png` | Orr-Sommerfeld spectrum, Poiseuille flow, Re = 10000, alpha = 1 | `os_figures.py` |
| `fig2_growth_vs_alpha.png` | Largest growth rate against alpha, Re = 8000 | `os_figures.py` |
| `fig2_Re5000.png`, `fig2_Re5700.png`, `fig2_Re5800.png`, `fig2_Re6000.png`, `fig2_Re8000.png` | Same, at other Reynolds numbers (stable below Re = 5772.22) | `os_figures_batch.py` |
| `fig3_neutral_curve.png` | Neutral stability curve (grid-resolved) | `os_figures_batch.py` |
| `fig4_growth_curves.png` | Energy amplification G(t) for several Re | `stage2_figures.py` |
| `fig5_Gmax_vs_Re.png` | Peak growth against Re, compared with (Re/71.5)^2 | `stage2_figures.py` |
| `fig6_beta_scan.png` | Peak growth against spanwise wavenumber beta | `stage2_figures.py` |
| `fig7_alpha_beta_map.png` | Peak growth over the (alpha, beta) plane, Poiseuille and Couette | `stage2b_figures.py` |
| `fig8_poiseuille_vs_couette.png` | Peak growth against Re, both flows | `stage2b_figures.py` |
| `fig9_optimal_perturbation.png` | Optimal perturbation: streamwise vortices turning into streaks | `stage2b_figures.py` |
| `fig10_absolute_instability_benchmark.png` | Im(omega) at the saddle point against R; zero crossing at R = 1.3157 | `absolute_instability.py` |

## Method notes

* Chebyshev collocation (Trefethen 2000). Wall conditions v = Dv = 0 are built into the basis by writing
  v = (1 - y^2)^2 q(y), in the spirit of the cheb4c routine of Weideman and Reddy (2000).
* Energy norm E = integral of (|Dv|^2 + k^2 |v|^2 + |eta|^2) dy with Clenshaw-Curtis weights.
* G(t) = || F expm(A t) F^-1 ||_2^2 with energy matrix Q = F^H F.
* The propagator is projected onto the N least-damped eigenmodes (as in Reddy and Henningson 1993). Without
  this, mesh-scale modes of the collocation matrix produce spurious growth at very short times.
* Stage 1 (`orr_sommerfeld.py`) imposes wall conditions by row replacement; Stage 2 uses the basis described above.
  Both reproduce the same Orszag eigenvalue.

## Limitations

* Linear theory only, one Fourier mode at a time, channel geometry (not pipe flow).
* Benchmarked at the parameters listed above; accuracy at much higher Re has not been tested.
* No nonlinear stage, no direct numerical simulation.
* The absolute-instability solver (stage 3A) is inviscid and does not check the pinch condition. It is validated by reproducing the published threshold of the tanh mixing layer.

## References

* Orszag, S. A. (1971). Accurate solution of the Orr-Sommerfeld stability equation. J. Fluid Mech. 50, 689-703.
* Trefethen, L. N., Trefethen, A. E., Reddy, S. C., Driscoll, T. A. (1993). Hydrodynamic stability without eigenvalues. Science 261, 578-584.
* Reddy, S. C., Henningson, D. S. (1993). Energy growth in viscous channel flows. J. Fluid Mech. 252, 209-238.
* Weideman, J. A. C., Reddy, S. C. (2000). A MATLAB differentiation matrix suite. ACM Trans. Math. Softw. 26, 465-519.
* Trefethen, L. N. (2000). Spectral Methods in MATLAB. SIAM.
* Schmid, P. J., Henningson, D. S. (2001). Stability and Transition in Shear Flows. Springer.

(Check page and volume details against the originals before citing.)
