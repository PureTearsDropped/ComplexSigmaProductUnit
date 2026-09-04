# Learning the unit: gradients averaged, or estimates averaged?

`physics/gd_vs_moments.py`, `physics/outlier_test.py`, results in `physics/results/`.  The aggregate problem —
U units from **one** observable y = Σ_u a_u·exp(W_u·Log₀ x), exponents and coefficients together, random
complex targets, 8 seeds; success = relative loss < 1e−8 and max|W − true| < 1e−3 (best unit permutation).

Three learners on the same layer (total arithmetic, float32 values):

- **Adam** — the gradients of all samples averaged, one step, repeated (full batch, 2000 steps, symmetry-broken
  start: identical units get identical gradients and never separate — measured, 0/8 from the all-ones start).
- **estimates averaged** (`csigma_fit.py`) — the model is linear in a given W, and linear in the exponents' correction
  to first order; so a = the least-squares solution (the general form of "the mean"), and δW = the least-squares
  average of the per-sample equations y_n − ŷ_n = Σ a_u P_nu (δW_u·L_n) (the general form of "the mean of the
  ratios"): variable projection's Gauss–Newton, a step halved until the loss decreases.  One unit at a time — the
  literal per-sample ratio Log₀(r_n/(a_u P_nu)) averaged — diverged under 1e−7 noise (the other units leak into
  the ratio), so the units are solved together.
- **von Mises centre** — the same per-sample corrections, but their centre taken robustly: the von Mises mean
  direction of the phases, the median magnitude, the resultant length R̄ as the confidence that scales the step
  (the centre of the gradient distribution instead of its mean).
- The **matrix-pencil start**: on a uniform grid of L (a log-spaced sweep, F = 1) y_n = Σ a_u z_u^n is an
  exponential sum in log x — Prony — and the Hankel pencil reads the z_u = e^{W_u Δ} off the data without
  iteration (|Im W|·Δ < π; the *left* singular vectors — the right ones give the conjugates, measured).

| problem | Adam | lstsq mean, random start | von Mises, random start | lstsq mean, pencil start | von Mises, pencil start |
|---|---|---|---|---|---|
| F=1 U=2, log grid | 7/8 | 0/8 | 0/8 | **8/8** | 8/8 |
| F=1 U=3, log grid | 4/8 | 0/8 | 0/8 | **8/8** | 7/8 |
| F=1 U=4, log grid | 2/8 | 0/8 | 0/8 | **8/8** | 7/8 |
| F=2 U=2, complex box | 4/8 | 3/8 | 0/8 | — | — |
| F=2 U=3, complex box | 1/8 | 0/8 | 0/8 | — | — |

The pencil start with the least-squares average is exact where Adam degrades with U (loss ~1e−13 in 1–3
iterations, milliseconds against a minute).  What decides is the **start**, i.e. the algebraic structure of the
model, not the averaging rule: from a random start neither average finds the units (0/8), and the von Mises
centre does not change that — a wrong start scatters the per-sample phases, R̄ is small, the steps shrink.

**Noise and outliers** (`outlier_test.py`: U = 2 on the grid, 0.1 % noise on every sample, plus gross outliers
— a random factor of 2–4 in magnitude, ±2 in phase — on 0 / 5 / 15 % of the samples):

| outliers | start | lstsq mean: median · worst \|W − true\| | von Mises: median · worst |
|---|---|---|---|
| 0 % | pencil | 9e−2 · 9.9 | 3e−1 · 12 |
| 0 % | truth | 7e−5 · 3e−4 | 2e−5 · 1e−4 |
| 5 % | pencil | 7.4 · 8.7 | 7.7 · 9.2 |
| 5 % | truth | 2e−3 · 2e−2 | 1e−3 · 3e−2 |
| 15 % | truth | 9e−3 · 3e−2 | **2e−3** · 6e−2 |

Two findings.  The pencil is fragile: 0.1 % noise on a 40-point sweep already puts its U = 2 start out of the
basin (median error 0.09, worst 10), and any outliers destroy it — in practice a robust initializer is the open
problem, not the refinement.  The von Mises centre is the better refinement under outliers (median 6× closer at
15 %, 2× at 5 %, when started near the truth) and equal without them — a modest, real gain as a step rule; it is
not a fix for the landscape.  Both are recorded as measured; neither is tuned.
