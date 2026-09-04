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

## Deciding the exponents: R as a candidate detector, verification as the judge

The user's trace of the wave (per-sample gradient directions, von Mises R per feature, 5 seeds) showed that
the direction consensus is **not** monotone and **not** a truth signal: at update 430 of the reference seed
R_T = 0.96 while the T exponent is still 2.09 off (a real, damping error with the phase almost right — every
sample agrees on the *same* correction), and the truth is reached only after further, weaker consensus
events.  Selecting, among the checkpoints with max R ≥ 0.3, the one whose coefficients refitted from scratch
give the smallest held-out error picked the minimum-error checkpoint in 5 of 5 seeds.  So R says "a
coherent hypothesis has appeared", and a cheap verification says whether it is the right one.

Two checks on this side (`physics/gn_from_trajectory.py`, `physics/hybrid_r_trigger.py`,
`physics/decide_exponents.py`, results in `physics/results/`):

- **High R is the condition of the linearisation.**  One estimates-averaged fit (Gauss–Newton with the
  coefficients re-solved) from the user's own trajectory points, on their data: update 430 (R 0.96, error 2.1)
  → machine precision in 6 iterations; update 475 (R 0.92) → 2 iterations; and for this single unit even
  update 300 (R 0.27, error 4.7) → 23 iterations.  A 900-step detour for one unit is not necessary; the
  R events are where the fit is cheapest.
- **On multi-unit aggregates the R gate (0.5 on the correction cloud) rarely fired**, so "Adam + R-triggered
  fit" equalled Adam alone (4/8, 2/8, 4/8, 1/8 on the four problems), and a fit attempted every 10 steps
  regardless of R gave the same success — a fit from a random-ish point does not find the units.  What did:
  **rounding the exponents to the admissible lattice and verifying by the linear refit's residual**, tried
  every 10 steps — integer-exponent aggregates with complex coefficients, random start, Adam alone 0/8:

| problem | Adam | + R-triggered fit | + R-triggered fit + round + verify | + round + verify every 10 steps |
|---|---|---|---|---|
| F=2 U=2 | 0/8 | 0/8 | 2/8 | **5/8** |
| F=3 U=3 | 0/8 | 0/8 | 0/8 | **4/8** |
| series RLC from the total impedance (F=4, U=3) | 0/8 | 0/8 | 0/8 | **5/8** |

The RLC's three units recovered from the aggregate impedance alone — the open item of `EXPERIMENT_RESULT.md`
— for the first time, by the sieve: the lattice proposes, the residual decides, and the decision is taken
while Adam is still far away (median step 1 390 of 2 000).  Rounding at the R events only (threshold 0.5)
was worse because the events are rare on multi-unit problems; a lower threshold, or none, costs one least
squares per check.  The two-stage rule the user proposed — R marks candidates; W fixed → a by least squares
(the exact form of the refit; no need to re-train it) → held-out error selects → freeze W, final fit — is
the same scheme with a continuous candidate set; the lattice version is its discrete form.  Not yet done:
noise, the wrong-but-coherent basins as a *sequence of hypotheses* (the user's reading), and the aliases
(a wrong exponent on which the samples agree is one the data cannot distinguish over their span).

## The three-stage rule, tried (`physics/three_stage.py`, results in `physics/results/three_stage.txt`)

Adam runs; every 10 steps a candidate is taken — either always, or only when the per-exponent von Mises R of the
per-sample gradient directions (the user's statistic) reaches 0.3 — and at a candidate one estimates-averaged fit
is made from the current W (the joint least squares: direction and length, no oracle needed), plus its rounding
to the lattice where the exponents are known to be discrete; W fixed, a by least squares, and the **held-out**
error of each candidate decides.  8 seeds, symmetry-broken random start.

| problem | Adam alone | fit every 10 steps → held-out | R ≥ 0.3 candidates → fit → held-out |
|---|---|---|---|
| plane wave, amplitude and exponents together (U = 1) | 8/8 at step 255 | 8/8 at step 10 | 8/8 at step 10 (1 candidate) |
| F=1 U=3 log grid | 4/8 | 4/8 | 3/8 (19 candidates) |
| F=1 U=4 log grid | 2/8 | 3/8 | 2/8 (32) |
| F=2 U=2 complex box | 4/8 | 4/8 | 4/8 (46) |
| F=2 U=3 complex box | 1/8 | 1/8 | 1/8 (57) |
| series RLC from the total impedance, integer lattice | 0/8 | **8/8 at step 390** | 3/8 (2.5 candidates) |

Read: (1) for one unit the first candidate already finishes — one least-squares fit at step 10 replaces 255
Adam steps, and the user's 900-step detour was never necessary; (2) with the lattice and the held-out judge
the RLC's three units are recovered from the aggregate in every seed (the earlier round-and-verify without
the least-squares step and without the held-out split: 5/8) — the sieve works when the candidates are dense;
(3) the R gate at 0.3 cuts the candidates 3–5× but never adds a success and sometimes removes one (the RLC:
3/8, the gate fires only 2–3 times in 2 000 steps) — as a *selector* R is a cost saving, not information, at
this threshold; the information is in the verification; (4) for continuous multi-unit aggregates none of the
three beats Adam: the wall there is the start (the pencil where the design allows it), not the step rule.

## The steering sweep (the research note's Stage A + B), and what it says

`physics/steering_sweep.py` (harvest / sweep / analyze), `steering_analyze2.py`, `steering_analyze3.py`; results in
`physics/results/steering*`.  Three tasks (wave, F=1 U=3 log grid, F=2 U=2 complex box) × 8 seeds × 2 000 Adam steps,
the per-sample gradients in closed form (g_n = −2w·e_n·conj(a·P_n·L_n)), every 10 steps the circular moments of
every exponent weight (M₁ = R e^{iμ}, κ, M₂, M₃), 4 824 checkpoints.  From 1 019 of them (every 50 steps + the three
highest-R events per trajectory) 275 branches at once as one batch — α ∈ ±30° (11) × ρ ∈ {0.5…1.5} (5) ×
κ_steer ∈ {5, 10, 20, 50, ∞} (5), the step of every exponent weight taking the direction φ ~ VM(μ+π+α, κ_steer)
with Adam's length × ρ, the coefficients stepping with Adam, K = 30 — 280 225 rollouts, 36 min on the RTX 5090
(the forward is the Tot arithmetic; at this size the GPU is launch-bound, 33 ms per step, and the CPU is no slower).

Three readings of the same table, in the order they were made — the first two are traps:

1. *Best of 275 against the α = 0, ρ = 1, κ = ∞ branch*: median "gain" 64×, > 10× at 668 of 1 019 checkpoints,
   the winner almost always stochastic.  This is the winner's curse — the maximum of 275 draws — and the sign of
   the best α agrees with the Gauss–Newton direction's side at chance (443/844).
2. *Deterministic branches only, the floor excluded*: the step length dominates (ρ = 0.5 best at 690 of 778
   checkpoints, median 16×), a best-of-11 α still 6× — and the best α keeps its sign across consecutive
   checkpoints 81 % of the time.  Still against the wrong baseline.
3. **Against the true Adam continuation** (the harvest trajectory itself, K steps on): the α = 0, ρ = 1, κ = ∞
   branch is not Adam — it keeps Adam's step *length* but takes the direction μ+π of the unit-vector mean of the
   per-sample gradients, which is not the gradient — and it is ~100× worse than Adam after 30 steps (median
   ratio 0.01).  Every steered branch, including the best of all 275 per checkpoint, is worse than plain Adam
   at the median (0.60), and **no branch beats Adam by 2× at any of the 631 checkpoints** where Adam is not
   at the floor; the same holds in every R bin and on every task.  The 81 % sign consistency of reading 2 is the
   consistency of the correction that rotates a bad direction back toward Adam's.

So, as specified in the note (μ+π as the base direction, a rotation α, a length ρ, von Mises noise), the
steering does not beat the optimizer it starts from on this horizon — the note's own 0° steering was slower
than Adam (117 vs 110 updates) and its +5° gain (94) is the size of such a rotation back.  What does beat Adam
here is known and needs no sweep: one joint least-squares fit from the same checkpoints (2–6 iterations to the
floor from the consensus events; `gn_from_trajectory.py`).  The sweep harness stays: it runs any branching
study at 275 branches per batch, and the checkpoints with M₂, M₃ are saved for the other question the note asks
— whether the higher moments at low R predict which basin a trajectory is heading for — which is not answered
here.  Not done: a sweep with Adam's own direction as the base (α rotating the true gradient step), longer
horizons, and the controller (Stage C–D), which this result puts on hold.

## Depth: the log-stream residual (`csigma_deep.py`, `physics/deep_test.py`, results in `physics/results/deep_test.txt`)

A block is a Σ-PU whose F outputs are exponent increments, Δ_ℓ = Σ_u a_ℓ·exp(W_ℓ·L_{ℓ−1}), and the stream adds them,
L_ℓ = L_{ℓ−1} + Δ_ℓ ⟺ x_ℓ = x_{ℓ−1}·exp(Δ_ℓ): the residual is multiplicative, Log₀ is taken once at the entry, the
stream keeps the unwrapped phase, and a = 0 makes every block the identity at the start.  Two targets a single
Σ-PU cannot represent: a chirp phase then a power law (Δ = 0.5i·√ω, head V = −0.8), and the chirp, then an
amplitude law Δ = 0.8·ω^{−0.3}, then a head.  Adam from the identity start, 3 000 steps, 6 seeds, held-out
relative loss (median / worst):

| model | target 1 | target 2 |
|---|---|---|
| single Σ-PU, U = 4 | 1.9e−2 / 3.2e−2 | 8.5e−5 / 1.9e−4 |
| log-stream, 1 block U=1 (the true architecture) | 8.3e−3 / 1.5e−1 | 9.0e−5 / 9.1e−3 |
| log-stream, 2 blocks U=2 + head by least squares | 5.9e−6 / 1.1e−5 | 3.7e−6 / 2.1e−5 |
| log-stream, 3 blocks U=2 + head by least squares | 2.4e−6 / 5.4e−6 | 7.2e−7 / 2.2e−5 |
| 1 + i·y/s embedding, 2 blocks U=2 (s = 1 / 4) | 7.1e−3 / 2.2e−2 | 1.2e−4 / 1.6e−4 |

The function class is right — the stream fits three to four orders below a single unit and the 1 + i·y/s
embedding (whose atan-saturated phase cannot carry the chirp) — and, as everywhere in this study, gradient
descent is what stops short: none reaches 1e−8, the exact architecture does worst (the identity start is a
saddle for it), and overparametrised streams do better.  Inverting the head layer by layer ("final-stage-first")
failed here because a head fitted on the identity stream is biased (held-out 0.1).

What decided it: **under the log the depth collapses.**  With a one-unit head, the phase of y unwrapped along
the grid gives Log y = Log c + V·L₀ + V·a·e^{W·L₀} — linear in (Log c, V, V·a) for a given W, a variable
projection over one complex unknown, sieved on a 41 × 41 grid and refined by Gauss–Newton: W = 0.5, V = −0.8,
a = 0.5i, c = 1.5 − 0.5i to machine precision (residual 4e−31) where 3 000 Adam steps left 1e−3 … 1e−6.  The
lesson is the same one as the pencil's: the model's algebra (here: a product of exponentials is a sum of
exponents, so a one-unit head makes the stream observable) is worth more than any step rule.  With two blocks
or a wider head the collapse is only partial (Δ₂ = Σ a₂ e^{W₂(L₀+Δ₁)} is nested), and that is the next question;
the phase unwrap needs the samples ordered along a path (a sweep), as the pencil needs a grid.
