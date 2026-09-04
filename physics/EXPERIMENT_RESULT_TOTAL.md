# Complex Product Unit on total arithmetic — Experiment 3 and the boundaries

Companion to `EXPERIMENT_RESULT.md`.  The same two experiments, a third one, and the boundaries the
first two "deliberately stayed away from", run on the complex Product Unit implemented on
**total-arith-cuda** ≥ v1.2.0 (`complex_sigma_product_unit.py` in this repository: `ComplexSigmaProductUnit`,
a `torch.nn.Module` on the `Tot` value+flag arithmetic; twin `julia/ScalarTotComplex.jl` there).  Runner: `physics_total.py`
(`--device cuda` for the GPU); the numbers below are the RTX 5090 run
(`results/physics_total_cuda_result.txt`, 201 s); the CPU run (`results/physics_total_cpu_result.txt`, 272 s)
gives the same numbers.

## What the arithmetic fixes (the specification list of §32, answered by the implementation)

| §32 item | choice |
|---|---|
| Arg₀ branch | (−π, π] |
| Arg₀(0, 0) | 0 — the reserved word: a zero has no direction |
| L₀(0) | 0 — a zero has no scale; with Arg₀, Log₀(0) = 0 and the input **drops out of the product** (P × 1) |
| L₀(ε) | log MIN with the flag ⟦≥⟧ ("at least this negative"): LOG_MIN as a *tagged boundary value*, not −∞ |
| −ε, complex ε | ε is the arithmetic's own underflow MIN⟦≤⟧, per component, direction kept; Arg₀ is atan2 of the components |
| U saturation | e^U → MAX⟦≥⟧ / MIN⟦≤⟧, the phase V untouched (§26, §28.2 — amplitude-only saturation) |
| backward at saturation | a flagged ⟦≥⟧ / ⟦±⟧ value is not a number: the sample leaves the loss (`masked_cmse`), the parameter coordinate leaves the step (`totalize_grads`) |
| gradient floor | the arithmetic: a gradient that underflows is ±MIN⟦≤⟧ with its sign and is used as is (§22, applied before it can become 0 — §24) |
| ε fixed or moving | fixed = float32 MIN (1.2e-38) |
| complex gradients | real derivatives on (A, B) and (a_re, a_im) through autograd on the values |
| 4-channel redundancy | softplus + eps channels (as in the demos), no regulariser |
| MAX / LOG_MIN / ε per dtype | float32: 3.4e38 / −87.34 / 1.2e-38 |

Two readings of a zero input exist and both are total: the normal form exp(W·Log₀ x) of §29 lets
the zero drop out (P = 1), the reserved-word power 0^W = 0 makes the product vanish.  The
document fixes the first and so does the layer (`Tot`'s Julia twin offers both).

## The learning schedule

Stage 1 fits the exponents on ratios to one reference sample — the coefficient cancels — and
stage 2 freezes them and fits the complex coefficients on the full observable, as before.  Two
details were forced by the data: for **imaginary exponents** (the wave) the ratio loss
|P_n/P_0 − q_n|² of the original demos is used (periodic in the phase, no 2π ambiguity); for
**real exponents over many decades** (the CPE, six decades of ω) the fit is done in the
unit's own log domain, (U_n − U_0, V_n − V_0) against Log₀ q_n with the phase compared modulo
2π — linear in W, where the ratio loss is a 1e12-steep valley in which Adam stalls (measured).
A flagged element — on the model side *or on the target side* — is no number and is left out;
dropping the target flags put a 5.6e37⟦≥⟧ into a loss once (loss 6e74: measured, fixed).

## Results (RTX 5090, float32 values; IEEE = the same formulas on plain values)

| experiment | IEEE | total |
|---|---|---|
| 1 RLC, ordinary range: exponents (max error) / coefficients (max error) | 4.8e-7 / 3.4e-6 | 1.5e-6 / 3.2e-6 |
| 1 RLC **+ c = 0, w = 0, c = ε** (10 % each) | **NaN at step 0** | 4.9e-7 / 3.7e-6, 903 + 300 flagged elements left out |
| 2 wave, ordinary: B (true 1.7, −2.89) / amplitude (true 0.8 − 0.6i) | 1.7, −2.89 / 0.800001 − 0.6i | same |
| 2 wave **+ X = e^{±100}** (float32 overflow / underflow, 10 % each) | **NaN at step 0** | 1.7, −2.89 / 0.800001 − 0.6i, 661 + 660 left out |
| 3 CPE, ordinary: W (true −0.8) / a, r (true 1, 0.002) | −0.800000 / 1.000000, 0.002000 | same |
| 3 CPE **+ ω = 0, ω = 1e−42, 1e−45** (Log₀ 0, ε) | **NaN at step 0** | −0.800000 / 1.000000, 0.002000, 3 + 2 left out |

Every stage stops at a loss of ~1e−13, the float32 floor of the values.  The IEEE runs die in the
first forward pass of each boundary set (log 0 = −inf, 0·(−inf) = NaN, inf/inf = NaN); the total
runs keep every ordinary sample and recover the same numbers.

## Experiment 3 — the constant-phase element

Z(ω) = R + 1/(Q·(iω)^α), α = 0.8 — a fractional power of a *complex* variable that exists in the
laboratory (electrochemical impedance; Warburg at α = ½).  Dimensionless: z = r + (iω)^{−α},
one Product Unit with a complex input s = iω and W = −α, plus a constant unit (W = 0) carrying r.
The phase of the unit is constant, −απ/2 = −0.4π at every ω: **arg reads the exponent**.  At the
boundaries: ω = 0 gives Log₀(i·0) = 0 and the unit drops out (P = 1, no phase — a zero has no
direction); ω = 1e−42 enters as ε = MIN⟦≤⟧ and gives |P| ≥ 1e30 with the phase −0.4π kept (a
blocking electrode, as a flagged limit rather than an infinity).  The same distinction in the
RLC: c = 0 makes the capacitor term drop out (a short — "no such element", as the arithmetic
reads an absent factor) while c = ε gives an open circuit ⟦≥⟧; physics wants the open at
C → 0, which says the natural variable of a series capacitor is the elastance 1/C, whose 0 *is*
the absent element.

## The branch cut, and Kahan's signed zero

s^{−α} on the negative real axis of the Laplace plane has two values, one per side.  IEEE stores
the side in the sign of the zero imaginary part: (−1 + 0i)^{−0.8} = −0.809 − 0.588i,
(−1 − 0i)^{−0.8} = −0.809 + 0.588i.  Total arithmetic has one zero and one arg for it, so
(−1)^{−0.8} is the principal value; the two sides are the two *values* −1 ± iε — and the
four-channel representation, being Cartesian, keeps im = ±MIN⟦≤⟧ and returns the two results
(flags ⟦≥≤±⟧ on the direction: the rule "a bound on one component turns the pair" does not yet
see that the bound is 1e−38 of the other).  The polar Julia twin stores arg/π in a Float64 and
rounds −1 + 7e−309 to −1, i.e. to the other side: a limitation of the polar representation, not
of the arithmetic, recorded in `results/cpe_total_demo_stdout.txt` (`julia/cpe_total_demo.jl`).

## Still open

- the direction flag of Arg₀ under a component bound is conservative (⟦±⟧ where the bound is tiny);
- the log-domain stage 1 has a 2π ambiguity for imaginary exponents and is used for real ones only;
- noise on the physical data (the Julia twin: 3 % multiplicative noise → α to 2e−3, r exact);
- more than one hidden unit from one aggregate observable without term-wise ratios is still not tested
  (the CPE recovers its single unit plus a constant from the aggregate; the RLC still uses term ratios).
