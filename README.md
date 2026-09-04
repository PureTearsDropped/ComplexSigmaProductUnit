# ComplexSigmaProductUnit

A **complex Σ-Product Unit** on total arithmetic (`complex_sigma_product_unit.py`, a `torch.nn.Module`
on the value + flag arithmetic of [total-arith-cuda](https://github.com/PureTearsDropped/total-arith-cuda)),
and three physics experiments that push it through the boundaries where IEEE arithmetic stops
(exact zeros, ε, overflow, the branch cut).

    y = Σᵢ aᵢ · exp( Σⱼ Wᵢⱼ · Log₀ xⱼ )        x ∈ ℂ,  W = (w⁺ᴿᵉ − w⁻ᴿᵉ) + i(w⁺ᴵᵐ − w⁻ᴵᵐ),  a ∈ ℂ

It is not the classical Product Unit (∏ xⱼ^{wⱼ}, real): the inputs and exponents are complex, the
logarithm is the *reserved-word* complex logarithm — **Log₀(0) = 0, Arg₀(0) = 0**: a zero has no scale
and no direction and drops out of the product — the smallest positive state ε is a tagged boundary
value (L₀(ε) = log MIN ⟦≥⟧, "at least this negative"), the amplitude saturates while the phase is
kept, and the sum with complex coefficients is part of the unit (a Σ-Π unit).  The design is
[`docs/complex_product_unit_finite_boundary_learning_v2.md`](docs/complex_product_unit_finite_boundary_learning_v2.md)
(Japanese); the implementation is `complex_sigma_product_unit.py` here — `ComplexSigmaProductUnit`, a
`torch.nn.Module` for any complex input (a complex tensor, a (re, im) pair, or `Tot`s with flags), CPU or
CUDA, autograd through the values — on the `Tot` arithmetic of total-arith-cuda ≥ v1.2.0, which also holds
the polar Julia twin (`julia/ScalarTotComplex.jl`).

## The experiments

| | model | learned | at the boundaries |
|---|---|---|---|
| 1 series RLC | z = r + i·w·l − i/(w·c): three units over [r, w, l, c] | exponents (1,0,0,0), (0,1,1,0), (0,−1,0,−1) and coefficients (1, i, −i, 0) | c = 0 (the term drops out), w = 0, c = ε (an open circuit ⟦≥⟧) |
| 2 plane wave | ψ = A·X^{ik}·T^{−iω}, X = eˣ, T = eᵗ: imaginary exponents | (k, −ω) = (1.7, −2.89) and A = 0.8 − 0.6i | X = e^{±100}: float32 overflow / underflow |
| 3 constant-phase element | z = r + (iω)^{−α}: a fractional power of a **complex** variable | α = 0.8, r = 0.002; the constant phase −απ/2 *is* the exponent | ω = 0 (Log₀ 0), ω below MIN (ε), the two sides of the branch cut of s^α |

Each experiment runs the same staged schedule — exponents first from ratios to a reference sample
(the coefficients cancel), freeze, then the complex coefficients — twice: on IEEE (the same
formulas on plain values) and on total arithmetic.  In the ordinary range both reach the float32
floor (loss ≈ 1e−13); at every boundary set **IEEE is NaN at step 0** while the total layer
leaves the flagged samples out and recovers the same numbers (RTX 5090: 201 s; CPU: 272 s).
Details, the §32 specification table and what was found on the way:
[`physics/EXPERIMENT_RESULT_TOTAL.md`](physics/EXPERIMENT_RESULT_TOTAL.md); the original two
experiments on IEEE: [`physics/EXPERIMENT_RESULT.md`](physics/EXPERIMENT_RESULT.md).

## Run

```bash
git clone https://github.com/PureTearsDropped/total-arith-cuda   # ≥ v1.2.0 (the arithmetic), next to this repository
git clone https://github.com/PureTearsDropped/ComplexSigmaProductUnit
cd ComplexSigmaProductUnit && pip install -r requirements.txt
python test_complex_sigma_product_unit.py       # the reserved words of the unit
python physics/physics_total.py                 # the three experiments, IEEE vs total, CPU (~5 min)
python physics/physics_total.py --device cuda   # the same on the GPU
python physics/physics_total.py --fast          # a shorter pass (~1 min)
python physics/rlc_physics_demo.py              # the original experiments 1–2 (IEEE, float64)
python physics/schrodinger_plane_wave_demo.py
julia julia/cpe_total_demo.jl                   # the CPE on the polar Julia twin, hand-written gradients
```

`--repo /path/to/total-arith-cuda` or the environment variable `TOTAL_ARITH_CUDA` point to the
arithmetic if it is not next to this repository.  Results as run are in `physics/results/`.

## Learning: gradients averaged, or estimates averaged?

`csigma_fit.py` learns the unit without gradient steps — coefficients by least squares, exponents by the
least-squares average of the per-sample corrections (variable projection), a matrix-pencil (Prony) start on a
log-spaced sweep, and a von Mises centre of the per-sample corrections as the robust alternative to the mean.
On the aggregate problem (U units from one observable) the pencil start is exact up to U = 4 where Adam
falls to 2/8; from a random start no average finds the units; the pencil breaks under 0.1 % noise; the von
Mises centre is 6× closer under 15 % outliers.  Numbers and the caveats: [`physics/LEARNING.md`](physics/LEARNING.md).

## Files

- `complex_sigma_product_unit.py` — the unit: `log0` / `arg0` / `clog0` (the reserved-word logarithm), `product_unit`,
  `tot_exp` (amplitude saturation, phase kept), `complex_sum`, `masked_cmse` (a flagged sample is no number),
  `totalize_grads` (the gradient floor as arithmetic), `ComplexSigmaProductUnit`.  Test: `test_complex_sigma_product_unit.py`.
- `csigma_fit.py` — the estimates-averaged learner (least squares, Gauss–Newton in the log domain, the matrix pencil, the von Mises centre); `physics/gd_vs_moments.py`, `physics/outlier_test.py` compare it with Adam.
- `physics/physics_total.py` — the three experiments on `ComplexSigmaProductUnit`, IEEE and total, with the boundaries.
- `physics/rlc_physics_demo.py`, `physics/schrodinger_plane_wave_demo.py` — the original experiments (PyTorch, float64), `physics/README_experiments_1_2.md`.
- `physics/EXPERIMENT_RESULT.md`, `physics/EXPERIMENT_RESULT_TOTAL.md` — results and limitations.
- `julia/cpe_total_demo.jl` — the CPE on the polar `TotComplex`, hand-written gradients checked by finite differences, the branch-cut limitation of the polar form.
- `docs/complex_product_unit_finite_boundary_learning_v2.md` — the design (4 channels, Log₀, ε, gradient floor, saturation).

License: 0BSD.  Cite with `CITATION.cff` (please record the commit ID and the total-arith-cuda version).

⚠️ AI-assisted; verify. / 生成AI使用・要検証
