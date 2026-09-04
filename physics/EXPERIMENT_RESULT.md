# Complex Product Unit — Physics Experiments

## Purpose

Test the staged Complex Product Unit idea on physical equations that genuinely produce complex-valued quantities.

Common exponent parameterization:

\[
W=A+iB,
\qquad
A=w^{\mathrm{Re}+}-w^{\mathrm{Re}-},
\qquad
B=w^{\mathrm{Im}+}-w^{\mathrm{Im}-}.
\]

All four strictly-positive internal exponent channels start at exactly 1:

\[
w^{\mathrm{Re}+}=w^{\mathrm{Re}-}=w^{\mathrm{Im}+}=w^{\mathrm{Im}-}=1,
\]

so initially

\[
A=B=0,\qquad W=0,\qquad P=1.
\]

The training schedule is staged:

1. infer exponents first while eliminating the unknown coefficient by ratios;
2. freeze exponents;
3. learn the complex coefficient(s).

---

# Experiment 1 — Series RLC impedance

Choose reference scales

- \(R_0=10\,\Omega\),
- \(\omega_0=1000\,\mathrm{rad/s}\),
- \(L_0=10\,\mathrm{mH}\),
- \(C_0=100\,\mu\mathrm{F}\),
- \(Z_0=10\,\Omega\).

These satisfy

\[
R_0=\omega_0L_0=\frac{1}{\omega_0C_0}=Z_0.
\]

With dimensionless variables

\[
r=R/R_0,\quad w=\omega/\omega_0,\quad l=L/L_0,\quad c=C/C_0,
\]

the series-RLC impedance becomes

\[
\boxed{
z=\frac{Z}{Z_0}=r+i\,wl-i\,w^{-1}c^{-1}
}
\]

with three latent Product Units:

\[
P_R=r,
\qquad
P_L=wl,
\qquad
P_C=w^{-1}c^{-1}.
\]

True effective real exponents over features \([r,w,l,c]\):

\[
A_R=(1,0,0,0),
\]

\[
A_L=(0,1,1,0),
\]

\[
A_C=(0,-1,0,-1),
\]

and

\[
B_R=B_L=B_C=0.
\]

The unknown physical coefficients are

\[
\boxed{a=(1,+i,-i)}.
\]

## Phase 1 — exponent fitting

For each physical term, divide all samples by one reference sample. The constant complex coefficient cancels:

\[
\frac{Z_n}{Z_0}=\frac{P_n}{P_0}.
\]

This allows the exponents to be fitted before the coefficients.

Recovered effective exponents:

```text
A =
[[ 1.000000,  0.000000,  0.000000,  0.000000],
 [ 0.000000,  1.000000,  1.000000,  0.000000],
 [-0.000000, -1.000000,  0.000000, -1.000000]]

B =
[[0,0,0,0],
 [0,0,0,0],
 [0,0,0,0]]
```

Maximum exponent error reached approximately machine precision.

Held-out mean relative magnitude error:

\[
5.69\times10^{-17}.
\]

## Phase 2 — coefficient fitting

The exponent parameters are frozen exactly. A fourth constant dummy Product Unit \(P_D=1\) is added, so

\[
z=a_RP_R+a_LP_L+a_CP_C+a_DP_D.
\]

True coefficient vector:

\[
(1,+i,-i,0).
\]

Coefficient learning was tested from several initializations, including flattened 2D Walsh row tensors

\[
h_r\otimes h_s.
\]

The four 2x2 Walsh row tensors flatten to

```text
[+1,+1,+1,+1]
[+1,-1,+1,-1]
[+1,+1,-1,-1]
[+1,-1,-1,+1]
```

All initializations converged to the physical coefficients. Examples:

```text
Walsh row 1 -> loss < 1e-10 at step 391
Walsh row 2 -> loss < 1e-10 at step 348
Walsh row 3 -> loss < 1e-10 at step 554
all +1      -> loss < 1e-10 at step 1089
all 0       -> loss < 1e-10 at step 508
```

Walsh row 2 was notably faster than the all-ones initialization in this particular Adam run. This is an optimization observation, not yet a general claim.

Final learned coefficients were numerically

\[
\boxed{(1,+i,-i,0)}
\]

with errors around \(10^{-11}\) to \(10^{-10}\), depending on initialization.

---

# Experiment 2 — Free-particle Schrödinger plane wave

Use the dimensionless free-particle solution

\[
\psi(x,t)=A\exp(i(kx-\omega t)).
\]

Set

\[
\hbar=1,
\qquad
m=\frac12,
\]

so the free-particle dispersion relation is

\[
\omega=k^2.
\]

For the experiment,

\[
k=1.7,
\qquad
\omega=2.89,
\]

and the unknown complex amplitude is

\[
A=0.8-0.6i.
\]

Introduce strictly-positive Product Unit inputs

\[
X=e^x,
\qquad
T=e^t.
\]

Then the physical wave becomes exactly a complex Product Unit:

\[
\boxed{
\psi=A\,X^{ik}T^{-i\omega}
}
\]

so the true effective exponents are

\[
A_{\mathrm{eff}}=(0,0),
\]

\[
\boxed{
B_{\mathrm{eff}}=(k,-\omega)=(1.7,-2.89)
}.
\]

## Phase 1 — complex exponent fitting

Again, the unknown coefficient is removed by ratios:

\[
\frac{\psi_n}{\psi_0}
=
\frac{P_n}{P_0}.
\]

Starting from all four internal exponent channels equal to 1 gives

\[
W=0,\qquad P=1.
\]

After training, the recovered exponent was

```text
A learned = [~1.1e-16, 0]
B learned = [1.7000000000000002, -2.89]
```

so the model recovered the imaginary physical exponent essentially exactly.

The optimization trajectory was not monotone: the exponent error initially moved away from the target before converging sharply around several hundred Adam steps. This is useful evidence that the all-ones internal initialization does not trivially linearize the problem, but it still reached the correct branch in this dataset.

## Phase 2 — complex amplitude fitting

After freezing the exponent, only the complex coefficient was trained.

True:

\[
A=0.8-0.6i.
\]

Learned:

\[
\boxed{
\hat A=0.8-0.6000000000000001i
}.
\]

Held-out complex MSE:

\[
1.10\times10^{-31}.
\]

Held-out mean relative complex error:

\[
2.75\times10^{-16}.
\]

---

# What these experiments show

Within ordinary finite numerical ranges, these synthetic physics tests support the following claims:

1. Four positive internal exponent channels initialized at 1 do indeed give the exact neutral start
   \[
   W=0,\qquad P=1.
   \]

2. Unknown constant complex coefficients can be eliminated during exponent fitting by sample ratios when the physical term is multiplicatively separable.

3. The same parameterization can recover negative real exponents, as in the capacitor term
   \[
   w^{-1}c^{-1}.
   \]

4. It can also recover genuinely imaginary exponents, as in the Schrödinger plane wave
   \[
   X^{ik}T^{-i\omega}.
   \]

5. After exponent freezing, complex coefficients can be learned separately with no exponent drift.

6. Walsh row-tensor coefficient initialization can improve early optimization relative to all +1 in at least the RLC coefficient-fit example, although this needs repeated-seed and harder-problem testing before drawing a general conclusion.

---

# What is not tested yet

These experiments deliberately stay away from the custom total-arithmetic boundaries. They do not yet test:

- exact zero inputs with \(\Log_0(0)=0\),
- the \(\varepsilon\) state with \(\log_0(\varepsilon)=-\infty\),
- MAX saturation,
- branch-cut crossings in complex input space,
- totalized `arg/log/exp` state propagation,
- gradient-floor behavior in the physical models,
- noisy physical data,
- recovery of multiple hidden Product Units from only one aggregate observable without component-wise ratio information.

The natural next experiment is therefore the same physics models pushed deliberately through zero/epsilon/MAX and branch boundaries using `total-arith-cuda` semantics.
