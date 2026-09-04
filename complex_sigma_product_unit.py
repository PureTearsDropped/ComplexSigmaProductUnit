#!/usr/bin/env python3
# ⚠️ 生成AI使用・要検証 / AI-assisted; verify.
"""complex_sigma_product_unit — the complex Σ-Product Unit (ComplexSigmaProductUnit) on total arithmetic: the `Tot`
(value + flag) of total-arith-cuda ≥ v1.2.0, which must be importable (a checkout next to this repository, the
environment variable TOTAL_ARITH_CUDA, or on sys.path).  `product_unit` is one Π; the layer adds the Σ with complex
coefficients.

The design is 「複素 Product Unit と有限境界学習系 — 4チャネル完全展開版」, built on this repository's
arithmetic instead of IEEE:

    x = (x⁺ᴿᵉ − x⁻ᴿᵉ) + i(x⁺ᴵᵐ − x⁻ᴵᵐ)                 §2   four input channels → a + ib
    Log₀(x) = L₀(r) + i·Arg₀(a, b),  r = √(a² + b²)     §7–§10
    L₀(0) = 0,  Arg₀(0, 0) = 0                          §4, §6   — the reserved word: 0 is a value with no scale and no
                                                                    direction, log 0 = 0 and arg 0 = 0 by definition
    L₀(ε) = LOG_MIN, tagged                             §4       — ε = MIN⟦≤⟧ (the arithmetic's underflow, direction kept)
                                                                    maps to log MIN with the flag ⟦≥⟧: "at least this negative"
    W = (w⁺ᴿᵉ − w⁻ᴿᵉ) + i(w⁺ᴵᵐ − w⁻ᴵᵐ)                  §11
    U + iV = Σ_j W_j·Log₀(x_j),  P = e^U (cos V + i sin V)   §12–§16, §29 (the normal form: exp∘Log₀, not a power)
    |P| saturates, the phase is kept                    §26, §28.2 — e^U → MAX⟦≥⟧ / MIN⟦≤⟧, V untouched
    y = Σ_i a_i P_i                                     §19
    gradient floor                                      §22–§24  — not an optimizer step: the arithmetic returns ±MIN⟦≤⟧
                                                                    for a gradient that underflowed, sign kept (`totalize_grads`)
    surrogate at saturation                             §27      — a flagged ⟦≥⟧ / ⟦±⟧ contribution is not a number: the
                                                                    sample is left out of the loss (`masked_cmse`), a flagged
                                                                    parameter gradient is left out of the step (`totalize_grads`)

Every value is a `Tot` (float32 value, uint8 flag: GE = true ≥ shown, LE = true ≤ shown, SUNK = sign unknown — on a
component; on the pair (re, im) SUNK reads "direction unknown", the two-point case of a direction being ℝ's sign).
Autograd runs on the values; the flags decide what may enter a loss or a step.  `total=False` runs the same formulas
in plain IEEE (log 0 = −inf, overflow = inf) for the comparison.  Twin: `julia/ScalarTotComplex.jl` (polar, audited).
"""
import math
import os
import sys
from pathlib import Path
import torch
try:
    from cuda_total import Tot, GE, LE, SUNK, MAX, MIN, F32, _sat, tot_mul, tot_add, tot_div, _true_zero, _danger_zero
except ImportError:                                                  # total-arith-cuda: TOTAL_ARITH_CUDA or ../total-arith-cuda
    sys.path.insert(0, os.environ.get('TOTAL_ARITH_CUDA', str(Path(__file__).resolve().parent.parent / 'total-arith-cuda')))
    from cuda_total import Tot, GE, LE, SUNK, MAX, MIN, F32, _sat, tot_mul, tot_add, tot_div, _true_zero, _danger_zero

NOB = GE | LE
UNKNOWN = GE | LE | SUNK
LOG_MAX = math.log(MAX)
LOG_MIN = math.log(MIN)


def _dev(t):
    return t.device


def _where_flag(cond, value, flag):
    return torch.where(cond, torch.full_like(flag, value), flag)


# ---------------------------------------------------------------- Log₀ on the magnitude, Arg₀ on the pair
def log0(r: Tot) -> Tot:
    """L₀ of a magnitude (r ≥ 0).  L₀(0) = 0 exact and unflagged (the reserved word); ε = (MIN, ⟦≤⟧) → log MIN ⟦≥⟧
    (LOG_MIN, tagged); otherwise log r with ScalarTot's rule — a bound survives only where it is provable
    (⟦≥⟧ above 1 or ⟦≤⟧ below 1 → ⟦≥⟧ on |log|); sign unknown → unknown."""
    v = r.val.double()
    z = v == 0
    safe = torch.where(z, torch.ones_like(v), v)
    u = torch.where(z, torch.zeros_like(v), torch.log(safe))              # the log branch never sees a 0 (no NaN in backward)
    f = r.flag
    ge = (f & GE) > 0; le = (f & LE) > 0; sunk = (f & SUNK) > 0
    out = torch.zeros_like(f)
    out = _where_flag(sunk | (ge & le), UNKNOWN, out)
    prov = ~sunk & ~(ge & le) & ((ge & (v > 1)) | (le & (v < 1)))         # the two provable cells (ε is one of them)
    out = _where_flag(prov, GE, out)
    out = _where_flag(~sunk & ~(ge & le) & ((ge & (v <= 1)) | (le & (v >= 1))) & ~z, NOB | SUNK, out)
    out = _where_flag(z & ge, UNKNOWN, out)                                # a dangerous zero: anything
    out = _where_flag(z & ~ge, 0, out)                                     # a true zero: L₀(0) = 0
    val, sflag = _sat(u, _dev(v))
    return Tot(val, sflag | out)


def arg0(a: Tot, b: Tot) -> Tot:
    """Arg₀(a, b): atan2(b, a) ∈ (−π, π], and 0 at (0, 0).  The direction is claimed exact only when both components
    are exact, or the flagged one is the only nonzero one (its sign is known: the direction is 0 or π, ±π/2)."""
    av, bv = a.val.double(), b.val.double()
    z = (av == 0) & (bv == 0)
    a_s = torch.where(z, torch.ones_like(av), av)                          # atan2(0, 1) = 0 with a finite derivative
    v = torch.atan2(bv, a_s)
    fa, fb = a.flag, b.flag
    sunk = ((fa | fb) & SUNK) > 0
    bnd_a = (fa & NOB) > 0; bnd_b = (fb & NOB) > 0
    tz_a = _true_zero(a.val, fa); tz_b = _true_zero(b.val, fb)
    moves = (bnd_a & ~tz_b) | (bnd_b & ~tz_a)                              # a bound on one component turns the pair
    danger = _danger_zero(a.val, fa) | _danger_zero(b.val, fb)
    out = torch.zeros_like(fa)
    out = _where_flag(sunk | moves | danger, NOB | SUNK, out)
    out = _where_flag(z & ~danger, 0, out)                                 # Arg₀(0, 0) = 0: the reserved word
    val, sflag = _sat(v, _dev(av))
    return Tot(val, sflag | out)


def hypot0(a: Tot, b: Tot) -> Tot:
    """|a + ib| with the magnitude flags of the components (monotone in each |component|; SUNK does not touch |·|)."""
    av, bv = a.val.double(), b.val.double()
    sq = av * av + bv * bv
    z = sq == 0
    r = torch.where(z, torch.zeros_like(sq), torch.sqrt(torch.where(z, torch.ones_like(sq), sq)))
    fa, fb = a.flag & NOB, b.flag & NOB
    ge = ((fa | fb) & GE) > 0; le = ((fa | fb) & LE) > 0
    out = torch.zeros_like(fa)
    out = _where_flag(ge & ~le, GE, out)
    out = _where_flag(le & ~ge, LE, out)
    out = _where_flag(ge & le, NOB, out)
    val, sflag = _sat(r, _dev(av))
    return Tot(val, sflag | out)


def clog0(a: Tot, b: Tot):
    """Log₀(a + ib) = (L₀(r), Arg₀(a, b)) as two Tots."""
    return log0(hypot0(a, b)), arg0(a, b)


# ---------------------------------------------------------------- the amplitude and the phase
def tot_exp(U: Tot) -> Tot:
    """e^U with the amplitude saturated (§26): overflow → MAX⟦≥⟧, underflow → MIN⟦≤⟧, never 0 (e^U > 0).
    Flags: ScalarTot's exp rule — the sign of e^U is certain, a bound on U flips with the sign of U."""
    v = U.val.double()
    raw = torch.exp(v)
    val, sflag = _sat(raw, _dev(v))
    under = raw == 0                                                       # exp(−huge) = 0 in double: e^U > 0 ⟹ ε
    val = torch.where(under, torch.full_like(val, MIN), val)
    sflag = sflag | under.to(torch.uint8) * LE
    f = U.flag
    ge = (f & GE) > 0; le = (f & LE) > 0; sunk = (f & SUNK) > 0
    out = torch.zeros_like(f)
    out = _where_flag(sunk | (ge & le), NOB, out)
    pos = v > 0; zero = v == 0
    out = _where_flag(~sunk & ge & ~le & pos, GE, out); out = _where_flag(~sunk & ge & ~le & ~pos & ~zero, LE, out)
    out = _where_flag(~sunk & le & ~ge & pos, LE, out); out = _where_flag(~sunk & le & ~ge & ~pos & ~zero, GE, out)
    out = _where_flag(~sunk & ge & ~le & zero, NOB, out)                   # (0, ≥): anything;  (0, ≤): exactly 1
    return Tot(val, sflag | out)


def tot_cis(V: Tot):
    """(cos V, sin V): periodic — a flagged phase says nothing about the result."""
    v = V.val.double()
    fl = _where_flag(V.flag > 0, NOB | SUNK, torch.zeros_like(V.flag))
    c, cf = _sat(torch.cos(v), _dev(v)); s, sf = _sat(torch.sin(v), _dev(v))
    return Tot(c, cf | fl), Tot(s, sf | fl)


# ---------------------------------------------------------------- the Product Unit
def _bcast(x: Tot, shape):
    return Tot(x.val.expand(shape), x.flag.expand(shape))


def log_product_unit(a: Tot, b: Tot, A: torch.Tensor, B: torch.Tensor, total: bool = True):
    """(U, V) = Σ_f W·Log₀(x) as Tots [N, U] — the Product Unit before the exp (its own domain: a fit of exponents
    is linear here, and the phase difference is compared modulo 2π by the caller)."""
    N, Fn = a.val.shape
    Un = A.shape[0]
    if not total:
        lg = torch.log(torch.complex(a.val.double(), b.val.double()))
        W = torch.complex(A.double(), B.double())
        L = torch.einsum('nf,uf->nu', lg, W)
        zf = torch.zeros(N, Un, dtype=torch.uint8, device=a.val.device)
        return Tot(L.real.to(F32), zf), Tot(L.imag.to(F32), zf)
    u, v = clog0(a, b)
    shape = (N, Un, Fn)
    u3 = _bcast(Tot(u.val.unsqueeze(1), u.flag.unsqueeze(1)), shape)
    v3 = _bcast(Tot(v.val.unsqueeze(1), v.flag.unsqueeze(1)), shape)
    A3 = _bcast(Tot(A.unsqueeze(0)), shape); B3 = _bcast(Tot(B.unsqueeze(0)), shape)
    Au = tot_mul(A3, u3); Bv = tot_mul(B3, v3); Av = tot_mul(A3, v3); Bu = tot_mul(B3, u3)
    Ure = tot_add(Au, Tot(-Bv.val, Bv.flag))                                # A u − B v   (negation keeps the flag)
    Vim = tot_add(Av, Bu)                                                  # A v + B u
    U = Tot(Ure.val[:, :, 0], Ure.flag[:, :, 0]); V = Tot(Vim.val[:, :, 0], Vim.flag[:, :, 0])
    for j in range(1, Fn):                                                 # the sum over features, term by term
        U = tot_add(U, Tot(Ure.val[:, :, j], Ure.flag[:, :, j]))
        V = tot_add(V, Tot(Vim.val[:, :, j], Vim.flag[:, :, j]))
    return U, V


def product_unit(a: Tot, b: Tot, A: torch.Tensor, B: torch.Tensor, total: bool = True):
    """P[n, u] = exp(Σ_f W[u, f]·Log₀(x[n, f])) for x = a + ib (Tot [N, F]) and W = A + iB (real tensors [U, F]).
    Returns (Pre, Pim) as Tots [N, U].  `total=False`: the same formulas on plain IEEE values (the comparison)."""
    N, Fn = a.val.shape
    Un = A.shape[0]
    if not total:
        av, bv = a.val.double(), b.val.double()
        lg = torch.log(torch.complex(av, bv))                              # log 0 = −inf + i·0 … and NaN follows
        W = torch.complex(A.double(), B.double())
        P = torch.exp(torch.einsum('nf,uf->nu', lg, W))
        return Tot(P.real.to(F32), torch.zeros(N, Un, dtype=torch.uint8, device=av.device)), \
               Tot(P.imag.to(F32), torch.zeros(N, Un, dtype=torch.uint8, device=av.device))
    U, V = log_product_unit(a, b, A, B)
    E = tot_exp(U)
    c, s = tot_cis(V)
    return tot_mul(E, c), tot_mul(E, s)


def batched_products(u: Tot, v: Tot, A: torch.Tensor, B: torch.Tensor):
    """the products for a BATCH of exponent sets: u, v = Log₀(x) as Tots [N, F] (or [Bn, N, F]), A, B real [Bn, U, F]
    → (Pre, Pim) as Tots [Bn, N, U].  Every branch of a sweep is one row; the arithmetic broadcasts."""
    Bn, Un, Fn = A.shape
    N = u.val.shape[-2]
    def lift(t):                                                         # [N,F] or [Bn,N,F] → [Bn,N,1,F]
        val = t.val if t.val.dim() == 3 else t.val.unsqueeze(0)
        flag = t.flag if t.flag.dim() == 3 else t.flag.unsqueeze(0)
        return Tot(val.unsqueeze(2).expand(Bn, N, Un, Fn), flag.unsqueeze(2).expand(Bn, N, Un, Fn))
    u4, v4 = lift(u), lift(v)
    A4 = Tot(A.unsqueeze(1).expand(Bn, N, Un, Fn)); B4 = Tot(B.unsqueeze(1).expand(Bn, N, Un, Fn))
    Au = tot_mul(A4, u4); Bv = tot_mul(B4, v4); Av = tot_mul(A4, v4); Bu = tot_mul(B4, u4)
    Ure = tot_add(Au, Tot(-Bv.val, Bv.flag)); Vim = tot_add(Av, Bu)
    U = Tot(Ure.val[..., 0], Ure.flag[..., 0]); V = Tot(Vim.val[..., 0], Vim.flag[..., 0])
    for j in range(1, Fn):
        U = tot_add(U, Tot(Ure.val[..., j], Ure.flag[..., j])); V = tot_add(V, Tot(Vim.val[..., j], Vim.flag[..., j]))
    E = tot_exp(U); c, s = tot_cis(V)
    return tot_mul(E, c), tot_mul(E, s)


def batched_sum(Pre: Tot, Pim: Tot, are: torch.Tensor, aim: torch.Tensor):
    """y[b, n] = Σ_u (are + i·aim)[b, u]·P[b, n, u]  →  (yre, yim) as Tots [Bn, N]."""
    Bn, N, Un = Pre.val.shape
    ar = Tot(are.unsqueeze(1).expand(Bn, N, Un)); ai = Tot(aim.unsqueeze(1).expand(Bn, N, Un))
    t1 = tot_mul(ar, Pre); t2 = tot_mul(ai, Pim); t3 = tot_mul(ar, Pim); t4 = tot_mul(ai, Pre)
    yre = tot_add(t1, Tot(-t2.val, t2.flag)); yim = tot_add(t3, t4)
    Yre = Tot(yre.val[..., 0], yre.flag[..., 0]); Yim = Tot(yim.val[..., 0], yim.flag[..., 0])
    for u in range(1, Un):
        Yre = tot_add(Yre, Tot(yre.val[..., u], yre.flag[..., u])); Yim = tot_add(Yim, Tot(yim.val[..., u], yim.flag[..., u]))
    return Yre, Yim


def complex_div(are: Tot, aim: Tot, bre: Tot, bim: Tot):
    """(are + i·aim)/(bre + i·bim) as Tots, through the real operations (a/0 = 0 included)."""
    den = tot_add(tot_mul(bre, bre), tot_mul(bim, bim))
    nre = tot_add(tot_mul(are, bre), tot_mul(aim, bim))
    t = tot_mul(are, bim)
    nim = tot_add(tot_mul(aim, bre), Tot(-t.val, t.flag))
    return tot_div(nre, den), tot_div(nim, den)


def complex_sum(Pre: Tot, Pim: Tot, are: torch.Tensor, aim: torch.Tensor):
    """y[n] = Σ_u (are + i·aim)[u]·P[n, u]  →  (yre, yim) as Tots [N]."""
    N, Un = Pre.val.shape
    shape = (N, Un)
    ar = _bcast(Tot(are.unsqueeze(0)), shape); ai = _bcast(Tot(aim.unsqueeze(0)), shape)
    t1 = tot_mul(ar, Pre); t2 = tot_mul(ai, Pim); t3 = tot_mul(ar, Pim); t4 = tot_mul(ai, Pre)
    yre = tot_add(t1, Tot(-t2.val, t2.flag)); yim = tot_add(t3, t4)
    Yre = Tot(yre.val[:, 0], yre.flag[:, 0]); Yim = Tot(yim.val[:, 0], yim.flag[:, 0])
    for u in range(1, Un):
        Yre = tot_add(Yre, Tot(yre.val[:, u], yre.flag[:, u])); Yim = tot_add(Yim, Tot(yim.val[:, u], yim.flag[:, u]))
    return Yre, Yim


# ---------------------------------------------------------------- the flag-aware loss and the totalized step
def usable(*tots):
    """a sample may enter the loss iff every flag on it is 0 or ⟦≤⟧ alone (the floor: a number with a sign)."""
    m = None
    for t in tots:
        ok = (t.flag & ~LE) == 0
        m = ok if m is None else (m & ok)
    return m


def masked_cmse(yre: Tot, yim: Tot, tre: Tot, tim: Tot, relative: bool = False):
    """mean |ŷ − t|² (or |ŷ − t|²/|t|²) over the usable samples only — the others are not numbers to be fitted.
    Returns (loss, n_used, n_excluded).  Indexing (not a where-mask) keeps the excluded samples' backward at exactly 0."""
    m = usable(yre, yim, tre, tim)
    idx = m.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return None, 0, int((~m).sum())
    dre = yre.val[idx].double() - tre.val[idx].double(); dim = yim.val[idx].double() - tim.val[idx].double()
    e = dre * dre + dim * dim
    if relative:
        t2 = tre.val[idx].double() ** 2 + tim.val[idx].double() ** 2
        e = e / torch.where(t2 == 0, torch.ones_like(t2), t2)
    return e.mean(), int(idx.numel()), int((~m).sum())


def totalize_grads(params):
    """§22–§24 as arithmetic: each parameter gradient is totalized — an underflowed gradient is ±MIN⟦≤⟧ (the floor,
    sign kept, used as is), an overflowed or NaN-derived one is ⟦≥⟧ / ⟦±⟧ (not a number: that coordinate does not
    step).  Returns (floored, excluded) counts."""
    floored = excluded = 0
    for p in params:
        if p.grad is None:
            continue
        val, flag = _sat(p.grad.double(), p.grad.device)
        bad = (flag & (GE | SUNK)) > 0
        floored += int(((flag & LE) > 0).sum()); excluded += int(bad.sum())
        val = torch.where(bad, torch.zeros_like(val), val)
        p.grad.copy_(val.to(p.grad.dtype))
    return floored, excluded


# ---------------------------------------------------------------- the layer (general: any complex input, any width)
def _positive(raw, eps):
    return torch.nn.functional.softplus(raw) + eps


def _raw_for(value, eps):
    y = value - eps
    return y + torch.log(-torch.expm1(-y))


class ComplexSigmaProductUnit(torch.nn.Module):
    """A **complex Σ-Product Unit** on total arithmetic — not the classical Product Unit (∏ x_j^{w_j}, real) but
    y = Σ_i a_i·exp(Σ_j W_ij·Log₀ x_j): complex inputs, complex exponents, the reserved-word Log₀, and the sum with
    complex coefficients built in (a Σ-Π unit; the name is the user's).  x ∈ ℂ^F (per sample) → P ∈ ℂ^U,
    P_u = exp(Σ_f W_uf·Log₀(x_f)),  W = (w⁺ᴿᵉ − w⁻ᴿᵉ) + i(w⁺ᴵᵐ − w⁻ᴵᵐ) with four strictly positive channels
    (softplus + eps, all exactly 1 at start ⟹ W = 0, P = 1), and optionally the complex coefficients a ∈ ℂ^{O×U}
    of y = a·P (four channels each).  Inputs: a complex tensor, a (re, im) pair of real tensors, or a (re, im) pair of
    `Tot` (flags in — an ε or a saturated measurement enters as what it is).  Output: (re, im) as `Tot` [N, U] or
    [N, O]; `.complex()` of them is the IEEE view.  Works on CPU and CUDA (the values are float32 as in `Tot`)."""
    def __init__(self, features: int, units: int, outputs: int = 0, eps: float = 1e-9, total: bool = True):
        super().__init__()
        self.eps = eps; self.total = total
        r1 = _raw_for(torch.ones(units, features), eps)
        self.wrp = torch.nn.Parameter(r1.clone()); self.wrn = torch.nn.Parameter(r1.clone())
        self.wip = torch.nn.Parameter(r1.clone()); self.win = torch.nn.Parameter(r1.clone())
        self.outputs = outputs
        if outputs:
            b = _raw_for(torch.ones(outputs, units), eps)                 # a = 1 + 0i at start (channels 2,1,1,1 → 1)
            self.arp = torch.nn.Parameter(_raw_for(torch.full((outputs, units), 2.0), eps)); self.arn = torch.nn.Parameter(b.clone())
            self.aip = torch.nn.Parameter(b.clone()); self.ain = torch.nn.Parameter(b.clone())

    def exponent(self):
        e = self.eps
        return _positive(self.wrp, e) - _positive(self.wrn, e), _positive(self.wip, e) - _positive(self.win, e)

    def coefficient(self):
        e = self.eps
        return _positive(self.arp, e) - _positive(self.arn, e), _positive(self.aip, e) - _positive(self.ain, e)

    def set_exponent(self, A: torch.Tensor, B: torch.Tensor):
        """initialise W = A + iB (real [U, F] each) exactly, as differences of channels ≥ 1."""
        with torch.no_grad():
            for p, v in ((self.wrp, 1 + A.double().clamp(min=0)), (self.wrn, 1 + (-A.double()).clamp(min=0)),
                         (self.wip, 1 + B.double().clamp(min=0)), (self.win, 1 + (-B.double()).clamp(min=0))):
                p.copy_(_raw_for(v, self.eps))

    def set_coefficient(self, a: torch.Tensor):
        """initialise a (complex [O, U]) exactly, as differences of channels ≥ 1 (the Walsh-row initialisations of
        the RLC demo go through here)."""
        with torch.no_grad():
            re, im = a.real.double(), a.imag.double()
            for p, v in ((self.arp, 1 + re.clamp(min=0)), (self.arn, 1 + (-re).clamp(min=0)),
                         (self.aip, 1 + im.clamp(min=0)), (self.ain, 1 + (-im).clamp(min=0))):
                p.copy_(_raw_for(v, self.eps))

    @staticmethod
    def as_tots(x, x_im=None):
        if isinstance(x, Tot):
            if x_im is None:
                x_im = Tot(torch.zeros_like(x.val), torch.zeros_like(x.flag))
            return x, x_im
        if x_im is None:
            if torch.is_complex(x):
                return Tot(x.real.double()), Tot(x.imag.double())
            return Tot(x.double()), Tot(torch.zeros_like(x, dtype=torch.float64))
        return Tot(x.double()), Tot(x_im.double())

    def logproducts(self, x, x_im=None):
        a, b = self.as_tots(x, x_im)
        A, B = self.exponent()
        return log_product_unit(a, b, A, B, total=self.total)

    def products(self, x, x_im=None):
        a, b = self.as_tots(x, x_im)
        A, B = self.exponent()
        return product_unit(a, b, A, B, total=self.total)

    def forward(self, x, x_im=None):
        Pre, Pim = self.products(x, x_im)
        if not self.outputs:
            return Pre, Pim
        are, aim = self.coefficient()
        if self.outputs == 1:
            return complex_sum(Pre, Pim, are[0], aim[0])
        outs = [complex_sum(Pre, Pim, are[o], aim[o]) for o in range(self.outputs)]
        return Tot(torch.stack([o[0].val for o in outs], 1), torch.stack([o[0].flag for o in outs], 1)), \
               Tot(torch.stack([o[1].val for o in outs], 1), torch.stack([o[1].flag for o in outs], 1))




def as_complex(re: Tot, im: Tot):
    """the IEEE view of a (re, im) Tot pair (the flags are dropped: look at them first)."""
    return torch.complex(re.val.double(), im.val.double())


def self_test():
    """the reserved words and the boundaries, as numbers."""
    dev = torch.device('cpu')
    T = lambda x: Tot(torch.as_tensor(x, dtype=torch.float64, device=dev))
    eps = Tot(torch.full((1,), MIN), torch.full((1,), LE, dtype=torch.uint8))
    z = T([0.0]); o = T([1.0]); m1 = T([-1.0])
    u, v = clog0(z, z)
    print('Log₀(0) =', u.val.item(), u.flag.item(), '  Arg₀(0,0) =', v.val.item(), v.flag.item())
    u, v = clog0(eps, z)
    print('Log₀(ε) = LOG_MIN tagged:', u.val.item(), 'flag', u.flag.item(), '(GE = at least this negative)')
    u, v = clog0(m1, z)
    print('Log₀(−1) = iπ:', u.val.item(), v.val.item())
    A = torch.tensor([[-0.8]]); B = torch.tensor([[0.0]])
    for name, a, b in (('(i·1)', z, o), ('(i·0)', z, z), ('(i·ε)', z, eps), ('(−1+iε)', m1, eps), ('(−1−iε)', m1, Tot(-eps.val, eps.flag))):
        Pre, Pim = product_unit(Tot(a.val.reshape(1, 1), a.flag.reshape(1, 1)), Tot(b.val.reshape(1, 1), b.flag.reshape(1, 1)), A, B)
        ph = math.atan2(Pim.val.item(), Pre.val.item()) / math.pi
        print(f'{name}^(-0.8) = {Pre.val.item():+.6e} {Pim.val.item():+.6e}i  flags {Pre.flag.item()}/{Pim.flag.item()}  arg/π = {ph:+.4f}')
    layer = ComplexSigmaProductUnit(2, 3, outputs=1)
    x = torch.complex(torch.randn(5, 2), torch.randn(5, 2))
    yre, yim = layer(x)
    print('layer: x [5,2] complex → y', tuple(yre.val.shape), 'at start y = a·Σ P = ', as_complex(yre, yim)[0].item(), '(W = 0 ⟹ P = 1, a = 1 ⟹ y = 3)')
    (yre.val.sum() + yim.val.sum()).backward()
    print('autograd through Tot:', all(p.grad is not None for p in layer.parameters()))
    print('OK')


if __name__ == '__main__':
    self_test()
