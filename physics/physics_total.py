#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""The three physics experiments on the complex Σ-Product Unit (`complex_sigma_product_unit.py`, on total-arith-cuda):
Experiment 1 (series RLC), Experiment 2 (free-particle plane wave) — the two of EXPERIMENT_RESULT.md — and
Experiment 3 (the constant-phase element, a fractional power of a complex variable), each run as before in
the ordinary range and then pushed through the boundaries the earlier experiments avoided: exact zeros
(Log₀ 0 = 0, Arg₀ 0 = 0), ε = MIN⟦≤⟧ (LOG_MIN, tagged), MAX saturation, and the branch cut of s^α, in the
same schedule (exponents first from ratios with the coefficient cancelled, freeze, then the coefficients),
in two arithmetics: IEEE (the same formulas on plain values) and total.

    python physics/physics_total.py [--device cuda] [--repo /path/to/total-arith-cuda] [--fast]
    (the arithmetic is total-arith-cuda ≥ v1.2.0: `--repo`, or the environment variable TOTAL_ARITH_CUDA,
     or a checkout next to this repository; the unit itself is ../complex_sigma_product_unit.py)
"""
import argparse, math, os, sys, time
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--device', default='cpu')
ap.add_argument('--repo', default=os.environ.get('TOTAL_ARITH_CUDA', str(Path(__file__).resolve().parents[2] / 'total-arith-cuda')))
ap.add_argument('--fast', action='store_true')
args = ap.parse_args()
sys.path.insert(0, args.repo); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cuda_total import Tot, GE, LE, SUNK, MAX, MIN                      # noqa: E402
from complex_sigma_product_unit import ComplexSigmaProductUnit, masked_cmse, totalize_grads, as_complex, usable, LOG_MIN, complex_div, _raw_for  # noqa: E402

torch.manual_seed(20260904)
torch.set_default_dtype(torch.float64)
dev = torch.device(args.device)
OUT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); OUT.append(s)

def tot(x, total=True):
    """the entry: total → totalized (inf → MAX⟦≥⟧, subnormal → MIN⟦≤⟧); IEEE → the raw float32 values, no flags"""
    x = torch.as_tensor(x, dtype=torch.float64, device=dev)
    if total:
        return Tot(x)
    return Tot(x.to(torch.float32), torch.zeros(x.shape, dtype=torch.uint8, device=dev))
def eps_tot(shape, sign=1.0):
    return Tot(torch.full(shape, sign * MIN, dtype=torch.float32, device=dev), torch.full(shape, LE, dtype=torch.uint8, device=dev))

def wrap(t):                                                             # (−π, π]
    return torch.remainder(t + math.pi, 2 * math.pi) - math.pi

# ---------------------------------------------------------------- the schedule
def ratios(Pre, Pim, total):
    """the target ratios P_n/P_0 as a (re, im) Tot pair — a flagged product gives a flagged ratio, and a flagged
    target is no number to be fitted (dropping the flags here put a 5.6e37⟦≥⟧ into a loss: measured, 6e74)."""
    if total:
        P0re = Tot(Pre.val[0:1].expand_as(Pre.val), Pre.flag[0:1].expand_as(Pre.flag))
        P0im = Tot(Pim.val[0:1].expand_as(Pim.val), Pim.flag[0:1].expand_as(Pim.flag))
        return complex_div(Pre, Pim, P0re, P0im)
    R = torch.complex(Pre.val.double(), Pim.val.double()); R = R / R[0:1]
    zf = torch.zeros_like(Pre.flag)
    return Tot(R.real.to(torch.float32), zf), Tot(R.imag.to(torch.float32), zf)

def stage1(layer, x_re, x_im, q, steps, lr, mode='ratio', probe=False):
    """exponents from ratios to the reference sample 0 (the coefficient cancels).  mode 'ratio': the complex MSE of
    P_n/P_0 against q_n (the loss of Experiments 1–2; periodic in the phase, so the wave's imaginary exponents have no
    2π ambiguity in it); mode 'log': the log domain of the unit, (U_n − U_0, V_n − V_0) against Log₀ q_n with the
    phase compared modulo 2π (linear in W: for real exponents over many decades, where the ratio loss is a 1e12-steep
    valley).  q: complex target ratios [N, U].  A flagged element (ε, saturation, unknown) is no number and is left out."""
    opt = torch.optim.Adam([layer.wrp, layer.wrn, layer.wip, layer.win], lr=lr)
    qre, qim = q
    qc = torch.complex(qre.val.double(), qim.val.double())
    lq = torch.log(qc.abs()); aq = torch.angle(qc)
    fl = ex = 0; used = 0; t0 = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        if mode == 'log':
            U, V = layer.logproducts(x_re, x_im)
            m = usable(U, V, qre, qim); m[0] = False                     # [N, U] element mask: a flagged element (model or target) is no number
            if int(m.sum()) == 0:
                return float('nan'), 0, int((~m).sum()), step
            U0 = U.val[0:1].expand_as(U.val); V0 = V.val[0:1].expand_as(V.val)
            dU = U.val[m].double() - U0[m].double(); dV = V.val[m].double() - V0[m].double()
            e = (dU - lq[m]) ** 2 + wrap(dV - aq[m]) ** 2
            idx = m
        else:
            Pre, Pim = layer.products(x_re, x_im)
            Rre, Rim = ratios(Pre, Pim, layer.total)
            m = usable(Rre, Rim, qre, qim); m[0] = False
            if int(m.sum()) == 0:
                return float('nan'), 0, int((~m).sum()), step
            e = (Rre.val[m].double() - qc.real[m]) ** 2 + (Rim.val[m].double() - qc.imag[m]) ** 2
            idx = m
        loss = e.mean()
        if probe:
            return loss.item(), int(m.sum()), int((~m).sum()), 0
        if not torch.isfinite(loss):
            return float('nan'), int(m.sum()), int((~m).sum()), step
        loss.backward()
        f, x_ = totalize_grads(layer.parameters()); fl += f; ex += x_
        opt.step()
        used = int(m.sum())
        if loss.item() < 1e-13:
            break
    return loss.item(), used, int((~m).sum()), step + 1

def stage2(layer, x_re, x_im, t_re, t_im, steps, lr, relative=False):
    """freeze the exponents, learn the complex coefficients on the full observable."""
    params = [layer.arp, layer.arn, layer.aip, layer.ain]
    opt = torch.optim.Adam(params, lr=lr)
    n_used = n_ex = 0; best = float('inf'); best_state = None; best_step = 0
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        yre, yim = layer(x_re, x_im)
        loss, n_used, n_ex = masked_cmse(yre, yim, t_re, t_im, relative=relative)
        if loss is None or not torch.isfinite(loss):
            return float('nan'), n_used, n_ex, step
        if loss.item() < best:                                           # Adam wanders once the gradient is at the float32
            best = loss.item(); best_step = step; best_state = [p.detach().clone() for p in params]   # floor: keep the best state
        if best < 1e-12:
            break
        loss.backward(); totalize_grads(layer.parameters()); opt.step()
    with torch.no_grad():
        for p, b in zip(params, best_state):
            p.copy_(b)
    return best, n_used, n_ex, best_step + 1

def freeze_exponents(layer):
    for p in (layer.wrp, layer.wrn, layer.wip, layer.win):
        p.requires_grad_(False)

def copy_exponents(dst, src, rows):
    with torch.no_grad():
        for name in ('wrp', 'wrn', 'wip', 'win'):
            getattr(dst, name)[:len(rows)] = getattr(src, name)[rows]

# ================================================================ Experiment 1: series RLC
def experiment1(total, boundary):
    say(f"\n== Experiment 1: series RLC  z = r + i·w·l − i/(w·c)   [{'total' if total else 'IEEE'} arithmetic]{'  + boundaries' if boundary else ''}")
    n = 1200 if args.fast else 3000
    lo, hi = math.log(0.5), math.log(2.0)
    x = torch.exp(torch.empty(n, 4).uniform_(lo, hi)).to(dev)            # [r, w, l, c]
    nb = 0
    if boundary:
        nb = n // 10
        x[:nb, 3] = 0.0                                                  # c = 0 exactly: no capacitor term (Log₀ 0 = 0)
        x[nb:2 * nb, 1] = 0.0                                            # w = 0 exactly: DC
    # the truth from the same normal form: P_R = r, P_L = w·l, P_C = exp(−Log₀ w − Log₀ c)
    truth = ComplexSigmaProductUnit(4, 3, outputs=1, total=total).to(dev)
    with torch.no_grad():
        Wt = torch.tensor([[1., 0, 0, 0], [0, 1, 1, 0], [0, -1, 0, -1]], device=dev)
        truth.wrp.copy_(_raw_for(1 + Wt.clamp(min=0), truth.eps)); truth.wrn.copy_(_raw_for(1 + (-Wt).clamp(min=0), truth.eps))
        truth.set_coefficient(torch.tensor([[1 + 0j, 0 + 1j, 0 - 1j]], device=dev))
    xt = tot(x, total)
    if boundary:                                                         # some ε inputs: c = ε (an almost-zero capacitor: an open circuit)
        fl = xt.flag.clone(); v = xt.val.clone()
        v[2 * nb:3 * nb, 3] = MIN; fl[2 * nb:3 * nb, 3] = LE if total else 0
        xt = Tot(v, fl)
    with torch.no_grad():
        Pre, Pim = truth.products(xt); zre, zim = truth(xt)
    P = as_complex(Pre, Pim)
    flagged = int((~usable(zre, zim)).sum())
    if boundary:
        say(f"   data: {nb} samples with c = 0, {nb} with w = 0, {nb} with c = ε; targets flagged (not numbers): {flagged}")
        i0 = 0; i1 = 2 * nb
        say(f"   the model's reading: c = 0 → P_C = exp(−Log₀ w − Log₀ 0) = 1/w = {P[i0, 2].real:.4f} (the term drops the absent factor)   c = ε → P_C = {P[i1, 2].real:.3e} flag {int(Pre.flag[i1, 2])} (⟦≥⟧: an open circuit)")
    q = ratios(Pre, Pim, total)                                          # term-wise ratios: the coefficients cancel
    layer = ComplexSigmaProductUnit(4, 3, total=total).to(dev)
    if boundary and total:                                               # the consistency of target and model at the truth
        with torch.no_grad():
            Lt, _, _, _ = stage1(truth, xt, None, q, 1, 0.0, 'ratio', probe=True)
        say(f"   stage-1 loss at the true exponents: {Lt:.2e} (0 ⟺ the model and the data agree on every usable sample)")
    L1, used, exc, st = stage1(layer, xt, None, q, 400 if args.fast else 1500, 0.05, 'ratio')
    A, B = layer.exponent()
    trueA = torch.tensor([[1., 0, 0, 0], [0, 1, 1, 0], [0, -1, 0, -1]], device=dev)
    say(f"   stage 1 (exponents from ratios): loss {L1:.2e} after {st} steps, samples used {used} excluded {exc};  max|A−true| = {(A - trueA).abs().max():.2e}  max|B| = {B.abs().max():.2e}")
    if not math.isfinite(L1):
        say("   → dead: NaN (log 0 = −inf in the ratio targets / products)"); return
    say("   learned A =", [[round(v, 5) for v in row] for row in A.detach().cpu().tolist()])
    final = ComplexSigmaProductUnit(4, 4, outputs=1, total=total).to(dev)      # 3 learned units + a constant dummy (W = 0)
    copy_exponents(final, layer, [0, 1, 2]); freeze_exponents(final)
    true_a = torch.tensor([[1 + 0j, 0 + 1j, 0 - 1j, 0 + 0j]], device=dev)
    inits = {'all_ones': torch.ones(1, 4, dtype=torch.complex128, device=dev), 'all_zero': torch.zeros(1, 4, dtype=torch.complex128, device=dev),
             'walsh_row_1': torch.tensor([[1, -1, 1, -1]], dtype=torch.complex128, device=dev)}
    for name, init in inits.items():
        final.set_coefficient(init)
        L2, used2, exc2, st2 = stage2(final, xt, None, zre, zim, 300 if args.fast else 1500, 0.035)
        a = torch.complex(*final.coefficient())
        say(f"   stage 2 init {name:12s}: loss {L2:.2e} after {st2} steps, used {used2} excluded {exc2};  max|a−(1,i,−i,0)| = {(a - true_a).abs().max():.2e}")

# ================================================================ Experiment 2: plane wave
def experiment2(total, boundary):
    say(f"\n== Experiment 2: ψ = A·exp(i(kx − ωt)) = A·X^(ik)·T^(−iω), X = e^x, T = e^t   [{'total' if total else 'IEEE'}]{'  + boundaries' if boundary else ''}")
    k, w = 1.7, 2.89; amp = 0.8 - 0.6j
    n = 1500 if args.fast else 3300
    x = torch.empty(n).uniform_(-1.2, 1.2); t = torch.empty(n).uniform_(-0.8, 0.8)
    nb = 0
    if boundary:
        nb = n // 10
        x[:nb] = 100.0; x[nb:2 * nb] = -100.0                            # X = e^{±100}: past float32's MAX / below its MIN
    X = torch.stack([torch.exp(x), torch.exp(t)], 1).to(dev)
    psi = amp * torch.exp(1j * (k * x - w * t)).to(dev)
    xt = tot(X, total)                                                   # total: the entry totalizes (inf → MAX⟦≥⟧, 3.7e-44 → MIN⟦≤⟧); IEEE: inf stays inf
    if boundary:
        say(f"   inputs: {nb} with X = e^100 → {float(xt.val[0, 0]):.3e} flag {int(xt.flag[0, 0])} (MAX⟦≥⟧), {nb} with X = e^−100 → {float(xt.val[nb, 0]):.3e} flag {int(xt.flag[nb, 0])} (ε⟦≤⟧)")
    layer = ComplexSigmaProductUnit(2, 1, total=total).to(dev)
    if boundary:                                                         # the reference sample must be an ordinary one
        perm = torch.cat([torch.arange(2 * nb, n), torch.arange(0, 2 * nb)]).to(dev)
        xt = Tot(xt.val[perm], xt.flag[perm]); psi = psi[perm]
    q = ratios(tot(psi.real.unsqueeze(1), total), tot(psi.imag.unsqueeze(1), total), total)   # ψ_n/ψ_0: the amplitude cancels
    L1, used, exc, st = stage1(layer, xt, None, q, 500 if args.fast else 2200, 0.025, 'ratio')
    A, B = layer.exponent()
    say(f"   stage 1: loss {L1:.2e} after {st} steps, used {used} excluded {exc};  A = {[round(v, 6) for v in A[0].tolist()]} (true 0,0)  B = {[round(v, 6) for v in B[0].tolist()]} (true {k}, {-w})")
    if not math.isfinite(L1):
        say("   → dead: NaN"); return
    final = ComplexSigmaProductUnit(2, 1, outputs=1, total=total).to(dev)
    copy_exponents(final, layer, [0]); freeze_exponents(final)
    L2, used2, exc2, st2 = stage2(final, xt, None, tot(psi.real, total), tot(psi.imag, total), 400 if args.fast else 1800, 0.04)
    a = torch.complex(*final.coefficient())[0, 0]
    say(f"   stage 2: loss {L2:.2e} after {st2} steps, used {used2} excluded {exc2};  amplitude = {a.item():.6f} (true {amp})")

# ================================================================ Experiment 3: the constant-phase element
def experiment3(total, boundary):
    say(f"\n== Experiment 3: CPE  z = r + (iω)^(−α), α = 0.8 — a fractional power of a complex variable   [{'total' if total else 'IEEE'}]{'  + boundaries' if boundary else ''}")
    alpha, r0 = 0.8, 0.002
    ws = torch.logspace(-2, 4, 40)
    if boundary:
        ws = torch.cat([ws, torch.tensor([0.0, 1e-42, 1e-45])])          # ω = 0 (Log₀ 0), ω below float32's MIN (ε, LOG_MIN tagged)
    ws = ws.to(dev)
    s_re = tot(torch.zeros_like(ws).unsqueeze(1), total); s_im = tot(ws.unsqueeze(1), total)
    truth = ComplexSigmaProductUnit(1, 2, outputs=1, total=total).to(dev)     # unit 0: (iω)^W, unit 1: the constant (W = 0) carrying r
    with torch.no_grad():
        truth.wrp.copy_(_raw_for(torch.tensor([[1.0], [1.0]], device=dev), truth.eps)); truth.wrn.copy_(_raw_for(torch.tensor([[1.0 + alpha], [1.0]], device=dev), truth.eps))
        truth.set_coefficient(torch.tensor([[1 + 0j, r0 + 0j]], device=dev))
    with torch.no_grad():
        Pre, Pim = truth.products(s_re, s_im); zre, zim = truth(s_re, s_im)
    P = as_complex(Pre, Pim)
    ph = torch.angle(P[:, 0]) / math.pi
    say(f"   the phase reads the exponent: arg P/π at ω = 1e-2, 1, 1e4 = {ph[0]:.4f}, {ph[20]:.4f}, {ph[39]:.4f} (−α/2 = {-alpha / 2})")
    if boundary:
        say(f"   ω = 0: Log₀(i·0) = 0 → P = {P[40, 0]:.3f} (the unit drops out; IEEE: {'NaN' if not total else '—'})   ω = 1e-42 → ε: P = {P[41, 0]:.3e} flag {int(Pre.flag[41, 0])} (|P| ≥ 1e30, phase {ph[41]:.4f}π kept)")
    q = ratios(Tot(Pre.val[:, :1], Pre.flag[:, :1]), Tot(Pim.val[:, :1], Pim.flag[:, :1]), total)
    layer = ComplexSigmaProductUnit(1, 1, total=total).to(dev)
    L1, used, exc, st = stage1(layer, s_re, s_im, q, 600 if args.fast else 2000, 0.05, 'log')
    A, B = layer.exponent()
    say(f"   stage 1: loss {L1:.2e} after {st} steps, used {used} excluded {exc};  W = {A[0, 0].item():+.6f}{B[0, 0].item():+.6f}i (true −0.8)")
    if not math.isfinite(L1):
        say("   → dead: NaN"); return
    final = ComplexSigmaProductUnit(1, 2, outputs=1, total=total).to(dev)
    copy_exponents(final, layer, [0]); freeze_exponents(final)
    final.set_coefficient(torch.tensor([[1 + 0j, 0 + 0j]], device=dev))  # the constant unit (r) starts at 0
    L2, used2, exc2, st2 = stage2(final, s_re, s_im, zre, zim, 800 if args.fast else 4000, 0.02, relative=True)
    a = torch.complex(*final.coefficient())[0]
    say(f"   stage 2: loss {L2:.2e} after {st2} steps, used {used2} excluded {exc2};  a = {a[0].item():.6f} (true 1)  r = {a[1].real.item():.6f} (true {r0})")

def branch_cut():
    say("\n== the branch cut of s^(−α) on the negative real axis of the Laplace plane, α = 0.8")
    a = torch.tensor([-1.0 + 0.0j, -1.0 - 0.0j], dtype=torch.complex128)
    say(f"   IEEE, the side in the sign of a zero: (−1+0i)^(−α) = {(a[0] ** -0.8).item():.6f}   (−1−0i)^(−α) = {(a[1] ** -0.8).item():.6f}")
    layer = ComplexSigmaProductUnit(1, 1).to(dev)
    with torch.no_grad():
        layer.wrn.copy_(_raw_for(torch.tensor([[1.8]], device=dev), layer.eps))
    re = tot(torch.tensor([[-1.0], [-1.0], [-1.0]]))
    im = Tot(torch.tensor([[0.0], [MIN], [-MIN]], dtype=torch.float32, device=dev), torch.tensor([[0], [LE], [LE]], dtype=torch.uint8, device=dev))
    Pre, Pim = layer.products(re, im)
    P = as_complex(Pre, Pim)
    for i, name in enumerate(('−1 (one zero, no sign)', '−1 + iε', '−1 − iε')):
        say(f"   total, the side as a value: ({name})^(−α) = {P[i, 0].item():.6f}  flags {int(Pre.flag[i, 0])}/{int(Pim.flag[i, 0])}  arg/π = {math.atan2(P[i, 0].imag, P[i, 0].real) / math.pi:+.4f}")
    say("   ε carries the side of the cut as a value — the four-channel (Cartesian) representation keeps im = ±MIN⟦≤⟧, where a polar one would round it away;")
    say("   the direction flag is conservative (⟦≥≤±⟧): a bound on one component turns the pair, and this rule does not yet see that the bound is 1e-38 of the other.")

say(f"complex Σ-Product Unit on total-arith-cuda — device {dev}, values float32 (MAX {MAX:.3e}, ε = MIN {MIN:.3e}, LOG_MIN {LOG_MIN:.2f})")
t0 = time.time()
for total in (False, True):
    experiment1(total, False)
for total in (False, True):
    experiment1(total, True)
for total in (False, True):
    experiment2(total, False)
for total in (False, True):
    experiment2(total, True)
for total in (False, True):
    experiment3(total, False)
for total in (False, True):
    experiment3(total, True)
branch_cut()
say(f"\n({time.time() - t0:.0f} s)")
out = Path(__file__).with_name('results'); out.mkdir(exist_ok=True)
(out / f'physics_total_{args.device}_result.txt').write_text('\n'.join(OUT), encoding='utf-8')
