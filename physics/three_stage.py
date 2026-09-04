#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""the three-stage rule (the user's, with the least-squares step in place of the oracle Δ):
    1. Adam runs; every 10 steps the per-sample gradient directions g = ∂ℓ_n/∂A + i∂ℓ_n/∂B are summarised per exponent
       by the von Mises resultant R; max R ≥ 0.3 (and Rayleigh-significant) marks a CANDIDATE checkpoint;
    2. at a candidate, one estimates-averaged fit from the current W (joint least squares: direction and length,
       computable without the truth) — and, when the exponents are known to be discrete, its rounding as a second candidate;
    3. W fixed → coefficients by least squares → the held-out error of each candidate; the smallest wins; Adam goes on
       (a later candidate may beat it) until a candidate reaches the floor.
Baselines: Adam alone; the fit tried every 10 steps without the gate.  Success = held-out relative loss < 1e-8 and
max|W − true| < 1e-3 (best unit permutation).      python physics/three_stage.py"""
import sys, math, time, os
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, os.environ.get('TOTAL_ARITH_CUDA', str(Path(__file__).resolve().parents[2] / 'total-arith-cuda')))
from complex_sigma_product_unit import ComplexSigmaProductUnit, Tot, masked_cmse, totalize_grads, usable
from csigma_fit import fit_moments, log_features, products, fit_coefficients
from gd_vs_moments import make, match_error
from decide_exponents import make_int
torch.set_default_dtype(torch.float64)

def make_wave(seed):
    g = torch.Generator().manual_seed(seed)
    k, w = 1.7, 2.89; n = 2000
    x = torch.empty(n).uniform_(-1.2, 1.2, generator=g); t = torch.empty(n).uniform_(-0.8, 0.8, generator=g)
    X = torch.stack([torch.exp(x), torch.exp(t)], 1)
    truth = ComplexSigmaProductUnit(2, 1, outputs=1); truth.set_exponent(torch.zeros(1, 2), torch.tensor([[k, -w]])); truth.set_coefficient(torch.tensor([[0.8 - 0.6j]]))
    xre, xim = Tot(X), Tot(torch.zeros_like(X))
    with torch.no_grad(): yre, yim = truth(xre, xim)
    return 2, 1, xre, xim, yre, yim, torch.complex(torch.zeros(1, 2), torch.tensor([[k, -w]])), torch.tensor([0.8 - 0.6j])

def grad_R(L, y, W, a, wgt):
    """per-exponent von Mises resultant of the per-sample gradient directions (the user's statistic)"""
    P = products(L, W); e = y - P @ a
    c = (torch.conj(e).unsqueeze(1) * (a.unsqueeze(0) * P)).unsqueeze(2) * L.unsqueeze(1)       # conj(e)·a·P·L  [N,U,F]
    g = -2 * wgt.unsqueeze(1).unsqueeze(2) * torch.conj(c)
    u = g / g.abs().clamp(min=1e-300); ok = torch.isfinite(u) & (g.abs() > 0)
    u = torch.where(ok, u, torch.zeros_like(u))
    return (u.sum(0) / ok.sum(0).clamp(min=1)).abs()

def heldout(Lv, yv, W, a):
    P = products(Lv, W); return ((P @ a - yv).abs() ** 2 / (yv.abs() ** 2).clamp(min=1e-300)).mean().item()

def run(F, U, xre, xim, yre, yim, mode, seed=0, steps=2000, lr=0.02, every=10, thr=0.3, lattice=False):
    g = torch.Generator().manual_seed(seed)
    layer = ComplexSigmaProductUnit(F, U, outputs=1)
    with torch.no_grad():
        for p in (layer.wrp, layer.wrn, layer.wip, layer.win): p.add_(0.1 * torch.randn(p.shape, generator=g))
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    L, m = log_features(xre, xim); y = torch.complex(yre.val.double(), yim.val.double()); m = m & usable(yre, yim); L, y = L[m], y[m]
    N = L.shape[0]; perm = torch.randperm(N, generator=g); tr, va = perm[:int(0.8 * N)], perm[int(0.8 * N):]
    Lt, yt, Lv, yv = L[tr], y[tr], L[va], y[va]
    wgt = 1 / (yt.abs() ** 2).clamp(min=1e-300); sig = 3.0 / math.sqrt(len(tr))
    best = (float('inf'), None, None, None); cands = 0; bestAdam = float('inf'); bestW_adam = None
    for t in range(steps):
        if mode != 'adam' and t % every == 0 and t > 0:
            A, B = layer.exponent(); W = torch.complex(A.detach(), B.detach()); a = fit_coefficients(products(Lt, W), yt, wgt)
            if mode == 'three':
                R = grad_R(Lt, yt, W, a, wgt); go = R.max().item() >= max(thr, sig)
            else:
                go = True
            if go:
                cands += 1
                trials = [fit_moments(Lt, yt, U, W.clone(), iters=30)[:2]]
                if lattice:
                    Wr = torch.complex(torch.round(trials[0][0].real), torch.round(trials[0][0].imag)); trials.append((Wr, fit_coefficients(products(Lt, Wr), yt, wgt)))
                for Wc, ac in trials:
                    hv = heldout(Lv, yv, Wc, ac)
                    if hv < best[0]: best = (hv, Wc, ac, t)
                if best[0] < 1e-10: return best[0], best[1], best[3], cands
        opt.zero_grad(set_to_none=True)
        pre, pim = layer(xre, xim)
        loss, _, _ = masked_cmse(pre, pim, yre, yim, relative=True)
        if loss is None or not torch.isfinite(loss): break
        if loss.item() < bestAdam:
            bestAdam = loss.item(); A, B = layer.exponent(); bestW_adam = torch.complex(A.detach(), B.detach())
        if bestAdam < 1e-10 and mode == 'adam': return bestAdam, bestW_adam, t, 0
        loss.backward(); totalize_grads(layer.parameters()); opt.step()
    if mode == 'adam' or best[1] is None: return bestAdam, bestW_adam, steps, cands
    return best[0], best[1], best[3], cands

def main():
    print("three-stage rule vs Adam vs the fit every 10 steps — success = held-out loss < 1e-8 and max|W−true| < 1e-3 (8 seeds)")
    cases = [('plane wave, amplitude and exponents together (F=2 U=1)', lambda s: make_wave(500 + s), False),
             ('F=1 U=3 log grid', lambda s: (1, 3) + make(300 + s, 1, 3, True), False), ('F=1 U=4 log grid', lambda s: (1, 4) + make(300 + s, 1, 4, True), False),
             ('F=2 U=2 complex box', lambda s: (2, 2) + make(300 + s, 2, 2, False), False), ('F=2 U=3 complex box', lambda s: (2, 3) + make(300 + s, 2, 3, False), False),
             ('series RLC from the total impedance (integer lattice)', lambda s: make_int(400 + s, 4, 3, True), True)]
    for name, mk, lattice in cases:
        print(name)
        for mode, label in (('adam', 'Adam alone'), ('every', 'fit every 10 steps (no gate)'), ('three', 'R ≥ 0.3 candidates → fit → held-out decides')):
            ok = 0; t0 = time.time(); when = []; nc = []
            for s in range(8):
                F, U, xre, xim, yre, yim, Wt, at = mk(s)[:8]
                Lh, W, t, c = run(F, U, xre, xim, yre, yim, mode, seed=s, lattice=lattice)
                e = match_error(W, Wt) if W is not None else float('inf'); succ = (Lh < 1e-8 and e < 1e-3); ok += succ
                succ and when.append(t); nc.append(c)
            print(f"   {label:46s} {ok}/8   decided at step median {sorted(when)[len(when)//2] if when else '—':>5}   candidates mean {sum(nc)/8:6.1f}   ({time.time()-t0:.0f} s)", flush=True)

if __name__ == '__main__':
    main()
