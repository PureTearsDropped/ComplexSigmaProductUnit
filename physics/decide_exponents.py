#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""deciding the exponents (the user's proposal, the varpro-powersum-nn discipline): when the per-sample corrections
agree (R up), refine by one estimates-averaged fit, then ROUND the exponents to the admissible lattice (integers
here — a physical law's exponents), refit the coefficients linearly and accept the decision only if the residual
drops to the floor.  The lattice is a sieve, the verification decides.  Aggregate problems with integer exponents
and complex coefficients — including the series RLC from its total impedance alone, without term-wise ratios.
    python physics/decide_exponents.py"""
import sys, math, time, os
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, os.environ.get('TOTAL_ARITH_CUDA', str(Path(__file__).resolve().parents[2] / 'total-arith-cuda')))
from complex_sigma_product_unit import ComplexSigmaProductUnit, Tot, masked_cmse, totalize_grads, usable
from csigma_fit import fit_moments, log_features, products, fit_coefficients
from hybrid_r_trigger import coherence
from gd_vs_moments import match_error
torch.set_default_dtype(torch.float64)

def make_int(seed, F, U, rlc=False):
    g = torch.Generator().manual_seed(seed)
    if rlc:                                                                # z = r + i·w·l − i/(w·c) over [r, w, l, c]
        Wt = torch.tensor([[1., 0, 0, 0], [0, 1, 1, 0], [0, -1, 0, -1]]); at = torch.tensor([[1 + 0j, 0 + 1j, 0 - 1j]]); F, U = 4, 3
    else:
        Wt = torch.randint(-2, 3, (U, F), generator=g).double()
        at = torch.complex(torch.empty(1, U).uniform_(-2, 2, generator=g), torch.empty(1, U).uniform_(-2, 2, generator=g))
    n = 600; x = torch.exp(torch.empty(n, F).uniform_(math.log(0.5), math.log(2.0), generator=g))   # real positive inputs
    truth = ComplexSigmaProductUnit(F, U, outputs=1); truth.set_exponent(Wt, torch.zeros_like(Wt)); truth.set_coefficient(at)
    xre, xim = Tot(x), Tot(torch.zeros_like(x))
    with torch.no_grad(): yre, yim = truth(xre, xim)
    return F, U, xre, xim, yre, yim, torch.complex(Wt, torch.zeros_like(Wt)), at[0]

def verify_rounded(L, y, W, wgt):
    Wr = torch.complex(torch.round(W.real), torch.round(W.imag))
    P = products(L, Wr); a = fit_coefficients(P, y, wgt)
    loss = ((wgt * (P @ a - y).abs() ** 2).mean()).item()
    return Wr, a, loss

def run(F, U, xre, xim, yre, yim, mode, seed=0, steps=2000, lr=0.02, every=10, thr=0.5):
    """mode: 'adam' | 'gn' (R-triggered fit) | 'decide' (R-triggered fit + rounding + verification) |
    'round-always' (rounding + verification every 10 steps, no R)"""
    g = torch.Generator().manual_seed(seed)
    layer = ComplexSigmaProductUnit(F, U, outputs=1)
    with torch.no_grad():
        for p in (layer.wrp, layer.wrn, layer.wip, layer.win): p.add_(0.1 * torch.randn(p.shape, generator=g))
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    L, m = log_features(xre, xim); y = torch.complex(yre.val.double(), yim.val.double()); m = m & usable(yre, yim); L, y = L[m], y[m]
    wgt = 1 / (y.abs() ** 2).clamp(min=1e-300); N = L.shape[0]; sig = 3.0 / math.sqrt(N)
    best = float('inf'); bestW = None; tries = 0
    for t in range(steps):
        if mode != 'adam' and t % every == 0 and t > 0:
            A, B = layer.exponent(); W = torch.complex(A.detach(), B.detach())
            a = fit_coefficients(products(L, W), y, wgt)
            R, e = coherence(L, y, W, a); Rm = R.mean().item()
            if mode == 'round-always':
                tries += 1
                Wr, ar, lr_ = verify_rounded(L, y, W, wgt)
                if lr_ < 1e-10: return lr_, Wr, t, tries
            elif Rm > thr and Rm > sig:
                tries += 1
                Wf, af, loss, it = fit_moments(L, y, U, W.clone(), iters=30)
                if loss < 1e-10: return loss, Wf, t, tries
                if mode == 'decide':
                    Wr, ar, lr_ = verify_rounded(L, y, Wf, wgt)
                    if lr_ < 1e-10: return lr_, Wr, t, tries
        opt.zero_grad(set_to_none=True)
        pre, pim = layer(xre, xim)
        loss, _, _ = masked_cmse(pre, pim, yre, yim, relative=True)
        if loss is None or not torch.isfinite(loss): break
        if loss.item() < best:
            best = loss.item(); A, B = layer.exponent(); bestW = torch.complex(A.detach(), B.detach())
        if best < 1e-10: return best, bestW, t, tries
        loss.backward(); totalize_grads(layer.parameters()); opt.step()
    return best, bestW, steps, tries

def main():
    print("integer exponents, complex coefficients, aggregate observable, random start (all channels 1 ± 0.1); success = loss < 1e-8 and max|W−true| < 1e-3 (8 seeds)")
    for F, U, rlc in ((2, 2, False), (3, 3, False), (4, 3, True)):
        print(f"{'series RLC from the total impedance (F=4, U=3)' if rlc else f'F={F} U={U}'}")
        for mode in ('adam', 'gn', 'decide', 'round-always'):
            ok = 0; t0 = time.time(); when = []; tries = []
            for s in range(8):
                F_, U_, xre, xim, yre, yim, Wt, at = make_int(400 + s, F, U, rlc)
                L, W, t, tr = run(F_, U_, xre, xim, yre, yim, mode, seed=s)
                e = match_error(W, Wt) if W is not None else float('inf'); succ = (L < 1e-8 and e < 1e-3); ok += succ
                succ and when.append(t); tries.append(tr)
            label = {'adam': 'Adam alone', 'gn': 'Adam + R-triggered fit', 'decide': 'Adam + R-triggered fit + round + verify', 'round-always': 'Adam + round + verify every 10 steps'}[mode]
            print(f"   {label:42s} {ok}/8   solved at step median {sorted(when)[len(when)//2] if when else '—':>5}   attempts mean {sum(tries)/8:5.1f}   ({time.time()-t0:.0f} s)", flush=True)

if __name__ == '__main__':
    main()
