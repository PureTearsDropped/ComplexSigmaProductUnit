#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""the aggregate problem — U units from ONE observable y = Σ a_u·exp(W_u·Log₀ x), exponents and coefficients
together — by gradient steps (Adam, full batch) and by estimates averaged (csigma_fit: least squares for a,
per-sample log-ratio averages for W, a matrix-pencil start on a log grid).  Random complex targets, 8 seeds each.
    python physics/gd_vs_moments.py"""
import sys, math, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
sys.path.insert(0, os.environ.get('TOTAL_ARITH_CUDA', str(Path(__file__).resolve().parents[2] / 'total-arith-cuda')))
from complex_sigma_product_unit import ComplexSigmaProductUnit, Tot, masked_cmse, totalize_grads, as_complex
from csigma_fit import fit, log_features, products
torch.set_default_dtype(torch.float64)

def make(seed, F, U, grid):
    g = torch.Generator().manual_seed(seed)
    A = torch.empty(U, F).uniform_(-1.5, 1.5, generator=g); B = torch.empty(U, F).uniform_(-2, 2, generator=g)
    a = torch.complex(torch.empty(1, U).uniform_(-2, 2, generator=g), torch.empty(1, U).uniform_(-2, 2, generator=g))
    if grid:                                                              # F = 1: a log-spaced sweep x = 10^{-2 … 4} (Δ = ln 10^6 / 39)
        n = 40; x = torch.complex(torch.logspace(-2, 4, n), torch.zeros(n)).unsqueeze(1); delta = math.log(1e6) / (n - 1)
    else:                                                                 # complex inputs in a box
        n = 400; x = torch.complex(torch.empty(n, F).uniform_(0.5, 2.0, generator=g), torch.empty(n, F).uniform_(-1.0, 1.0, generator=g)); delta = None
    truth = ComplexSigmaProductUnit(F, U, outputs=1); truth.set_exponent(A, B); truth.set_coefficient(a)
    xre, xim = Tot(x.real), Tot(x.imag)
    with torch.no_grad(): yre, yim = truth(xre, xim)
    return xre, xim, yre, yim, torch.complex(A, B), a[0], delta

def match_error(W, Wt):
    """max |W − true| over the best unit permutation"""
    import itertools
    U = W.shape[0]
    return min(max((W[list(p)] - Wt).abs().max().item() for _ in [0]) for p in itertools.permutations(range(U)))

def gd(F, U, xre, xim, yre, yim, steps=2000, lr=0.02, seed=0):
    g = torch.Generator().manual_seed(seed)
    layer = ComplexSigmaProductUnit(F, U, outputs=1)
    with torch.no_grad():
        for p in (layer.wrp, layer.wrn, layer.wip, layer.win): p.add_(0.1 * torch.randn(p.shape, generator=g))   # symmetry broken
    opt = torch.optim.Adam(layer.parameters(), lr=lr); best = float('inf'); bestW = None
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        pre, pim = layer(xre, xim)
        loss, _, _ = masked_cmse(pre, pim, yre, yim, relative=True)
        if loss is None or not torch.isfinite(loss): break
        if loss.item() < best:
            best = loss.item(); A, B = layer.exponent(); bestW = torch.complex(A.detach(), B.detach())
        if best < 1e-10: break
        loss.backward(); totalize_grads(layer.parameters()); opt.step()
    return best, bestW

def moments(F, U, xre, xim, yre, yim, init, delta, seed=0, center='lstsq'):
    layer = ComplexSigmaProductUnit(F, U, outputs=1)
    W, a, loss, it = fit(layer, xre, xim, yre, yim, U, init=init, delta=delta, seed=seed, center=center, iters=200 if center == 'vonmises' else 60)
    with torch.no_grad():
        pre, pim = layer(xre, xim)
        lt, _, _ = masked_cmse(pre, pim, yre, yim, relative=True)          # the loss of the *layer* (total arithmetic) at the fit
    return lt.item(), W, it

print("the aggregate problem: U units from one observable, W and a together — success = relative loss < 1e-8 and max|W−true| < 1e-3 (8 seeds)")
for F, U, grid in ((1, 2, True), (1, 3, True), (1, 4, True), (2, 2, False), (2, 3, False)):
    rows = []
    for method in (('Adam (gradients averaged)', 'gd'), ('lstsq mean, random starts', 'rand'), ('von Mises centre, random starts', 'vm')) + ((('lstsq mean, pencil start', 'pencil'), ('von Mises centre, pencil start', 'vmp')) if grid else ()):
        ok = 0; t0 = time.time(); losses = []
        for s in range(8):
            xre, xim, yre, yim, Wt, at, delta = make(300 + s, F, U, grid)
            if method[1] == 'gd':
                L, W = gd(F, U, xre, xim, yre, yim, seed=s)
            else:
                L, W, it = moments(F, U, xre, xim, yre, yim, 'pencil' if method[1] in ('pencil', 'vmp') else 'random', delta, seed=s, center='vonmises' if method[1] in ('vm', 'vmp') else 'lstsq')
            e = match_error(W, Wt) if W is not None else float('inf')
            ok += (L < 1e-8 and e < 1e-3); losses.append(L)
        losses.sort()
        rows.append(f"   {method[0]:28s} {ok}/8   median loss {losses[4]:.1e}   worst {losses[-1]:.1e}   ({time.time()-t0:.0f} s)")
    print(f"F={F} U={U} {'log grid (40 points)' if grid else 'complex box (400 points)'}"); print('\n'.join(rows), flush=True)
