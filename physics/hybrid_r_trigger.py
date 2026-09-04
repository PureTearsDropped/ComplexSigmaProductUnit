#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""the hybrid (the user's proposal): Adam explores until the per-sample corrections agree in direction — the von
Mises resultant length R of the per-sample exponent estimates rises — and at that moment ONE estimates-averaged
fit (csigma_fit.fit_moments: least squares for a, Gauss–Newton for W) finishes the job.  R is the condition of
the linearisation: the phases have stopped wrapping.  Control: the same fit attempted every K steps regardless
of R.  Aggregate problems where each method alone failed (Adam 4/8 … 1/8, the fit from random starts 0/8).
    python physics/hybrid_r_trigger.py"""
import sys, math, time, os
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, os.environ.get('TOTAL_ARITH_CUDA', str(Path(__file__).resolve().parents[2] / 'total-arith-cuda')))
from complex_sigma_product_unit import ComplexSigmaProductUnit, Tot, masked_cmse, totalize_grads, usable
from csigma_fit import fit_moments, log_features, products, fit_coefficients
torch.set_default_dtype(torch.float64)
from gd_vs_moments import make, match_error                             # the same problems and seeds

def coherence(L, y, W, a):
    """R per exponent coordinate: the resultant length of the directions of the per-sample corrections
    d_n = e_n / (a_u P_nu L_nf) — 1 when every sample asks for the same move, ~1/√N when they disagree."""
    P = products(L, W); e = y - P @ a
    J = (a.unsqueeze(0) * P).unsqueeze(2) * L.unsqueeze(1)
    d = e.unsqueeze(1).unsqueeze(2) / J
    u = d / d.abs().clamp(min=1e-300)
    ok = torch.isfinite(u)
    u = torch.where(ok, u, torch.zeros_like(u))
    return (u.sum(0) / ok.sum(0).clamp(min=1)).abs(), e

def hybrid(F, U, xre, xim, yre, yim, mode, seed=0, steps=2000, lr=0.02, every=10, thr=0.5):
    g = torch.Generator().manual_seed(seed)
    layer = ComplexSigmaProductUnit(F, U, outputs=1)
    with torch.no_grad():
        for p in (layer.wrp, layer.wrn, layer.wip, layer.win): p.add_(0.1 * torch.randn(p.shape, generator=g))
    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    L, m = log_features(xre, xim); y = torch.complex(yre.val.double(), yim.val.double()); m = m & usable(yre, yim); L, y = L[m], y[m]
    N = L.shape[0]; sig = 3.0 / math.sqrt(N)                              # Rayleigh: N R² > 3 ⟹ the agreement is not chance
    best = float('inf'); bestW = None; fired = 0; fired_at = None; Rmax = 0.0
    for t in range(steps):
        if t % every == 0 and t > 0:
            A, B = layer.exponent(); W = torch.complex(A.detach(), B.detach())
            a = fit_coefficients(products(L, W), y, 1 / (y.abs() ** 2).clamp(min=1e-300))
            R, e = coherence(L, y, W, a); Rm = R.mean().item(); Rmax = max(Rmax, Rm)
            go = (mode == 'control') or (mode == 'R' and Rm > thr and Rm > sig)
            if go:
                fired += 1
                Wf, af, loss, it = fit_moments(L, y, U, W.clone(), iters=30)
                if loss < 1e-10:
                    return loss, Wf, t, fired, Rm
                if mode == 'control' and fired > 40:
                    pass
        opt.zero_grad(set_to_none=True)
        pre, pim = layer(xre, xim)
        loss, _, _ = masked_cmse(pre, pim, yre, yim, relative=True)
        if loss is None or not torch.isfinite(loss): break
        if loss.item() < best:
            best = loss.item(); A, B = layer.exponent(); bestW = torch.complex(A.detach(), B.detach())
        if best < 1e-10: return best, bestW, t, fired, Rmax
        loss.backward(); totalize_grads(layer.parameters()); opt.step()
    return best, bestW, steps, fired, Rmax

def main():
    print("hybrid: Adam + one estimates-averaged fit when the per-sample corrections agree (R > 0.5 and Rayleigh-significant), every 10 steps checked;")
    print("control: the same fit attempted every 10 steps regardless of R.   success = loss < 1e-8 and max|W−true| < 1e-3 (8 seeds)")
    for F, U, grid in ((1, 3, True), (1, 4, True), (2, 2, False), (2, 3, False)):
        print(f"F={F} U={U} {'log grid' if grid else 'complex box'}")
        for mode in ('R', 'control'):
            ok = 0; t0 = time.time(); when = []; fires = []; Rs = []
            for s in range(8):
                xre, xim, yre, yim, Wt, at, delta = make(300 + s, F, U, grid)
                L, W, t, fired, Rm = hybrid(F, U, xre, xim, yre, yim, mode, seed=s)
                e = match_error(W, Wt) if W is not None else float('inf')
                succ = (L < 1e-8 and e < 1e-3); ok += succ
                when.append(t if succ else None); fires.append(fired); Rs.append(Rm)
            ws = [w for w in when if w is not None]
            print(f"   {'R-triggered fit':18s} {ok}/8   solved at Adam step median {sorted(ws)[len(ws)//2] if ws else '—'}   fits attempted mean {sum(fires)/8:.1f}   R at the fit / max R: median {sorted(Rs)[4]:.2f}   ({time.time()-t0:.0f} s)" if mode == 'R' else
                  f"   {'fit every 10 steps':18s} {ok}/8   solved at Adam step median {sorted(ws)[len(ws)//2] if ws else '—'}   fits attempted mean {sum(fires)/8:.1f}   ({time.time()-t0:.0f} s)", flush=True)


if __name__ == '__main__':
    main()
