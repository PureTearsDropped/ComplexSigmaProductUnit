#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""the comparison ChatGPT proposed: the plain stack (Log at every layer) vs the log-stream residual at depth 2 / 4 / 8,
the same parameter count, the same start (W ≈ 0, a = 0 or a = small), the gradient norm by depth, held-out complex
RMSE, and the 1000-block identity / gradient stability.      python physics/depth_compare.py"""
import sys, math, time, os
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, os.environ.get('TOTAL_ARITH_CUDA', str(Path(__file__).resolve().parents[2] / 'total-arith-cuda')))
from csigma_deep import DeepCSPU
torch.set_default_dtype(torch.float64)

def target(kind, n=400):
    w = torch.logspace(-1, 2, n); x = torch.complex(w, torch.zeros(n)).unsqueeze(1); L0, m = DeepCSPU.entry(x)
    depth = {1: 1, 2: 2, 3: 3}[kind]; truth = DeepCSPU(1, 1, depth)
    with torch.no_grad():
        truth.W[0].copy_(torch.tensor([[0.5 + 0j]])); truth.a[0].copy_(torch.tensor([[0.5j]]))                       # chirp phase 0.5 i √ω
        if depth >= 2: truth.W[1].copy_(torch.tensor([[-0.3 + 0j]])); truth.a[1].copy_(torch.tensor([[0.8 + 0j]]))    # amplitude law
        if depth >= 3: truth.W[2].copy_(torch.tensor([[0.2 + 0.4j]])); truth.a[2].copy_(torch.tensor([[-0.3 + 0.2j]])) # a complex twist
        truth.V.copy_(torch.tensor([[-0.8 + 0j]] if depth == 1 else [[0.6 + 0j]])); truth.c.copy_(torch.tensor([1.5 - 0.5j]))
        y = truth(L0)
    return L0, y

def nparams(m): return sum(p.numel() for p in m.parameters())

def train(model, L0, y, steps=3000, lr=0.02, seed=0):
    g = torch.Generator().manual_seed(seed); n = len(y); perm = torch.randperm(n, generator=g); tr, va = perm[:int(0.8 * n)], perm[int(0.8 * n):]
    w = 1 / (y.abs() ** 2); opt = torch.optim.Adam(model.parameters(), lr=lr); best = float('inf'); gnorm0 = None
    for t in range(steps):
        if t % 10 == 0: model.head_lstsq(L0[tr], y[tr], w[tr])
        opt.zero_grad(set_to_none=True)
        yh = model(L0[tr]); loss = (w[tr] * (yh - y[tr]).abs() ** 2).mean()
        if not torch.isfinite(loss): return float('nan'), gnorm0
        loss.backward()
        if t == 0: gnorm0 = [float(torch.sqrt(sum((p.grad.abs() ** 2).sum() for p in blk if p.grad is not None))) for blk in [(model.W[i], model.a[i]) for i in range(model.depth)]]
        with torch.no_grad(): lv = (w[va] * (model(L0[va]) - y[va]).abs() ** 2).mean().item()
        best = min(best, lv)
        if best < 1e-12: break
        opt.step()
    return best, gnorm0

print("plain stack (Log every layer) vs log-stream residual — same parameter count, W ≈ 0.1·noise, a as stated, Adam 3000 steps, 6 seeds, held-out relative MSE median [worst]")
for kind in (2, 3):
    L0, y = target(kind); print(f"target {kind}: {'chirp → amplitude law → head' if kind == 2 else 'chirp → amplitude law → complex twist → head'}")
    configs = [('stacked 2 layers, U=8', lambda s: DeepCSPU(1, 8, 2, embed='stacked', seed=s, a_init=0.05)),
               ('log-stream 2 blocks, U=8, a=0', lambda s: DeepCSPU(1, 8, 2, seed=s)),
               ('log-stream 2 blocks, U=8, a=0.05·noise', lambda s: DeepCSPU(1, 8, 2, seed=s, a_init=0.05)),
               ('log-stream 4 blocks, U=4, a=0', lambda s: DeepCSPU(1, 4, 4, seed=s)),
               ('log-stream 4 blocks, U=4, a=0.05·noise', lambda s: DeepCSPU(1, 4, 4, seed=s, a_init=0.05)),
               ('log-stream 8 blocks, U=2, a=0', lambda s: DeepCSPU(1, 2, 8, seed=s)),
               ('log-stream 8 blocks, U=2, a=0.05·noise', lambda s: DeepCSPU(1, 2, 8, seed=s, a_init=0.05))]
    for name, mk in configs:
        res = []; gn = None; t0 = time.time()
        for s in range(6):
            m = mk(s); b, g0 = train(m, L0, y, seed=s); res.append(b if b == b else float('inf')); gn = gn or g0
        res.sort(); npar = nparams(mk(0))
        print(f"   {name:40s} params {npar:3d}   {res[3]:.1e} [{res[-1]:.1e}]   grad norm per block at step 0: {['%.1e' % v for v in gn]}   ({time.time()-t0:.0f} s)", flush=True)

# 1000 blocks: forward identity and backward identity at the start; and with small a the growth of the gradient norm by depth
print("\n1000 blocks, F=2 U=2: the stream at the start and the gradient through the stack")
L0 = torch.complex(torch.randn(64, 2), torch.randn(64, 2))
for a0 in (0.0, 0.01, 0.05):
    m = DeepCSPU(2, 2, 1000, seed=0, a_init=a0)
    L, Ls = m.stream(L0)
    dev = (L - L0).abs().max().item()
    s = (L.abs() ** 2).sum(); s.backward()
    gfirst = m.a[0].grad.abs().max().item(); glast = m.a[-1].grad.abs().max().item()
    print(f"   a = {a0:<5}: max|L_1000 − L_0| = {dev:.2e}   ∂/∂a: first block {gfirst:.2e}, last block {glast:.2e}   (ratio {gfirst / max(glast, 1e-300):.2f}; 1 = identity backward)")
