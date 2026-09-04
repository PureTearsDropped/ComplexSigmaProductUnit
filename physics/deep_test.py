#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""two-layer targets that a single Σ-PU cannot represent, learned by the deep log-stream CSPU from the identity start,
against the 1 + i·y/s embedding and against a single wide Σ-PU.
  target 1 (chirp): x = ω on a log grid; block: Δ = i·c·ω^{0.5}  (a phase growing as √ω); head: y = b·(ω e^{i c √ω})^{−0.8}
  target 2 (two blocks): block 1 as above, block 2: Δ = d·ω^{−0.3} (an amplitude law), head V = 0.6
    python physics/deep_test.py"""
import sys, math, time, os
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, os.environ.get('TOTAL_ARITH_CUDA', str(Path(__file__).resolve().parents[2] / 'total-arith-cuda')))
from csigma_deep import DeepCSPU
torch.set_default_dtype(torch.float64)

def target(kind, n=400):
    w = torch.logspace(-1, 2, n); x = torch.complex(w, torch.zeros(n)).unsqueeze(1)
    L0, m = DeepCSPU.entry(x)
    truth = DeepCSPU(1, 1, 2 if kind == 2 else 1)
    with torch.no_grad():
        truth.W[0].copy_(torch.tensor([[0.5 + 0j]])); truth.a[0].copy_(torch.tensor([[0.5j]]))            # Δ₁ = 0.5 i √ω
        if kind == 2:
            truth.W[1].copy_(torch.tensor([[-0.3 + 0j]])); truth.a[1].copy_(torch.tensor([[0.8 + 0j]]))   # Δ₂ = 0.8 ω^{−0.3}
            truth.V.copy_(torch.tensor([[0.6 + 0j]])); truth.c.copy_(torch.tensor([1.5 - 0.5j]))
        else:
            truth.V.copy_(torch.tensor([[-0.8 + 0j]])); truth.c.copy_(torch.tensor([1.5 - 0.5j]))
        y = truth(L0)
    return L0, y, truth

def train(model, L0, y, steps=3000, lr=0.02, head_refit=False, seed=0):
    g = torch.Generator().manual_seed(seed); n = len(y); perm = torch.randperm(n, generator=g); tr, va = perm[:int(0.8 * n)], perm[int(0.8 * n):]
    w = 1 / (y.abs() ** 2); opt = torch.optim.Adam(model.parameters(), lr=lr); best = (float('inf'), None)
    for t in range(steps):
        if head_refit and t % 10 == 0: model.head_lstsq(L0[tr], y[tr], w[tr])
        opt.zero_grad(set_to_none=True)
        yh = model(L0[tr]); loss = (w[tr] * (yh - y[tr]).abs() ** 2).mean()
        if not torch.isfinite(loss): break
        with torch.no_grad():
            lv = (w[va] * (model(L0[va]) - y[va]).abs() ** 2).mean().item()
        if lv < best[0]: best = (lv, t)
        if lv < 1e-12: break
        loss.backward(); opt.step()
    return best

print("deep Σ-PU with the log-stream residual, identity start (a = 0), Adam lr 0.02, 3000 steps; success = held-out relative loss < 1e-8 (6 seeds)")
for kind in (1, 2):
    L0, y, truth = target(kind)
    print(f"target {kind}: {'chirp phase then a power law (1 block + head)' if kind == 1 else 'chirp, then an amplitude law, then the head (2 blocks)'}")
    configs = [('single Σ-PU, U=4 (no stream)', lambda s: DeepCSPU(1, 4, 0, head_units=4, seed=s), False),
               ('log-stream, 1 block U=1', lambda s: DeepCSPU(1, 1, 1, seed=s), False),
               ('log-stream, 1 block U=1 + head lstsq every 10', lambda s: DeepCSPU(1, 1, 1, seed=s), True),
               ('log-stream, 2 blocks U=2', lambda s: DeepCSPU(1, 2, 2, seed=s), False),
               ('log-stream, 2 blocks U=2 + head lstsq', lambda s: DeepCSPU(1, 2, 2, seed=s), True),
               ('log-stream, 3 blocks U=2 + head lstsq', lambda s: DeepCSPU(1, 2, 3, seed=s), True),
               ('1 + i·y/s embedding, 2 blocks U=2 (s=1)', lambda s: DeepCSPU(1, 2, 2, embed='oneplus', scale=1.0, seed=s), True),
               ('1 + i·y/s embedding, 2 blocks U=2 (s=4)', lambda s: DeepCSPU(1, 2, 2, embed='oneplus', scale=4.0, seed=s), True)]
    for name, mk, refit in configs:
        ok = 0; losses = []; steps = []; t0 = time.time()
        for s in range(6):
            lv, t = train(mk(s), L0, y, head_refit=refit, seed=s)
            losses.append(lv); ok += (lv < 1e-8); steps.append(t)
        losses.sort()
        print(f"   {name:50s} {ok}/6   held-out median {losses[3]:.1e}  worst {losses[-1]:.1e}   ({time.time()-t0:.0f} s)", flush=True)

# ---- final-stage-first: invert the head to get the stream the block must produce, fit the block on its increment
from csigma_fit import fit_moments, products, fit_coefficients
def unwrap(ph):
    d = torch.diff(ph); d = d - 2 * math.pi * torch.round(d / (2 * math.pi)); return torch.cat([ph[:1], ph[:1] + torch.cumsum(d, 0)])

def layerwise(L0, y, U_block=1, rounds=4):
    """y = c·exp(V·L₁), L₁ = L₀ + Δ, Δ = Σ a exp(W·L₀).  1) head on the identity stream (varpro); 2) invert the head
    along the sorted grid: L₁ = Log(y/c)/V with the phase unwrapped; 3) fit Δ = L₁ − L₀ as a Σ-PU by estimates averaged
    (a random start refined by Gauss–Newton, coefficients by least squares); 4) refit the head on the new stream; repeat."""
    n = len(y); g = torch.Generator().manual_seed(0); perm = torch.randperm(n, generator=g); tr, va = perm[:int(0.8 * n)], perm[int(0.8 * n):]
    w = 1 / (y.abs() ** 2)
    V, c = torch.zeros(1, 1, dtype=torch.complex128), torch.ones(1, dtype=torch.complex128); W = None; a = None
    Delta = torch.zeros_like(L0)
    for r in range(rounds):
        L1 = L0 + Delta
        Vn, cn, _, _ = fit_moments(L1[tr], y[tr], 1, V if r else torch.tensor([[-0.5 + 0j]]), iters=60)     # the head (varpro)
        V, c = Vn, cn
        ph = unwrap(torch.angle(y / c[0]))                                                                 # invert the head: L₁ from y (grid order)
        L1_target = torch.complex(torch.log((y / c[0]).abs()), ph) / V[0, 0]
        Dt = L1_target - L0[:, 0]                                                                           # what the block must add
        best = None
        for s in range(6):                                                                                  # the block: a Σ-PU on Δ, estimates averaged
            gg = torch.Generator().manual_seed(s)
            W0 = 0.3 * torch.complex(torch.randn(U_block, 1, generator=gg), torch.randn(U_block, 1, generator=gg))
            Wf, af, lf, _ = fit_moments(L0[tr], Dt[tr], U_block, W0, iters=60)
            if best is None or lf < best[2]: best = (Wf, af, lf)
        W, a, _ = best
        Delta = (products(L0, W) @ a).unsqueeze(1)
        L1 = L0 + Delta; P = torch.exp(L1 @ V.T); c = fit_coefficients(P[tr], y[tr], w[tr])
        lv = ((w[va] * (torch.exp(L1[va] @ V.T) @ c - y[va]).abs() ** 2).mean()).item()
    return lv, W, a, V, c

print("\nfinal-stage-first (invert the head, fit the block's increment by estimates averaged), target 1:")
L0, y, truth = target(1)
lv, W, a, V, c = layerwise(L0, y)
print(f"   held-out {lv:.1e}   block W = {W[0,0]:.6f} (true 0.5)  a = {a[0]:.6f} (true 0.5i)   head V = {V[0,0]:.6f} (true −0.8)  c = {c[0]:.6f} (true 1.5−0.5i)")

# ---- the depth collapses under the log: with a one-unit head, Log y (phase unwrapped along the grid) = Log c + V·L₀ + V·a·e^{W·L₀}
#      — linear in (Log c, V, V·a) given W: a variable projection over ONE complex unknown, sieved on a grid and refined.
def log_varpro(L0, y, grid=41):
    ph = unwrap(torch.angle(y)); Ly = torch.complex(torch.log(y.abs()), ph); l0 = L0[:, 0]
    def solve(W):
        A = torch.stack([torch.ones_like(l0), l0, torch.exp(W * l0)], 1)
        th = torch.linalg.lstsq(A, Ly.unsqueeze(1)).solution[:, 0]; r = Ly - A @ th
        return th, r, (r.abs() ** 2).mean().item()
    best = None
    for re in torch.linspace(-2, 2, grid):
        for im in torch.linspace(-2, 2, grid):
            W = complex(re.item(), im.item()); th, r, loss = solve(W)
            if best is None or loss < best[2]: best = (W, th, loss)
    W, th, loss = best
    for _ in range(30):                                                     # Gauss–Newton on the one unknown
        th, r, loss = solve(W); J = -(th[2] * l0 * torch.exp(W * l0))
        dW = (torch.linalg.lstsq(J.unsqueeze(1), r.unsqueeze(1)).solution[0, 0]).item()
        W2 = W + dW; th2, r2, loss2 = solve(W2)
        if loss2 < loss: W = W2
        else: break
    th, r, loss = solve(W)
    logc, V, Va = th; a = Va / V
    return W, V, a, torch.exp(logc), loss

print("\nthe depth collapses under the log (one-unit head): Log y = Log c + V·L₀ + V·a·e^{W·L₀} — varpro over W alone, target 1:")
W, V, a, c, loss = log_varpro(L0, y)
print(f"   log-domain residual {loss:.1e}   W = {W:.6f} (true 0.5)   V = {V:.6f} (true −0.8)   a = {a:.6f} (true 0.5i)   c = {c:.6f} (true 1.5−0.5i)")
L1 = L0 + (a * torch.exp(W * L0[:, 0])).unsqueeze(1); yh = c * torch.exp(V * L1[:, 0])
print(f"   reconstruction relative error {(((yh - y).abs() ** 2) / y.abs() ** 2).mean().item():.1e}")
