# from the user's trajectory points (their seed, their data): does one estimates-averaged fit (Gauss–Newton, csigma_fit)
# finish the job where R was high but the exponent wrong (update 430: A_T = 2.09, R_T = 0.96)?
import sys, math, torch
from pathlib import Path; import os; sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, os.environ.get('TOTAL_ARITH_CUDA', str(Path(__file__).resolve().parents[2] / 'total-arith-cuda')))
from csigma_fit import fit_moments, products
torch.manual_seed(20260904); torch.set_default_dtype(torch.float64)
k = 1.7; omega = k * k; n = 4096
x = torch.empty(n).uniform_(-1.2, 1.2); t = torch.empty(n).uniform_(-0.8, 0.8)
inp = torch.stack([torch.exp(x), torch.exp(t)], 1); psi = (0.8 - 0.6j) * torch.exp(1j * (k * x - omega * t))
L = torch.log(inp[:3300]).to(torch.complex128); y = psi[:3300]
Wt = torch.tensor([[0 + 1.7j, 0 - 2.89j]])
for upd, A0, A1, B0, B1, R in ((100, -1.2428, 1.2276, -0.7592, 1.4212, 0.09), (300, -1.5494, 2.2247, 1.6272, 1.3036, 0.27), (400, -0.7639, 2.6046, 1.7929, -1.4057, 0.76), (430, -0.0607, 2.089, 1.7203, -2.7076, 0.96), (475, -0.0002, -0.0413, 1.6969, -2.8977, 0.92)):
    W0 = torch.tensor([[complex(A0, B0), complex(A1, B1)]])
    W, a, loss, it = fit_moments(L, y, 1, W0, iters=30)
    print(f"update {upd:3d} (R_T = {R:.2f}, |W−true| = {(W0 - Wt).abs().max():.3f}) → GN: {it:2d} iterations, loss {loss:.1e}, |W−true| = {(W - Wt).abs().max():.1e}, a = {a[0]:.6f}")
