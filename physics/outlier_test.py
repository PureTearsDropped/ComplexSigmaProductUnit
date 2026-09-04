# gross outliers on the data: does the von Mises centre resist them where the least-squares mean does not?
import sys, math, torch
from pathlib import Path; import os; sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); sys.path.insert(0, os.environ.get('TOTAL_ARITH_CUDA', str(Path(__file__).resolve().parents[2] / 'total-arith-cuda')))
from csigma_fit import pencil_init, fit_moments, products
torch.set_default_dtype(torch.float64)
def perm_err(W, Wt):
    import itertools
    return min((W[list(p)] - Wt).abs().max().item() for p in itertools.permutations(range(W.shape[0])))
n = 40; L = torch.log(torch.logspace(-2, 4, n)).to(torch.complex128).unsqueeze(1); delta = math.log(1e6) / (n - 1)
for frac in (0.0, 0.05, 0.15):
    for start in ('pencil', 'truth'):
        res = {'lstsq': [], 'vonmises': []}
        for s in range(8):
            g = torch.Generator().manual_seed(500 + s)
            W = torch.complex(torch.empty(2, 1).uniform_(-1.5, 1.5, generator=g), torch.empty(2, 1).uniform_(-2, 2, generator=g))
            a = torch.complex(torch.empty(2).uniform_(-2, 2, generator=g), torch.empty(2).uniform_(-2, 2, generator=g))
            y = products(L, W) @ a
            y = y * (1 + 1e-3 * torch.randn(n, generator=g))                                  # 0.1 % noise everywhere
            k = int(frac * n)
            if k:
                idx = torch.randperm(n, generator=g)[:k]
                y[idx] = y[idx] * torch.complex(torch.empty(k).uniform_(2, 4, generator=g), torch.empty(k).uniform_(-2, 2, generator=g))   # gross outliers
            W0 = pencil_init(y, 2, delta, L[0, 0]) if start == 'pencil' else W.clone()
            for c in res:
                Wf, af, loss, it = fit_moments(L, y, 2, W0.clone(), iters=200, center=c)
                res[c].append(perm_err(Wf, W))
        line = '   '.join(f"{c}: median |W−true| {sorted(v)[4]:.1e}, worst {max(v):.1e}" for c, v in res.items())
        print(f"outliers {frac:4.0%} start {start:6s}:  {line}")
