#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""Gradient-distribution-guided steering — the batch study of the research note (Stage A harvest, Stage B branching
sweep, and the falsification analysis that comes before a controller), on the Tot arithmetic, branches as a batch dim.

  harvest : Adam trajectories on three tasks × seeds; every 10 steps the state of each exponent weight from the
            per-sample gradients g_n = ∂ℓ_n/∂A + i∂ℓ_n/∂B (closed form): M₁ = R e^{iμ}, κ, M₂, M₃, |g| statistics,
            train/held-out loss, and the Adam moments (so a branch continues Adam exactly).
  sweep   : from every checkpoint (every 50 steps + the top-R events), 275 branches at once — α ∈ ±30° (11) ×
            ρ ∈ {0.5…1.5} (5) × κ_steer ∈ {5,10,20,50,∞} (5): the step of every exponent weight keeps Adam's length
            × ρ but takes the direction φ ~ VM(μ+π+α, κ_steer) (∞ = deterministic); the coefficients step with Adam.
            K = 30 steps; the score is the held-out loss at t+K (and the worst loss on the path).  α = 0, ρ = 1,
            κ = ∞ is plain Adam.
  analyze : does the sweep's optimum need a learned controller?  α* against (a) the angle of the joint least-squares
            (Gauss–Newton) direction from μ+π — computable without the truth — and (b) the circular moments M₂, M₃;
            sign accuracy and the gain of the best branch over Adam, by R bins.

    python physics/steering_sweep.py harvest|sweep|analyze [--device cuda] [--out physics/results/steering]
"""
import argparse, math, os, sys, time, json
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, os.environ.get('TOTAL_ARITH_CUDA', str(Path(__file__).resolve().parents[2] / 'total-arith-cuda')))
from complex_sigma_product_unit import Tot, clog0, usable, batched_products, batched_sum, as_complex
from csigma_fit import products, fit_coefficients

ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('--device', default='cpu'); ap.add_argument('--out', default=str(Path(__file__).with_name('results') / 'steering'))
ap.add_argument('--seeds', type=int, default=8); ap.add_argument('--steps', type=int, default=2000); ap.add_argument('--K', type=int, default=30)
args = ap.parse_args(); dev = torch.device(args.device); OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------- tasks (the same three as before)
def task(name, seed):
    g = torch.Generator().manual_seed(seed)
    if name == 'wave':                                                        # ψ = A X^{ik} T^{−iω}, X = e^x, T = e^t
        n = 2000; x = torch.empty(n).uniform_(-1.2, 1.2, generator=g); t = torch.empty(n).uniform_(-0.8, 0.8, generator=g)
        X = torch.complex(torch.stack([torch.exp(x), torch.exp(t)], 1), torch.zeros(n, 2)); Wt = torch.tensor([[1.7j, -2.89j]]); at = torch.tensor([0.8 - 0.6j])
    elif name == 'grid3':                                                     # F=1 U=3 on a log grid, complex exponents
        n = 40; X = torch.complex(torch.logspace(-2, 4, n), torch.zeros(n)).unsqueeze(1)
        Wt = torch.complex(torch.empty(3, 1).uniform_(-1.5, 1.5, generator=g), torch.empty(3, 1).uniform_(-2, 2, generator=g)); at = torch.complex(torch.empty(3).uniform_(-2, 2, generator=g), torch.empty(3).uniform_(-2, 2, generator=g))
    else:                                                                     # 'box2': F=2 U=2, complex inputs in a box
        n = 400; X = torch.complex(torch.empty(n, 2).uniform_(0.5, 2.0, generator=g), torch.empty(n, 2).uniform_(-1.0, 1.0, generator=g))
        Wt = torch.complex(torch.empty(2, 2).uniform_(-1.5, 1.5, generator=g), torch.empty(2, 2).uniform_(-2, 2, generator=g)); at = torch.complex(torch.empty(2).uniform_(-2, 2, generator=g), torch.empty(2).uniform_(-2, 2, generator=g))
    L = torch.log(X); y = products(L, Wt) @ at                                # the truth (double, exact model)
    perm = torch.randperm(X.shape[0], generator=g); ntr = int(0.8 * X.shape[0])
    return dict(name=name, seed=seed, X=X.to(dev), y=y.to(dev), tr=perm[:ntr].to(dev), va=perm[ntr:].to(dev), Wt=Wt, at=at, U=Wt.shape[0], F=Wt.shape[1])

# ---------------------------------------------------------------- the batched forward (Tot) and the closed-form gradients
class Batch:
    """B parameter sets on one task: W [B,U,F] complex, a [B,U] complex; forward through the Tot arithmetic."""
    def __init__(self, T, idx):
        self.T = T; self.idx = idx
        X = T['X'][idx]; self.u, self.v = clog0(Tot(X.real), Tot(X.imag)); self.L = torch.log(X); self.y = T['y'][idx]
        self.w = 1 / (self.y.abs() ** 2).clamp(min=1e-300)
    def forward(self, W, a):
        Pre, Pim = batched_products(self.u, self.v, W.real, W.imag)
        yre, yim = batched_sum(Pre, Pim, a.real, a.imag)
        ok = usable(yre, yim)                                                  # [B,N]: a flagged prediction is no number
        P = as_complex(Pre, Pim); yh = as_complex(yre, yim)
        e = torch.where(ok, self.y.unsqueeze(0) - yh, torch.zeros_like(yh))
        return P, e, ok
    def loss(self, W, a):
        P, e, ok = self.forward(W, a)
        return ((self.w * e.abs() ** 2).sum(1) / ok.sum(1).clamp(min=1)), ok.sum(1)
    def grads(self, W, a):
        """per-sample complex gradients of ℓ_n = w_n|e_n|²: gW [B,N,U,F] = −2w·e·conj(a·P·L), ga [B,N,U] = −2w·e·conj(P)"""
        P, e, ok = self.forward(W, a)
        aP = a.unsqueeze(1) * P                                                # [B,N,U]
        gW = -2 * (self.w * e).unsqueeze(2).unsqueeze(3) * torch.conj(aP.unsqueeze(3) * self.L.unsqueeze(0).unsqueeze(2))
        ga = -2 * (self.w * e).unsqueeze(2) * torch.conj(P)
        n = ok.sum(1).clamp(min=1)
        return gW, ga, gW.sum(1) / n.view(-1, 1, 1), ga.sum(1) / n.view(-1, 1), ok

def circ(gW):
    """circular moments of the per-sample gradient directions per weight: M1 (R, μ), κ, M2, M3, |g| median/mean"""
    mag = gW.abs(); nz = mag > 0
    unit = torch.where(nz, gW / mag.clamp(min=1e-300), torch.zeros_like(gW)); n = nz.sum(1).clamp(min=1)
    M1 = unit.sum(1) / n; M2 = (unit ** 2).sum(1) / n; M3 = (unit ** 3).sum(1) / n
    R = M1.abs(); mu = torch.angle(M1)
    kappa = torch.where(R < 0.53, 2 * R + R ** 3 + 5 * R ** 5 / 6, torch.where(R < 0.85, -0.4 + 1.39 * R + 0.43 / (1 - R).clamp(min=1e-9), 1 / (R ** 3 - 4 * R ** 2 + 3 * R).clamp(min=1e-9)))
    return dict(R=R, mu=mu, kappa=kappa, M2=M2, M3=M3, gmed=torch.quantile(mag.flatten(1), 0.5, dim=1) if mag.dim() > 2 else mag.median(), gmean=mag.mean(1))

class Adam2:
    """Adam elementwise on re and im (as torch.optim.Adam would on two real tensors); state per batch row"""
    def __init__(self, shapes, lr=0.02, b1=0.9, b2=0.999):
        self.lr, self.b1, self.b2 = lr, b1, b2; self.t = 0
        self.m = [torch.zeros(s, dtype=torch.complex128, device=dev) for s in shapes]
        self.v = [torch.zeros(s, dtype=torch.complex128, device=dev) for s in shapes]        # re: v of re, im: v of im
    def steps(self, grads):
        self.t += 1; out = []
        for i, g in enumerate(grads):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * torch.complex(g.real ** 2, g.imag ** 2)
            mh = self.m[i] / (1 - self.b1 ** self.t); vh = self.v[i] / (1 - self.b2 ** self.t)
            out.append(self.lr * torch.complex(mh.real / (vh.real.sqrt() + 1e-12), mh.imag / (vh.imag.sqrt() + 1e-12)))
        return out
    def clone_rows(self, rows):
        o = Adam2.__new__(Adam2); o.lr, o.b1, o.b2, o.t = self.lr, self.b1, self.b2, self.t
        o.m = [m[rows] for m in self.m]; o.v = [v[rows] for v in self.v]; return o

# ---------------------------------------------------------------- Stage A
def harvest():
    rows = []; ck = {}
    for name in ('wave', 'grid3', 'box2'):
        for s in range(args.seeds):
            T = task(name, 1000 + s); U, F = T['U'], T['F']; tr, va = Batch(T, T['tr']), Batch(T, T['va'])
            g = torch.Generator().manual_seed(s)
            W = (0.1 * torch.complex(torch.randn(1, U, F, generator=g), torch.randn(1, U, F, generator=g))).to(dev)   # W ≈ 0, symmetry broken
            a = torch.ones(1, U, dtype=torch.complex128, device=dev)
            opt = Adam2([W.shape, a.shape]); prev = None; t0 = time.time()
            for t in range(args.steps + 1):
                gW, ga, mW, ma, ok = tr.grads(W, a)
                if t % 10 == 0:
                    c = circ(gW); Ltr, _ = tr.loss(W, a); Lva, _ = va.loss(W, a)
                    key = f"{name}/{s}/{t}"
                    ck[key] = dict(W=W[0].cpu(), a=a[0].cpu(), m=[m[0].cpu() for m in opt.m], v=[v[0].cpu() for v in opt.v], t=opt.t)
                    for u in range(U):
                        for f in range(F):
                            rows.append(dict(task=name, seed=s, step=t, unit=u, feat=f, loss_tr=Ltr[0].item(), loss_va=Lva[0].item(),
                                             Werr=(W[0].cpu() - T['Wt']).abs().max().item(), R=c['R'][0, u, f].item(), mu=c['mu'][0, u, f].item(), kappa=c['kappa'][0, u, f].item(),
                                             M2re=c['M2'][0, u, f].real.item(), M2im=c['M2'][0, u, f].imag.item(), M3re=c['M3'][0, u, f].real.item(), M3im=c['M3'][0, u, f].imag.item(),
                                             gmean=c['gmean'][0, u, f].item(), gabs=mW[0, u, f].abs().item()))
                if t == args.steps: break
                dW, da = opt.steps([mW, ma]); W = W - dW; a = a - da
            print(f"harvest {name} seed {s}: final held-out {Lva[0].item():.2e}, |W−true| {(W[0].cpu() - T['Wt']).abs().max().item():.2e}  ({time.time()-t0:.0f} s)", flush=True)
    import csv
    with open(OUT / 'harvest.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    torch.save(ck, OUT / 'checkpoints.pt')
    print(f"{len(rows)} weight-checkpoint rows, {len(ck)} checkpoints → {OUT}")

# ---------------------------------------------------------------- Stage B
ALPHAS = [math.radians(d) for d in (-30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30)]
RHOS = [0.5, 0.75, 1.0, 1.25, 1.5]
KAPPAS = [5.0, 10.0, 20.0, 50.0, float('inf')]

def sweep():
    ck = torch.load(OUT / 'checkpoints.pt'); import csv
    grid = [(al, rh, ka) for al in ALPHAS for rh in RHOS for ka in KAPPAS]; Bn = len(grid)
    al_t = torch.tensor([g[0] for g in grid], device=dev); rh_t = torch.tensor([g[1] for g in grid], device=dev); ka_t = torch.tensor([g[2] for g in grid], device=dev)
    # which checkpoints: every 50 steps, plus the 3 highest-R events per trajectory
    hv = list(csv.DictReader(open(OUT / 'harvest.csv')))
    bykey = {}
    for r in hv: bykey.setdefault((r['task'], int(r['seed']), int(r['step'])), []).append(float(r['R']))
    chosen = set(k for k in bykey if k[2] % 50 == 0 and k[2] < args.steps)
    for (task_, seed) in set((k[0], k[1]) for k in bykey):
        ev = sorted([k for k in bykey if k[0] == task_ and k[1] == seed and k[2] < args.steps], key=lambda k: -max(bykey[k]))[:3]
        chosen.update(ev)
    chosen = sorted(chosen); print(f"{len(chosen)} checkpoints × {Bn} branches × K={args.K}", flush=True)
    out = []; tasks = {}; t0 = time.time(); gen = torch.Generator(device=dev).manual_seed(0)
    for i, (name, seed, step) in enumerate(chosen):
        if (name, seed) not in tasks: T = task(name, 1000 + seed); tasks[(name, seed)] = (T, Batch(T, T['tr']), Batch(T, T['va']))
        T, tr, va = tasks[(name, seed)]; U, F = T['U'], T['F']
        c0 = ck[f"{name}/{seed}/{step}"]
        W = c0['W'].to(dev).unsqueeze(0).expand(Bn, U, F).clone(); a = c0['a'].to(dev).unsqueeze(0).expand(Bn, U).clone()
        opt = Adam2([W.shape, a.shape]); opt.t = c0['t']
        opt.m = [c0['m'][0].to(dev).unsqueeze(0).expand(Bn, U, F).clone(), c0['m'][1].to(dev).unsqueeze(0).expand(Bn, U).clone()]
        opt.v = [c0['v'][0].to(dev).unsqueeze(0).expand(Bn, U, F).clone(), c0['v'][1].to(dev).unsqueeze(0).expand(Bn, U).clone()]
        worst = torch.zeros(Bn, device=dev); L0 = va.loss(W, a)[0][0].item()
        for k in range(args.K):
            gW, ga, mW, ma, ok = tr.grads(W, a)
            c = circ(gW); dW, da = opt.steps([mW, ma])
            # steering: the direction of every exponent step ← μ + π + α (+ von Mises noise), the length ← ρ·|Adam step|
            centre = c['mu'] + math.pi + al_t.view(-1, 1, 1)
            noise = torch.zeros_like(centre)
            fin = torch.isfinite(ka_t)
            if fin.any():
                vm = torch.distributions.VonMises(torch.zeros_like(centre[fin]), ka_t[fin].view(-1, 1, 1).expand_as(centre[fin]))
                noise[fin] = vm.sample()
            phi = centre + noise
            dW = rh_t.view(-1, 1, 1) * dW.abs() * torch.exp(1j * phi)
            W = W - dW; a = a - da
            Lk = va.loss(W, a)[0]; worst = torch.maximum(worst, torch.where(torch.isfinite(Lk), Lk, torch.full_like(Lk, float('inf'))))
        Lva, _ = va.loss(W, a)
        for b, (al, rh, ka) in enumerate(grid):
            out.append(dict(task=name, seed=seed, step=step, alpha_deg=round(math.degrees(al)), rho=rh, kappa_s=ka, L0=L0, LK=Lva[b].item(), worst=worst[b].item(),
                            Werr=(W[b].cpu() - T['Wt']).abs().max().item()))
        if i % 20 == 0: print(f"  {i}/{len(chosen)}  {name} seed {seed} step {step}: Adam {Lva[grid.index((0.0, 1.0, float('inf')))].item():.2e}  best {Lva.min().item():.2e}  ({time.time()-t0:.0f} s)", flush=True)
    with open(OUT / 'sweep.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0].keys())); wr.writeheader(); wr.writerows(out)
    print(f"{len(out)} branch results → {OUT/'sweep.csv'}  ({time.time()-t0:.0f} s)")

# ---------------------------------------------------------------- the analysis
def analyze():
    import csv, statistics
    sw = list(csv.DictReader(open(OUT / 'sweep.csv'))); hv = list(csv.DictReader(open(OUT / 'harvest.csv'))); ck = torch.load(OUT / 'checkpoints.pt')
    by = {}
    for r in sw: by.setdefault((r['task'], int(r['seed']), int(r['step'])), []).append(r)
    st = {}
    for r in hv: st.setdefault((r['task'], int(r['seed']), int(r['step'])), []).append(r)
    print(f"{len(by)} checkpoints; per checkpoint the best branch vs Adam (α=0, ρ=1, κ=∞), K steps ahead, held-out loss")
    rows = []
    for key, rs in by.items():
        adam = [r for r in rs if int(r['alpha_deg']) == 0 and float(r['rho']) == 1.0 and r['kappa_s'] == 'inf'][0]
        best = min(rs, key=lambda r: float(r['LK'])); bestdet = min([r for r in rs if r['kappa_s'] == 'inf' and float(r['rho']) == 1.0], key=lambda r: float(r['LK']))
        # the Gauss–Newton direction at the checkpoint, and its angle from μ+π for each weight (needs no truth)
        name, seed, step = key; T = task(name, 1000 + seed); c0 = ck[f"{name}/{seed}/{step}"]
        L = torch.log(T['X'][T['tr']]).cpu(); y = T['y'][T['tr']].cpu(); W = c0['W']; wgt = 1 / (y.abs() ** 2)
        P = products(L, W); a = fit_coefficients(P, y, wgt); e = y - P @ a
        J = ((a.unsqueeze(0) * P).unsqueeze(2) * L.unsqueeze(1)).reshape(len(y), -1); sw_ = wgt.sqrt()
        dGN = torch.linalg.lstsq(J * sw_.unsqueeze(1), (e * sw_).unsqueeze(1)).solution[:, 0].reshape(W.shape)
        mus = {(int(r['unit']), int(r['feat'])): float(r['mu']) for r in st[key]}; Rs = {(int(r['unit']), int(r['feat'])): float(r['R']) for r in st[key]}
        angs = []
        for (u, f), mu in mus.items():
            d = dGN[u, f]
            if d.abs() > 0: angs.append(((torch.angle(d).item() - (mu + math.pi)) + math.pi) % (2 * math.pi) - math.pi)
        aGN = statistics.median(angs) if angs else float('nan')
        rows.append(dict(key=key, R=max(Rs.values()), adam=float(adam['LK']), best=float(best['LK']), alpha_best=int(best['alpha_deg']), rho_best=float(best['rho']), kappa_best=best['kappa_s'],
                         alpha_det=int(bestdet['alpha_deg']), aGN=math.degrees(aGN), L0=float(adam['L0'])))
    gains = [r['adam'] / max(r['best'], 1e-300) for r in rows]
    print(f"gain of the best branch over Adam (ratio of held-out losses after K steps): median {statistics.median(gains):.2f}, > 2× in {sum(g > 2 for g in gains)}/{len(rows)}, > 10× in {sum(g > 10 for g in gains)}/{len(rows)}")
    det = [r for r in rows if r['alpha_det'] != 0]
    print(f"deterministic α (ρ=1, κ=∞): best α ≠ 0 at {len(det)}/{len(rows)} checkpoints; of those, sign(α*) = sign(angle of the GN direction from μ+π): {sum((r['alpha_det'] > 0) == (r['aGN'] > 0) for r in det if r['aGN'] == r['aGN'])}/{sum(1 for r in det if r['aGN'] == r['aGN'])}")
    for lo, hi in ((0, 0.3), (0.3, 0.6), (0.6, 1.01)):
        sel = [r for r in rows if lo <= r['R'] < hi]
        if sel:
            g2 = [r['adam'] / max(r['best'], 1e-300) for r in sel]
            print(f"  max R in [{lo}, {hi}): {len(sel)} checkpoints, median gain {statistics.median(g2):.2f}, best α ≠ 0 in {sum(r['alpha_det'] != 0 for r in sel)}, median |α*| {statistics.median(abs(r['alpha_det']) for r in sel)}°, stochastic (κ<∞) wins {sum(r['kappa_best'] != 'inf' for r in sel)}")
    with open(OUT / 'analysis.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

{'harvest': harvest, 'sweep': sweep, 'analyze': analyze}[args.stage]()
