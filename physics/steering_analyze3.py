#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""against the TRUE Adam continuation: the harvest trajectory is Adam, so its held-out loss at step t+K is the baseline
(the sweep's α = 0, ρ = 1, κ = ∞ branch keeps Adam's step LENGTH but takes the direction μ+π of the unit-vector
mean — that is already a steered step, not Adam).    python physics/steering_analyze3.py"""
import csv, math, statistics
from pathlib import Path
from collections import defaultdict
OUT = Path(__file__).with_name('results') / 'steering'
hv = {(r['task'], int(r['seed']), int(r['step'])): float(r['loss_va']) for r in csv.DictReader(open(OUT / 'harvest.csv'))}
by = defaultdict(list)
for r in csv.DictReader(open(OUT / 'sweep.csv')): by[(r['task'], int(r['seed']), int(r['step']))].append(r)
an = {}
for r in csv.DictReader(open(OUT / 'analysis.csv')): an[tuple(eval(r['key']))] = r
f = float; K = 30
def med(v): return statistics.median(v) if v else float('nan')
def block(sel, label):
    if not sel: return
    print(f"{label}: {len(sel)} checkpoints with Adam(t+K) > 1e-8")
    for name, g in (('μ+π direction, Adam length (α=0, ρ=1, κ=∞)', 'mu'), ('best deterministic α, ρ=1', 'alpha'), ('best ρ, α=0, κ=∞', 'rho'),
                    ('best deterministic (α, ρ)', 'ar'), ('best of all 275 (winner\'s curse)', 'all'), ('noise draws (α=0, ρ=1): median draw', 'noise')):
        gains = [x[g] for x in sel]
        print(f"   {name:46s} gain over Adam: median {med(gains):6.2f}   >2× {sum(v > 2 for v in gains):4d}   <0.5× (worse) {sum(v < 0.5 for v in gains):4d}")
    nz = [x for x in sel if x['abest'] != 0 and x['aGN'] == x['aGN']]
    print(f"   sign of the best deterministic α = sign of the GN angle from μ+π: {sum((x['abest'] > 0) == (x['aGN'] > 0) for x in nz)}/{len(nz)}")
rows = []
for key, rs in by.items():
    adam = hv.get((key[0], key[1], key[2] + K))
    if adam is None or not (adam > 1e-8 and math.isfinite(adam)): continue
    L = lambda cond: [f(r['LK']) for r in rs if cond(r)]
    mu = L(lambda r: int(r['alpha_deg']) == 0 and f(r['rho']) == 1.0 and r['kappa_s'] == 'inf')[0]
    alpha = min(L(lambda r: f(r['rho']) == 1.0 and r['kappa_s'] == 'inf'))
    rho = min(L(lambda r: int(r['alpha_deg']) == 0 and r['kappa_s'] == 'inf'))
    ar = min(L(lambda r: r['kappa_s'] == 'inf'))
    allb = min(L(lambda r: True))
    noise = statistics.median(L(lambda r: int(r['alpha_deg']) == 0 and f(r['rho']) == 1.0 and r['kappa_s'] != 'inf'))
    det = {int(r['alpha_deg']): f(r['LK']) for r in rs if r['kappa_s'] == 'inf' and f(r['rho']) == 1.0}
    rows.append(dict(key=key, mu=adam / mu, alpha=adam / alpha, rho=adam / rho, ar=adam / ar, all=adam / allb, noise=adam / noise,
                     abest=min(det, key=det.get), aGN=f(an[key]['aGN']) if key in an else float('nan'), R=f(an[key]['R']) if key in an else float('nan')))
block(rows, 'ALL')
for lo, hi in ((0, 0.3), (0.3, 0.6), (0.6, 1.01)): block([x for x in rows if lo <= x['R'] < hi], f'max R in [{lo}, {hi})')
for t in ('wave', 'grid3', 'box2'): block([x for x in rows if x['key'][0] == t], t)
