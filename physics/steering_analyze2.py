#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""the sweep read without the winner's curse: deterministic branches only, the fraction of stochastic branches that
beat Adam (a null: noise that helps by luck beats it half the time), checkpoints at the float floor excluded, and
the consistency of the best α along a trajectory (what a controller could learn).    python physics/steering_analyze2.py"""
import csv, math, statistics, sys
from pathlib import Path
from collections import defaultdict
OUT = Path(__file__).with_name('results') / 'steering'
rows = list(csv.DictReader(open(OUT / 'sweep.csv')))
by = defaultdict(list)
for r in rows: by[(r['task'], int(r['seed']), int(r['step']))].append(r)
an = {tuple(eval(r['key'])) if r['key'].startswith('(') else r['key']: r for r in csv.DictReader(open(OUT / 'analysis.csv'))}
def f(x): return float(x)
res = defaultdict(list)
per_traj = defaultdict(list)
for key, rs in by.items():
    adam = f([r for r in rs if int(r['alpha_deg']) == 0 and f(r['rho']) == 1.0 and r['kappa_s'] == 'inf'][0]['LK'])
    if not (adam > 1e-8 and math.isfinite(adam)): continue                                   # at the floor: ratios are noise
    det = {int(r['alpha_deg']): f(r['LK']) for r in rs if r['kappa_s'] == 'inf' and f(r['rho']) == 1.0}
    abest = min(det, key=det.get); gdet = adam / det[abest]
    rho = {f(r['rho']): f(r['LK']) for r in rs if r['kappa_s'] == 'inf' and int(r['alpha_deg']) == 0}
    rbest = min(rho, key=rho.get); grho = adam / rho[rbest]
    sto = [f(r['LK']) for r in rs if r['kappa_s'] != 'inf' and f(r['rho']) == 1.0 and int(r['alpha_deg']) == 0]     # pure noise around Adam's direction
    frac = sum(1 for v in sto if v < adam) / len(sto); gsto_med = adam / statistics.median(sto)
    R = f(an[key]['R']) if key in an else float('nan'); aGN = f(an[key]['aGN']) if key in an else float('nan')
    res['all'].append((gdet, abest, grho, rbest, frac, gsto_med, R, aGN, key))
    per_traj[(key[0], key[1])].append((key[2], abest, gdet))
def summ(items, label):
    if not items: return
    gd = [x[0] for x in items]; gr = [x[2] for x in items]; fr = [x[4] for x in items]; gs = [x[5] for x in items]
    nz = [x for x in items if x[1] != 0]
    sign_ok = sum(1 for x in nz if x[7] == x[7] and (x[1] > 0) == (x[7] > 0)); nn = sum(1 for x in nz if x[7] == x[7])
    print(f"{label}: {len(items)} checkpoints (Adam not at the floor)")
    print(f"   deterministic α (ρ=1): best-of-11 gain median {statistics.median(gd):.2f}, >2× in {sum(g > 2 for g in gd)}, >10× in {sum(g > 10 for g in gd)};  best α ≠ 0 in {len(nz)}, |α*| median {statistics.median(abs(x[1]) for x in nz) if nz else 0}°, sign = GN angle's sign {sign_ok}/{nn}")
    print(f"   step length only (α=0, κ=∞): best-of-5 ρ gain median {statistics.median(gr):.2f}; ρ* = 1.5 in {sum(1 for x in items if x[3] == 1.5)}, 0.5 in {sum(1 for x in items if x[3] == 0.5)}")
    print(f"   pure noise around Adam's direction (α=0, ρ=1, κ<∞, 4 draws): beats Adam in {100*statistics.mean(fr):.0f} % of draws (50 % = neutral); median-draw gain median {statistics.median(gs):.2f}")
summ(res['all'], 'ALL')
for lo, hi in ((0, 0.3), (0.3, 0.6), (0.6, 1.01)):
    summ([x for x in res['all'] if lo <= x[6] < hi], f'max R in [{lo}, {hi})')
for name in ('wave', 'grid3', 'box2'):
    summ([x for x in res['all'] if x[8][0] == name], name)
# consistency of α* along a trajectory: agreement of sign between consecutive checkpoints (a learnable signal would persist)
agree = tot = 0
for k, lst in per_traj.items():
    lst.sort()
    for (s1, a1, g1), (s2, a2, g2) in zip(lst, lst[1:]):
        if a1 != 0 and a2 != 0 and g1 > 1.5 and g2 > 1.5: tot += 1; agree += ((a1 > 0) == (a2 > 0))
print(f"consistency: consecutive checkpoints (50 steps apart) where a deterministic α ≠ 0 helps > 1.5× in both: same sign in {agree}/{tot} ({100*agree/max(tot,1):.0f} %; 50 % = no persistent side)")

# the steering angle AFTER the step length is fixed at its best: does α still help once ρ = 0.5 (Adam's overshoot removed)?
print()
for rho0 in (0.5, 1.0):
    ga = []; nz = 0; fr = []; sg = 0; nn = 0
    for key, rs in by.items():
        base = [r for r in rs if int(r['alpha_deg']) == 0 and f(r['rho']) == rho0 and r['kappa_s'] == 'inf']
        if not base: continue
        b = f(base[0]['LK'])
        if not (b > 1e-8 and math.isfinite(b)): continue
        det = {int(r['alpha_deg']): f(r['LK']) for r in rs if r['kappa_s'] == 'inf' and f(r['rho']) == rho0}
        ab = min(det, key=det.get); ga.append(b / det[ab]); nz += (ab != 0)
        if ab != 0 and key in an and an[key]['aGN'] == an[key]['aGN']: nn += 1; sg += ((ab > 0) == (f(an[key]['aGN']) > 0))
        sto = [f(r['LK']) for r in rs if r['kappa_s'] != 'inf' and f(r['rho']) == rho0 and int(r['alpha_deg']) == 0]
        fr.append(sum(1 for v in sto if v < b) / len(sto))
    print(f"at ρ = {rho0}: {len(ga)} checkpoints; best-of-11 α gain over α = 0: median {statistics.median(ga):.2f}, >2× in {sum(g > 2 for g in ga)}, >10× in {sum(g > 10 for g in ga)}; α* ≠ 0 in {nz}; sign = GN {sg}/{nn}; noise beats the deterministic step in {100*statistics.mean(fr):.0f} % of draws")
