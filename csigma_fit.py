#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""csigma_fit — learning a complex Σ-Product Unit by *estimates averaged*, not by gradient steps.

y_n = Σ_u a_u · exp(W_u · L_n),  L_n = Log₀(x_n) ∈ ℂ^F.   The model is linear in the coefficients and, in the
log domain, linear in the exponents; so instead of averaging gradients (Adam) the fit alternates two averages
(the user's proposal, 2026-09-04):
    coefficients — given W, a = the least-squares solution of y ≈ P·a   ("the mean" in its general form);
    exponents    — given a, every usable sample n says how the exponents should move: to first order
                   y_n − Σ_u a_u P_nu = Σ_u a_u P_nu (δW_u · L_n), one complex equation per sample in all the δW;
                   the least-squares solution is the weighted average of the per-sample estimates ("the mean of
                   the ratios" — one unit at a time it is exactly Log₀(r_n/(a_u P_nu)) averaged over n; done one
                   unit at a time it diverged under 1e-7 noise, the other units' errors leaking into the ratio, so
                   the units are solved together: variable projection's Gauss–Newton step).
For F = 1 on a uniform grid of L (a log-spaced frequency sweep) the exponents are first read off *without
iteration*: y_n = Σ a_u z_u^n with z_u = e^{W_u Δ} is an exponential sum, and the matrix pencil (Hankel SVD)
gives z_u — Prony's method, with |Im W_u|·Δ < π for an unambiguous branch.  This is variable projection
(varpro) with a Prony start, the structure the gradient method cannot see.
Every value goes through the total arithmetic of the layer (Log₀ 0 = 0, ε tagged, flags): a flagged sample is
no number and is left out of every average.
"""
import math
import torch
from complex_sigma_product_unit import ComplexSigmaProductUnit, Tot, clog0, usable, as_complex


def log_features(x_re: Tot, x_im: Tot):
    """L = Log₀(x) as complex128 [N, F] and the mask of the usable (unflagged / floored) samples."""
    u, v = clog0(x_re, x_im)
    L = torch.complex(u.val.double(), v.val.double())
    m = usable(u, v).all(dim=1)
    return L, m


def products(L, W):
    """P[n, u] = exp(Σ_f W[u, f]·L[n, f])  (complex128)."""
    return torch.exp(L @ W.T)


def fit_coefficients(P, y, w=None):
    """a = argmin Σ w_n |y_n − P_n·a|²  (complex least squares; a non-finite P — an exponent far off — gives a = 0)."""
    if not torch.isfinite(P).all():
        return torch.zeros(P.shape[1], dtype=P.dtype)
    sw = torch.ones_like(y.real) if w is None else w.sqrt()
    return torch.linalg.lstsq(P * sw.unsqueeze(1), (y * sw).unsqueeze(1)).solution[:, 0]


def vonmises_center(z, weights=None):
    """the centre of a cloud of complex estimates as a direction and a size: the von Mises mean direction θ̄ of the
    phases (arg Σ w e^{iθ}), its mean resultant length R̄ ∈ [0, 1] (the concentration: 1 = every sample agrees,
    0 = no direction at all) and the weighted median of the magnitudes.  Returns (R̄ · median|z| · e^{iθ̄}, R̄)."""
    z = z[torch.isfinite(z)]
    if z.numel() == 0:
        return torch.zeros((), dtype=torch.complex128), 0.0
    w = torch.ones_like(z.real) if weights is None else weights[:z.numel()]
    u = z / z.abs().clamp(min=1e-300)
    res = (w * u).sum() / w.sum()
    R = res.abs().item()
    mag = torch.quantile(z.abs(), 0.5).item()
    return R * mag * res / max(R, 1e-300), R


def pencil_init(y, U, delta, L0):
    """the matrix pencil on a uniform grid L_n = L0 + n·Δ (F = 1): y_n = Σ a_u z_u^n → W_u = log(z_u)/Δ
    (the offset L0 goes into the coefficients).  Returns W [U, 1] complex128."""
    N = y.numel(); M = N // 2
    H = torch.stack([y[i:i + N - M] for i in range(M + 1)], 0)              # (M+1) × (N−M) Hankel, H = Σ c_u p_u q_uᵀ
    Uh, S, Vh = torch.linalg.svd(H, full_matrices=False)
    Us = Uh[:, :U]                                                           # the left vectors span {p_u} as they are
    Z = torch.linalg.pinv(Us[:-1]) @ Us[1:]                                  # (the right ones span their conjugates: W̄)
    z = torch.linalg.eigvals(Z)
    W = torch.log(z) / delta                                                 # principal branch: |Im W|·Δ < π
    return W.unsqueeze(1)


def fit_moments(L, y, U, W0, iters=60, damp=1.0, tol=1e-12, weights='relative', center='lstsq'):
    """alternate the two averages from W0 [U, F]; returns W, a, the relative loss, the iteration count.
    center 'lstsq': the correction of all exponents at once as the least-squares average of the per-sample equations
    (Gauss–Newton); 'vonmises': for each exponent, the per-sample corrections δ_n = e_n / (a_u P_nu L_nf) form a
    cloud in ℂ whose centre is taken robustly — the von Mises mean direction of the phases, the median magnitude,
    the resultant length R̄ as the confidence that scales the step (the user's proposal: a centre of the gradient
    distribution instead of its mean; leakage from the other units and boundary samples scatter the phases and
    move the centre little)."""
    W = W0.clone(); N, F = L.shape
    wgt = 1 / (y.abs() ** 2).clamp(min=1e-300) if weights == 'relative' else torch.ones_like(y.real)
    P = products(L, W); a = fit_coefficients(P, y, wgt)
    loss = ((wgt * (P @ a - y).abs() ** 2).mean()).item()
    sw = wgt.sqrt()
    for it in range(iters):
        e = y - P @ a                                                        # the residual
        J = (a.unsqueeze(0) * P).unsqueeze(2) * L.unsqueeze(1)               # [N, U, F]: ∂ŷ_n/∂W_uf = a_u P_nu L_nf
        if not torch.isfinite(J).all():
            break
        if center == 'vonmises':
            dW = torch.zeros(U, F, dtype=torch.complex128)
            for u in range(U):
                for f in range(F):
                    d = e / J[:, u, f]                                       # what each sample says W_uf should move by
                    c, R = vonmises_center(d, wgt)
                    dW[u, f] = c
        else:
            J = J.reshape(N, U * F)
            dW = torch.linalg.lstsq(J * sw.unsqueeze(1), (e * sw).unsqueeze(1)).solution[:, 0].reshape(U, F)
        step = damp; accepted = False
        while step >= 1e-4:                                                  # a step that does not reduce the loss is halved …
            Wn = W + step * dW
            Pn = products(L, Wn); an = fit_coefficients(Pn, y, wgt)
            new = ((wgt * (Pn @ an - y).abs() ** 2).mean()).item()
            if math.isfinite(new) and new <= loss:
                accepted = True; break
            step /= 2
        if not accepted:                                                     # … and no acceptable step means: stop here
            break
        W, P, a = Wn, Pn, an
        if new < tol or abs(loss - new) < 1e-3 * loss:
            loss = new; break
        loss = new
    return W, a, loss, it + 1


def fit(layer: ComplexSigmaProductUnit, x_re: Tot, x_im: Tot, y_re: Tot, y_im: Tot, U: int, init='random', iters=60, seed=0, delta=None, restarts=4, center='lstsq'):
    """fit the layer's exponents and coefficients by estimates averaged; init 'pencil' (F = 1, uniform grid in L,
    `delta` = the grid step) or 'random' (restarts, the best kept).  Writes the parameters into the layer."""
    L, m = log_features(x_re, x_im)
    y = torch.complex(y_re.val.double(), y_im.val.double())
    m = m & usable(y_re, y_im)
    L, y = L[m], y[m]
    N, F = L.shape
    best = None
    g = torch.Generator().manual_seed(seed)
    starts = []
    if init == 'pencil':
        starts.append(pencil_init(y, U, delta, L[0, 0]))
    for _ in range(restarts if init == 'random' else 1):
        starts.append(torch.complex(torch.randn(U, F, generator=g), torch.randn(U, F, generator=g)) * 0.3)
    for W0 in starts:
        W, a, loss, it = fit_moments(L, y, U, W0, iters=iters, center=center)
        if best is None or loss < best[2]:
            best = (W, a, loss, it)
    W, a, loss, it = best
    layer.set_exponent(W.real, W.imag)
    if layer.outputs:
        layer.set_coefficient(a.reshape(1, -1))
    return W, a, loss, it
