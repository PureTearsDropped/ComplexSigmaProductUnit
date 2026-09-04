#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""csigma_deep — a deep complex Σ-Product Unit with a LOG-STREAM residual (the multiplicative residual):

    L₀ = Log₀(x)                                   (the only Log: the entry, on the total arithmetic — zeros drop out)
    Δ_ℓ[n, f] = Σ_u a_ℓ[f, u] · exp(W_ℓ[u]·L_{ℓ−1}[n])      a block: a Σ-PU whose F outputs are exponent increments
    L_ℓ = L_{ℓ−1} + Δ_ℓ                            ⟺  x_ℓ = x_{ℓ−1} · exp(Δ_ℓ): the residual is multiplicative
    y = Σ_u c[u] · exp(V[u]·L_D)                   the head

The stream keeps the magnitude and the *unwrapped* phase; the output of a block is never logged again, so no
branch cut and no zero appear inside the network.  With a_ℓ = 0 at the start every block is the identity — a
ResNet's zero-initialised last layer, from the four-channel construction (a = a⁺ − a⁻).
Alternative for comparison (`embed='oneplus'`, the user's z = 1 + i·y/s): each block outputs y_ℓ ∈ ℂ^F, the next
stream is Log(1 + i·y_ℓ/s) — bounded, off the cut, but the phase saturates (atan).
"""
import math
import torch
from complex_sigma_product_unit import Tot, clog0, usable


class DeepCSPU(torch.nn.Module):
    def __init__(self, F, U, depth, head_units=1, embed='logstream', scale=1.0, seed=0, a_init=0.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.F, self.U, self.depth, self.embed, self.scale = F, U, depth, embed, scale
        self.W = torch.nn.ParameterList([torch.nn.Parameter(0.1 * torch.complex(torch.randn(U, F, generator=g), torch.randn(U, F, generator=g))) for _ in range(depth)])
        self.a = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(F, U, dtype=torch.complex128)) for _ in range(depth)])   # identity start
        if a_init:                                                        # a small symmetry-breaking start (W gets no gradient while a = 0)
            with torch.no_grad():
                for a in self.a: a.add_(a_init * torch.complex(torch.randn(a.shape, generator=g), torch.randn(a.shape, generator=g)))
        self.V = torch.nn.Parameter(0.1 * torch.complex(torch.randn(head_units, F, generator=g), torch.randn(head_units, F, generator=g)))
        self.c = torch.nn.Parameter(torch.ones(head_units, dtype=torch.complex128))

    @staticmethod
    def entry(x, with_zero=False):
        """Log₀ on the total arithmetic; returns the complex stream and the mask of usable samples.  With `with_zero`,
        also q ∈ {0, 1} per feature — the discrete zero state the log stream cannot carry (Log₀ 0 = 0 is the reserved
        word, and exp(0) = 1 ≠ 0): the value is q·exp(L)."""
        u, v = clog0(Tot(x.real), Tot(x.imag))
        L = torch.complex(u.val.double(), v.val.double()); m = usable(u, v).all(dim=1)
        if with_zero:
            return L, m, (x != 0).double()
        return L, m

    @staticmethod
    def value(L, q=None):
        """back to the value: x = q · exp(L) — an exact zero stays an exact zero"""
        return torch.exp(L) if q is None else q * torch.exp(L)

    def stream(self, L0):
        L = L0; Ls = [L0]
        for W, a in zip(self.W, self.a):
            P = torch.exp(L @ W.T)                                        # [N, U]
            d = P @ a.T                                                   # [N, F]
            if self.embed == 'logstream':
                L = L + d
            elif self.embed == 'stacked':                                 # the plain stack: the next layer logs the output (principal branch)
                L = torch.log(d)
            else:                                                         # z = 1 + i·y/s, logged again (principal branch)
                L = torch.log(1 + 1j * d / self.scale)
            Ls.append(L)
        return L, Ls

    def forward(self, L0):
        L, _ = self.stream(L0)
        return torch.exp(L @ self.V.T) @ self.c

    def head_lstsq(self, L0, y, w):
        """the head's coefficients are linear given the exponents: solve them exactly (final-stage-first)"""
        with torch.no_grad():
            L, _ = self.stream(L0); P = torch.exp(L @ self.V.T); sw = w.sqrt()
            self.c.copy_(torch.linalg.lstsq(P * sw.unsqueeze(1), (y * sw).unsqueeze(1)).solution[:, 0])
