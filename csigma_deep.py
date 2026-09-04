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
    def __init__(self, F, U, depth, head_units=1, embed='logstream', scale=1.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.F, self.U, self.depth, self.embed, self.scale = F, U, depth, embed, scale
        self.W = torch.nn.ParameterList([torch.nn.Parameter(0.1 * torch.complex(torch.randn(U, F, generator=g), torch.randn(U, F, generator=g))) for _ in range(depth)])
        self.a = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(F, U, dtype=torch.complex128)) for _ in range(depth)])   # identity start
        self.V = torch.nn.Parameter(0.1 * torch.complex(torch.randn(head_units, F, generator=g), torch.randn(head_units, F, generator=g)))
        self.c = torch.nn.Parameter(torch.ones(head_units, dtype=torch.complex128))

    @staticmethod
    def entry(x):
        """Log₀ on the total arithmetic; returns the complex stream and the mask of usable samples"""
        u, v = clog0(Tot(x.real), Tot(x.imag))
        return torch.complex(u.val.double(), v.val.double()), usable(u, v).all(dim=1)

    def stream(self, L0):
        L = L0; Ls = [L0]
        for W, a in zip(self.W, self.a):
            P = torch.exp(L @ W.T)                                        # [N, U]
            d = P @ a.T                                                   # [N, F]
            if self.embed == 'logstream':
                L = L + d
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
