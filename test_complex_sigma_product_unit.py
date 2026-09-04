#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""the reserved words of the complex Σ-Product Unit, as numbers.   python test_complex_sigma_product_unit.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from complex_sigma_product_unit import (Tot, GE, LE, MIN, LOG_MIN, clog0, product_unit, ComplexSigmaProductUnit, as_complex)


def test_reserved_words():
    """Log₀(0) = 0, Arg₀(0,0) = 0 (unflagged), L₀(ε) = log MIN tagged ⟦≥⟧, a zero input drops out of the product
    (P = 1), i·i = −1, the layer starts at W = 0 ⟹ P = 1, and autograd reaches every parameter."""
    z = Tot(torch.zeros(1, 1, dtype=torch.float64)); o = Tot(torch.ones(1, 1, dtype=torch.float64))
    u, v = clog0(z, z)
    assert u.val.item() == 0 and v.val.item() == 0 and u.flag.item() == 0 and v.flag.item() == 0
    eps = Tot(torch.full((1, 1), MIN), torch.full((1, 1), LE, dtype=torch.uint8))
    u, _ = clog0(eps, z)
    assert abs(u.val.item() - LOG_MIN) < 1e-4 and u.flag.item() == GE
    Pre, Pim = product_unit(z, z, torch.tensor([[-0.8]]), torch.tensor([[0.0]]))
    assert Pre.val.item() == 1.0 and Pim.val.item() == 0.0 and Pre.flag.item() == 0
    Pre, Pim = product_unit(z, o, torch.tensor([[2.0]]), torch.tensor([[0.0]]))          # i² = −1
    assert abs(Pre.val.item() + 1) < 1e-6 and abs(Pim.val.item()) < 1e-6
    layer = ComplexSigmaProductUnit(2, 3, outputs=1)
    yre, yim = layer(torch.complex(torch.randn(4, 2), torch.randn(4, 2)))
    assert torch.allclose(as_complex(yre, yim), torch.full((4,), 3 + 0j, dtype=torch.complex128))
    (yre.val.sum() + yim.val.sum()).backward()
    assert all(p.grad is not None for p in layer.parameters())


if __name__ == '__main__':
    test_reserved_words()
    print('OK: the reserved words hold')
