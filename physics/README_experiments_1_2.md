# Complex Product Unit — Physics sanity tests

Two PyTorch experiments:

- `rlc_physics_demo.py` — series RLC impedance
  - exponent-first fitting by coefficient-cancelling ratios
  - freezes exponents
  - learns complex coefficients from the full complex impedance
  - compares Walsh row-tensor coefficient initializations

- `schrodinger_plane_wave_demo.py` — free-particle Schrödinger plane wave
  - rewrites `exp(i(kx-ωt))` as `X^(ik) T^(-iω)` with `X=e^x`, `T=e^t`
  - recovers genuinely imaginary exponents
  - then learns the unknown complex amplitude

Run:

```bash
python rlc_physics_demo.py
python schrodinger_plane_wave_demo.py
```

See `EXPERIMENT_RESULT.md` for results and limitations.
