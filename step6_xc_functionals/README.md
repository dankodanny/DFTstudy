# Step 6: Exchange-Correlation Functionals

## What this covers

The XC functional `E_xc[ρ]` is where *all* the approximations of DFT
live. This step surveys the hierarchy of functionals (Jacob's ladder)
and implements the key ones.

Implemented:

1. **LDA exchange** (Dirac 1930) — 3D `e_x ∝ ρ^(1/3)`
2. **VWN correlation** (Vosko-Wilk-Nusair 1980) — fit to QMC data
3. **PW92 correlation** (Perdew-Wang 1992) — simpler parametrization
4. **PBE exchange** (Perdew-Burke-Ernzerhof 1996) — GGA with `F_x(s)`
5. **Jacob's ladder** visualization — LDA → GGA → mGGA → hybrid → double hybrid

## How to run

```bash
conda activate dft_study
python step6_xc_functionals/xc_functionals.py
```

## What it produces

Console output: practical functional-selection guide, rules of thumb.

PNG files:
- `jacobs_ladder.png` — 5-rung ladder showing cost vs accuracy tradeoff
- `lda_exchange.png` — `e_x(ρ)` and `V_x(ρ)` vs density
- `pbe_enhancement.png` — `F_x(s)` for PBE vs revPBE vs LDA
- `correlation_vs_exchange.png` — relative magnitudes of E_c vs E_x

## Key functions

| Function | Formula |
|----------|---------|
| `lda_exchange_3d(rho)` | `e_x = -¾(3/π)^(1/3) · ρ^(1/3)` |
| `vwn_correlation_3d(rho)` | VWN-5 parametrization of uniform electron gas |
| `pw92_correlation_3d(rho)` | Perdew-Wang 1992 |
| `pbe_exchange(rho, grad_rho)` | `e_x^PBE = e_x^LDA · F_x(s)` |

## How to use these in your own code

```python
from xc_functionals import lda_exchange_3d, pw92_correlation_3d

# For a density rho on a 3D grid:
e_x, V_x = lda_exchange_3d(rho)
e_c, V_c = pw92_correlation_3d(rho)

E_xc = integrate(rho * (e_x + e_c), grid)    # energy
V_xc = V_x + V_c                              # potential → goes into KS eq
```

## Which functional should I use?

Quick reference (full table in the script):

| Use case | Recommended |
|----------|-------------|
| Quick estimate, metals | LDA (SVWN) |
| Solids, general | PBE (GGA) |
| Organic molecules | B3LYP (hybrid) — most cited functional in history |
| Band gaps in solids | HSE06 (range-separated hybrid) |
| Non-covalent interactions | ω-B97X-D (with dispersion) |
| Benchmark-quality | Double hybrid (B2PLYP, DSD-PBEP86) |

## Expected runtime

~5 seconds (this step is mostly visualization).

## Next: [Step 7 — Basis sets](../step7_basis_sets/)
