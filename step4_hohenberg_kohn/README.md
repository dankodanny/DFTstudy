# Step 4: Hohenberg-Kohn Theorems

## What this covers

The two theorems that justify DFT (Hohenberg & Kohn, 1964):

1. **Theorem 1 (existence)** — the external potential V_ext is uniquely
   determined by the ground-state density ρ₀(r)
2. **Theorem 2 (variational)** — there exists a universal functional
   F[ρ] whose minimization gives the exact ground-state energy
3. **Energy functional decomposition** — the road from HK to Kohn-Sham
4. **v-representability** — which densities are valid

This is the theoretical bridge between HF (Step 3) and Kohn-Sham (Step 5).

## How to run

```bash
conda activate dft_study
python step4_hohenberg_kohn/hohenberg_kohn.py
```

## What it produces

Console output: theorem demonstrations with numerical verification.

PNG files:
- `hk_theorem1.png` — 4 different potentials → 4 distinct densities
- `hk_theorem2.png` — trial Gaussian densities always give `E[ρ] ≥ E₀`
- `energy_decomposition.png` — bar chart of T_s, E_ext, E_H, E_x, E_c for Ne
- `v_representability.png` — valid vs invalid densities

## Key functions

| Function | What it demonstrates |
|----------|---------------------|
| `demonstrate_hk_theorem1()` | Different V_ext produce different ρ (unique map) |
| `demonstrate_hk_theorem2()` | Variational principle: `E[ρ_trial] ≥ E_0` |
| `energy_functional_decomposition()` | Why we split `F = T_s + E_H + E_xc` |
| `v_representability_demo()` | Valid ρ: non-negative, smooth, normalized |

## How to experiment

- Add more potentials in `demonstrate_hk_theorem1()` — they'll always give
  unique densities
- Change trial density form in `demonstrate_hk_theorem2()` (e.g. Lorentzian
  instead of Gaussian) — the minimum still bounds `E_exact` from above

## Expected runtime

~10 seconds.

## Why this matters

HK theorems are **existence proofs** — they tell us `F[ρ]` exists but
not what it is. Finding good approximations to `F[ρ]` is the central
challenge of DFT. Kohn-Sham (Step 5) shows how to sidestep the hardest
part (kinetic energy `T[ρ]`) by using orbitals.

## Next: [Step 5 — Kohn-Sham equations](../step5_kohn_sham/)
