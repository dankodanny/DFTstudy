# Step 2: The Many-Body Problem

## What this covers

Why solving the Schrödinger equation exactly is impossible for more than
a few electrons, and what approximations we need:

1. **Exponential wall** — Hilbert space scales as `N_grid^N_electrons`
2. **Electron correlation** — repulsion makes electrons avoid each other
3. **Born-Oppenheimer approximation** — separate nuclear/electronic motion (H₂⁺ PES)
4. **Electron density preview** — why `ρ(r)` is the right variable for DFT

## How to run

```bash
conda activate dft_study
python step2_many_body/many_body.py
```

## What it produces

Console output: Hilbert-space scaling table, H₂⁺ binding curve minimum.

PNG files:
- `electron_correlation.png` — 2D `|Ψ(x1,x2)|²` with 0/weak/strong e-e repulsion
- `born_oppenheimer_h2plus.png` — H₂⁺ PES: `E_elec(R) + V_nn(R)`
- `electron_density_preview.png` — `ρ(x)` from Ψ via integration

## Key functions

| Function | What it does |
|----------|-------------|
| `solve_two_electrons_1d(N, L, interaction)` | Exact 2-electron diagonalization in product basis (N²×N²) |
| `demonstrate_scaling()` | Prints Hilbert-space growth table |
| `visualize_correlation()` | 3-panel plot: no/weak/strong interaction |
| `born_oppenheimer_demo()` | Scans H₂⁺ bond length, plots PES |
| `density_preview()` | Integrates out one coordinate to get ρ(x) |

## How to experiment

- Increase `interaction_strength` in `solve_two_electrons_1d` to see
  stronger correlation (electrons avoid the `x₁=x₂` diagonal more)
- Change `grid_pts` in `demonstrate_scaling()` — the point is the
  exponential, not the specific numbers

## Expected runtime

~30 seconds (the 3-panel correlation figure solves `1600×1600` matrices 3×).

## Note on H₂⁺ result

The softening parameter (0.01) is small, so at very short R the softened
Coulomb gives an artificially deep well; the minimum is at the start of
the scan range. The *concept* (PES from BO) is what matters — real 3D
codes in Step 8 give the right bond length.

## Next: [Step 3 — Hartree-Fock](../step3_hartree_fock/)
