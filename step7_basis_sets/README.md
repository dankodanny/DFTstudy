# Step 7: Basis Sets

## What this covers

In Steps 1–5 we used a real-space grid. In real quantum chemistry codes
(including PySCF in Step 8), orbitals are expanded in a **basis set**:
`φᵢ(r) = Σ C_{μi} χ_μ(r)`. This converts the differential equation into
a matrix eigenvalue problem `FC = SCε`.

Covered here:

1. **Slater-type orbitals (STO)** — correct physics but expensive
2. **Gaussian-type orbitals (GTO)** — wrong physics but fast (analytical integrals)
3. **Contracted GTOs** (STO-3G, STO-6G) — GTOs mimicking STOs
4. **Basis set convergence** — even-tempered basis for H atom
5. **Standard basis hierarchy** — Pople (6-31G) vs Dunning (cc-pVXZ)
6. **Gaussian product theorem** — why all integrals are analytical

## How to run

```bash
conda activate dft_study
python step7_basis_sets/basis_sets.py
```

## What it produces

Console output: basis-set comparison table, convergence data, naming
conventions.

PNG files:
- `sto_vs_gto.png` — STO vs single GTO vs STO-3G vs STO-6G
- `basis_convergence.png` — H atom energy vs # of even-tempered Gaussians
- `basis_hierarchy.png` — bar chart of basis set sizes (for carbon)
- `gaussian_product.png` — product of two Gaussians is a Gaussian

## Key functions

| Function | What it does |
|----------|-------------|
| `compare_sto_gto()` | Plots STO vs GTO approximations |
| `demonstrate_basis_convergence()` | H atom with 1–15 Gaussians, analytical integrals |
| `demonstrate_basis_set_hierarchy()` | Size comparison of STO-3G through cc-pVQZ |
| `gaussian_integral_tutorial()` | Product rule visualization |

## How to experiment

- Change `alpha_0` and `beta` in `demonstrate_basis_convergence()` to
  test different even-tempered basis spacings
- Add a 4s Gaussian to the STO-6G fit and see if it improves

## Expected runtime

~5 seconds.

## Naming conventions cheat sheet

**Pople-style** `6-31G(d,p)`:
- `6` primitives for core
- `31` = split valence (3 primitives inner + 1 outer)
- `(d,p)` = polarization (d on heavy atoms, p on H)
- Add `+` for diffuse functions (anions, Rydberg states)

**Dunning-style** `cc-pVTZ`:
- `cc` correlation-consistent
- `pV` polarized valence
- `TZ` triple-zeta (DZ=double, QZ=quadruple, 5Z=quintuple)
- Designed for systematic convergence to the basis-set limit

## Next: [Step 8 — Practical DFT with PySCF](../step8_pyscf_examples/)
