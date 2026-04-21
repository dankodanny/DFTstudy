# Step 3: Hartree-Fock Method

## What this covers

A complete 1D Hartree-Fock solver built from scratch. HF is the first
practical approximation to the many-electron problem and is the direct
ancestor of Kohn-Sham DFT.

Core ideas implemented:

1. **Slater determinant** — antisymmetric wavefunction from orbitals
2. **Fock operator** `F = h + J − K`
3. **Coulomb operator J** — classical electrostatic repulsion
4. **Exchange operator K** — non-local, quantum-only, from antisymmetry
5. **SCF procedure** — iterate density → potential → orbitals → density

## How to run

```bash
conda activate dft_study
python step3_hartree_fock/hartree_fock.py
```

## What it produces

Console output: SCF convergence table, final energies.

PNG files:
- `hf_helium_1d.png` — 4-panel: SCF curve, orbital, density, V_eff
- `fock_components.png` — V_nuc, J, −K, and total V_eff overlaid

## Key class

### `HartreeFock1D(N_grid, x_max, N_elec, softening)`

Instantiate, set a potential, then call `.solve()`:

```python
from hartree_fock import HartreeFock1D
import numpy as np

hf = HartreeFock1D(N_grid=150, x_max=15.0, N_elec=2, softening=0.5)
V_nuc = -2.0 / np.sqrt(hf.x**2 + 0.5**2)    # 1D helium nucleus
hf.set_potential(V_nuc)
E, orbitals, eigvals = hf.solve(max_iter=50, tol=1e-8, mixing=0.5)
```

## How to experiment

- Vary `N_elec` (must be even for the restricted HF version)
- Change `softening` — smaller = sharper Coulomb, harder to converge
- Change `mixing` — smaller value = slower but more stable SCF
- Try other potentials in `demo_helium_1d()`: e.g. harmonic `0.5*x**2`

## Expected runtime

~20 seconds (Be demo + component plot).

## What HF misses

HF captures **exchange exactly** but misses **correlation entirely**.
The missing piece `E_corr = E_exact - E_HF` is small (~1% of total
energy) but chemically crucial. DFT aims to capture both through the
exchange-correlation functional.

## Next: [Step 4 — Hohenberg-Kohn theorems](../step4_hohenberg_kohn/)
