# Step 5: Kohn-Sham DFT from Scratch

## What this covers

A **complete Kohn-Sham DFT solver** written from scratch with only NumPy
and SciPy. This is the heart of DFT — everything else (functionals,
basis sets, PySCF) is engineering around this core algorithm.

Implemented:

1. **KS equations** `[-½∇² + V_KS]φᵢ = εᵢφᵢ` on a real-space grid
2. **Effective potential** `V_KS = V_ext + V_H + V_xc`
3. **Hartree potential** — direct integration of `ρ(r')/|r-r'|`
4. **LDA exchange-correlation** — simple 1D-adapted functional
5. **Self-consistent field loop** with density mixing

## How to run

```bash
conda activate dft_study
python step5_kohn_sham/kohn_sham.py
```

## What it produces

Console output: SCF convergence tables, final energy decomposition.

PNG files:
- `ks_helium.png` — 4-panel: SCF curve, orbitals, density, potentials
- `ks_h2_molecule.png` — H₂ binding curve + equilibrium density
- `scf_convergence.png` — density evolution during SCF iterations

## Key class

### `KohnShamDFT1D(N_grid, x_max, N_elec, softening)`

```python
from kohn_sham import KohnShamDFT1D

# 1D helium atom
dft = KohnShamDFT1D(N_grid=200, x_max=15.0, N_elec=2, softening=0.5)
dft.set_nuclear_potential(Z_list=[2.0], R_list=[0.0])
E = dft.solve(max_iter=100, tol=1e-9, mixing=0.4)

# Access results
dft.E_total          # total energy
dft.rho              # ground-state density
dft.orbitals         # KS orbitals
dft.eigenvalues      # KS orbital energies
dft.E_components     # dict: E_band, E_H, E_xc, E_vxc, V_nn
```

### Multi-atom molecules

```python
# 1D H2 molecule
dft.set_nuclear_potential(Z_list=[1.0, 1.0], R_list=[-0.7, 0.7])
```

## How to experiment

| Parameter | Effect |
|-----------|--------|
| `N_grid` | More = better accuracy, slower (~N³ scaling) |
| `x_max` | Must be large enough that density → 0 at boundary |
| `N_elec` | Must be even (restricted KS; closed-shell only) |
| `softening` | 1D Coulomb regularizer; larger = easier convergence |
| `mixing` | 0.1–0.5 typical; smaller = more stable, slower |
| `tol` | Energy convergence threshold (1e-7 is tight enough) |

## Expected runtime

~1–2 minutes (Be SCF convergence demo is the slow part).

## The SCF algorithm in 6 lines

```python
while not converged:
    V_KS = V_ext + V_H(rho) + V_xc(rho)     # build effective potential
    H = T + diag(V_KS)                       # kinetic + local potential
    eigenvalues, orbitals = eigh(H)          # diagonalize
    rho_new = sum(2 * orbitals[:, :N_occ]**2, axis=1)
    rho = mixing * rho_new + (1 - mixing) * rho
    # check energy convergence
```

## Next: [Step 6 — XC functionals](../step6_xc_functionals/)
