# Step 8: Practical DFT with PySCF

## What this covers

Real molecular DFT calculations using **PySCF** — an open-source
quantum chemistry package that implements everything from Steps 1–7
(plus analytical integrals, DIIS convergence, etc.).

Examples included:

1. **H₂ single-point** — first DFT calculation (PBE/cc-pVDZ)
2. **Functional comparison** — HF / LDA / GGA / hybrid on H₂O
3. **Basis set convergence** — STO-3G up to cc-pVQZ for H₂O
4. **Geometry optimization** — optimize H₂O (requires `geometric`)
5. **Potential energy surface** — H₂ dissociation curve
6. **Molecular properties** — dipole moment, HOMO-LUMO gap, Mulliken charges

## How to run

```bash
conda activate dft_study
python step8_pyscf_examples/pyscf_dft.py
```

The script calls each demo in sequence. To run just one, import it:

```python
from step8_pyscf_examples.pyscf_dft import first_dft_calculation
first_dft_calculation()
```

## What it produces

Console output: energies, orbital levels, dipole moments, equilibrium geometries.

PNG files:
- `functional_comparison.png` — H₂O energies & HOMO-LUMO gap by functional
- `basis_convergence.png` — H₂O energy vs basis set size (sto-3g → cc-pVQZ)
- `h2_pes.png` — H₂ dissociation curve (HF vs PBE vs B3LYP)

## Key PySCF idioms

### Define a molecule
```python
from pyscf import gto, dft
mol = gto.M(
    atom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469',
    basis='6-31g(d)',
    charge=0,
    spin=0,       # N_alpha - N_beta (0 = closed shell)
)
```

### Run DFT
```python
mf = dft.RKS(mol)       # restricted Kohn-Sham
mf.xc = 'b3lyp'         # any libxc functional name
E = mf.kernel()
```

### Available quantities after SCF
```python
mf.e_tot        # total energy (hartree)
mf.mo_energy    # KS orbital energies
mf.mo_coeff     # MO coefficients (C matrix)
mf.make_rdm1()  # 1-electron reduced density matrix
mf.dip_moment() # dipole moment (Debye)
```

### Geometry optimization
```python
from pyscf.geomopt import geometric_solver
mol_eq = geometric_solver.optimize(mf, maxsteps=50)
```

## Expected runtime

~2 minutes for the full script. The heaviest part is the cc-pVQZ
calculation on H₂O (~30 seconds).

## Common functional keywords

| Keyword | Type | Notes |
|---------|------|-------|
| `'svwn'` | LDA | Slater exchange + VWN correlation |
| `'pbe'` | GGA | Most common for solids |
| `'blyp'` | GGA | Common for molecules |
| `'b3lyp'` | Hybrid | 20% HF exchange; most cited ever |
| `'pbe0'` | Hybrid | 25% HF exchange |
| `'m06-2x'` | Meta-GGA hybrid | Minnesota family, good for kinetics |
| `'wb97x-d'` | RS hybrid + D | Dispersion-corrected |

## How to extend

- Try other molecules: `gto.M(atom='...', basis='...')` — any SMILES-like format
- Run excited states: `td = mf.TDDFT(); td.kernel()`
- Compare to post-HF: `from pyscf import cc; cc.CCSD(mf).run()`
- Periodic solids: `from pyscf.pbc import gto as pbcgto`
- Charges: `pop, chg = mf.mulliken_pop()`

## Troubleshooting

**SCF not converging?**
```python
mf = dft.RKS(mol)
mf.xc = 'b3lyp'
mf.level_shift = 0.2   # level shifting
mf.damp = 0.3          # damping
mf.max_cycle = 200
mf.kernel()
```

**Geometry optimization fails?** Install `geometric`:
```bash
pip install geometric
```
(It's already in `environment.yml`.)

## Done!

You've gone from the time-independent Schrödinger equation all the way
to real molecular DFT calculations. Next steps on your own:

- **TDDFT** for excited states & UV-vis spectra
- **Periodic DFT** for solids (PySCF has PBC support)
- **Post-HF methods** (MP2, CCSD(T)) for benchmark accuracy
- **Machine-learned functionals** (DeepMind's DM21, etc.)
