# DFT Study: Density Functional Theory from Scratch

A step-by-step tutorial building up Density Functional Theory from the Schrodinger equation,
implemented in Python from basic principles, culminating in practical calculations with PySCF.

## Prerequisites

- Linear algebra (eigenvalues, matrices)
- Multivariable calculus
- Basic quantum mechanics concepts
- Python with NumPy/SciPy

## Setup

```bash
conda env create -f environment.yml
conda activate dft_study
```

## Tutorial Structure

| Step | Topic | Key Concepts |
|------|-------|-------------|
| 1 | [Schrodinger Equation](step1_schrodinger/) | Time-independent SE, particle in a box, hydrogen atom |
| 2 | [Many-Body Problem](step2_many_body/) | Born-Oppenheimer approximation, electron-electron interaction |
| 3 | [Hartree-Fock Method](step3_hartree_fock/) | Mean-field theory, Slater determinants, SCF procedure |
| 4 | [Hohenberg-Kohn Theorems](step4_hohenberg_kohn/) | Electron density, existence & variational theorems |
| 5 | [Kohn-Sham Equations](step5_kohn_sham/) | KS ansatz, effective potential, SCF from scratch |
| 6 | [XC Functionals](step6_xc_functionals/) | LDA, GGA, hybrid functionals, Jacob's ladder |
| 7 | [Basis Sets](step7_basis_sets/) | STO vs GTO, basis set convergence |
| 8 | [Practical DFT with PySCF](step8_pyscf_examples/) | Real molecular calculations |

## Road to DFT

```
Schrodinger Equation (exact, but unsolvable for N>1)
        |
        v
Born-Oppenheimer Approximation (separate nuclear/electronic motion)
        |
        v
Hartree-Fock (mean-field, misses correlation)
        |
        v
Hohenberg-Kohn Theorems (density is the fundamental variable)
        |
        v
Kohn-Sham Equations (map interacting -> non-interacting system)
        |
        v
Exchange-Correlation Functionals (the art of DFT)
        |
        v
Practical DFT Calculations (PySCF, basis sets, real molecules)
```
