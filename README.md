# DFT Study: Density Functional Theory from Scratch

A step-by-step tutorial building up Density Functional Theory from the
Schrödinger equation, implemented in Python from basic principles, and
culminating in practical calculations with PySCF.

## Prerequisites

- Linear algebra (eigenvalues, matrices)
- Multivariable calculus
- Basic quantum mechanics concepts
- Python with NumPy / SciPy

## Setup (one-time)

```bash
conda env create -f environment.yml
conda activate dft_study
```

This installs: NumPy, SciPy, Matplotlib, Jupyter, PySCF, and `geometric`
(for geometry optimization).

## How to run the tutorials

Every step is a single self-contained Python file with plots and printed
output. You can run them from the project root **or** from inside each
step directory — both work:

```bash
conda activate dft_study

# from project root
python step1_schrodinger/schrodinger.py

# or from inside the folder
cd step1_schrodinger
python schrodinger.py
```

PNG plots are saved next to each script. Each step directory has its own
`README.md` with usage details for that specific file.

### Recommended order

| # | Folder | Script | Runtime | Read first |
|---|--------|--------|---------|------------|
| 1 | [`step1_schrodinger/`](step1_schrodinger/README.md) | `schrodinger.py` | ~5 s | Particle-in-box, HO, H atom |
| 2 | [`step2_many_body/`](step2_many_body/README.md) | `many_body.py` | ~30 s | Why we need approximations |
| 3 | [`step3_hartree_fock/`](step3_hartree_fock/README.md) | `hartree_fock.py` | ~20 s | First mean-field method |
| 4 | [`step4_hohenberg_kohn/`](step4_hohenberg_kohn/README.md) | `hohenberg_kohn.py` | ~10 s | The HK theorems |
| 5 | [`step5_kohn_sham/`](step5_kohn_sham/README.md) | `kohn_sham.py` | ~2 min | **Complete KS-DFT solver** |
| 6 | [`step6_xc_functionals/`](step6_xc_functionals/README.md) | `xc_functionals.py` | ~5 s | LDA/GGA/hybrid |
| 7 | [`step7_basis_sets/`](step7_basis_sets/README.md) | `basis_sets.py` | ~5 s | STO vs GTO |
| 8 | [`step8_pyscf_examples/`](step8_pyscf_examples/README.md) | `pyscf_dft.py` | ~2 min | Real molecules with PySCF |

### Running everything at once

```bash
for f in step*/*.py; do
    echo "=== $f ==="
    python "$f"
done
```

## The conceptual roadmap

```
Schrödinger equation  (exact, intractable for N > 2)
         |
         v
Born-Oppenheimer      (decouple nuclei from electrons)
         |
         v
Hartree-Fock          (mean-field, misses correlation)
         |
         v
Hohenberg-Kohn        (electron density is sufficient!)
         |
         v
Kohn-Sham             (interacting → non-interacting mapping)
         |
         v
XC functionals        (where the approximations live)
         |
         v
Basis sets + PySCF    (practical implementation)
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'pyscf'`**
Activate the env: `conda activate dft_study`.

**`FileNotFoundError: ... .png`**
You're on an older version of the scripts. Pull latest — the savefig
paths are now relative to the script, not the working directory.

**Plot windows don't appear**
If you're running over SSH or in a headless environment:
```bash
MPLBACKEND=Agg python step1_schrodinger/schrodinger.py
```
Plots still get saved to PNG files.

**SCF doesn't converge in Step 5**
Reduce `mixing` (e.g. 0.2), increase `max_iter`, or increase `softening`.

## Repository

<https://github.com/dankodanny/DFTstudy>
