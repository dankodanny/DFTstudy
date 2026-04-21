# Step 1: The Schrödinger Equation

## What this covers

Numerical solutions of the time-independent Schrödinger equation
`H |ψ⟩ = E |ψ⟩` for three systems with known analytical solutions — so
we can validate our finite-difference method before moving on:

1. **Particle in a 1D box** — simplest quantum system
2. **Quantum harmonic oscillator** — nontrivial potential
3. **Hydrogen atom** (radial equation) — the only atom with an exact solution

## How to run

From the project root **or** from inside this directory — both work:

```bash
# from project root
conda activate dft_study
python step1_schrodinger/schrodinger.py

# or from inside this folder
cd step1_schrodinger
python schrodinger.py
```

## What it produces

Console output: tables comparing numerical vs analytical energies for each system.

PNG files (saved next to the script):
- `particle_in_box.png` — wavefunctions + energy level comparison
- `harmonic_oscillator.png` — energy ladder with wavefunctions on the potential
- `hydrogen_atom.png` — 1s/2s/3s radial functions + energy levels

## Key functions

| Function | What it does |
|----------|-------------|
| `solve_particle_in_box(L, N)` | Build finite-difference `H`, return eigenvalues/vectors |
| `solve_harmonic_oscillator(omega, N, x_max)` | Same, with parabolic potential |
| `solve_hydrogen_atom(Z, l, N, r_max)` | Radial Schrödinger eq. `u(r) = r·R(r)` |

## How to experiment

Change parameters at the top of each `plot_*` function:
- `N` — number of grid points (higher = more accurate, slower)
- `L` / `x_max` / `r_max` — grid extent
- `Z` — nuclear charge (try `Z=2` for He⁺, `Z=3` for Li²⁺)
- `l` — angular momentum quantum number (0=s, 1=p, 2=d)

## Expected runtime

~5 seconds on a modern laptop.

## Next: [Step 2 — Many-body problem](../step2_many_body/)
