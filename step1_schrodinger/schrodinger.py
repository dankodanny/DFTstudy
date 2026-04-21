"""
==============================================================================
STEP 1: The Schrodinger Equation - Foundation of Quantum Mechanics
==============================================================================

The time-independent Schrodinger equation is the starting point for ALL
electronic structure methods, including DFT:

    H |psi> = E |psi>

where:
    H   = Hamiltonian operator (total energy operator)
    psi = wavefunction (contains ALL information about the system)
    E   = energy eigenvalue

For a single particle in a potential V(r):

    H = -hbar^2/(2m) * nabla^2 + V(r)
      = T (kinetic) + V (potential)

In atomic units (hbar=1, m_e=1, e=1, 4*pi*eps_0=1):

    H = -1/2 * nabla^2 + V(r)

We will solve this numerically for:
    1. Particle in a 1D box (exact solution known)
    2. Harmonic oscillator (exact solution known)
    3. Hydrogen atom radial equation (exact solution known)

These validate our numerical methods before tackling harder problems.
==============================================================================
"""

import os
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt
from scipy.linalg import eigh


# ==========================================================================
# PART 1: Particle in a 1D Box
# ==========================================================================
#
# The simplest quantum system. A particle confined to [0, L] with
# infinite potential walls:
#
#   V(x) = 0      for 0 < x < L
#   V(x) = inf    otherwise
#
# Exact solutions:
#   psi_n(x) = sqrt(2/L) * sin(n*pi*x/L)
#   E_n = n^2 * pi^2 / (2*L^2)    (in atomic units)
#
# Numerical approach: Finite difference method
# We discretize -1/2 * d^2/dx^2 on a grid and diagonalize.

def solve_particle_in_box(L=1.0, N=100):
    """
    Solve the particle in a 1D box using finite differences.

    The key idea: replace the continuous derivative with a discrete
    approximation on a grid of N points:

        d^2 psi/dx^2 ≈ (psi[i+1] - 2*psi[i] + psi[i-1]) / dx^2

    This turns the differential equation into a matrix eigenvalue problem:

        H @ psi = E * psi

    where H is a tridiagonal matrix.

    Parameters
    ----------
    L : float
        Box length (atomic units, bohr)
    N : int
        Number of grid points

    Returns
    -------
    x : grid points
    energies : first few eigenvalues
    wavefunctions : corresponding eigenvectors
    """
    # Grid spacing
    dx = L / (N + 1)
    x = np.linspace(dx, L - dx, N)  # interior points only (psi=0 at walls)

    # Build the kinetic energy matrix T = -1/2 * d^2/dx^2
    # Using the 3-point finite difference stencil:
    #   d^2f/dx^2 ≈ (f[i-1] - 2f[i] + f[i+1]) / dx^2
    #
    # So T_ij = -1/2 * (1/dx^2) * {-2 if i==j, +1 if |i-j|==1}
    #         = (1/dx^2) * {+1 if i==j, -1/2 if |i-j|==1}

    diag = np.ones(N) / dx**2          # main diagonal: 1/dx^2
    off_diag = -0.5 * np.ones(N-1) / dx**2  # off-diagonals: -1/(2*dx^2)

    H = np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

    # V(x) = 0 inside the box, so H = T (no potential term needed)

    # Solve the eigenvalue problem
    energies, wavefunctions = eigh(H)

    # Normalize wavefunctions (they come out normalized in discrete sense,
    # but let's be explicit)
    for i in range(wavefunctions.shape[1]):
        norm = np.sqrt(np.trapezoid(wavefunctions[:, i]**2, x))
        wavefunctions[:, i] /= norm

    return x, energies, wavefunctions


def plot_particle_in_box():
    """Compare numerical and exact solutions for particle in a box."""
    L = 1.0
    N = 200
    x, energies, psi = solve_particle_in_box(L, N)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left panel: Wavefunctions ---
    ax = axes[0]
    n_states = 4
    for n in range(1, n_states + 1):
        # Exact solution
        psi_exact = np.sqrt(2/L) * np.sin(n * np.pi * x / L)

        # Numerical solution (may have sign flip)
        psi_num = psi[:, n-1]
        if np.dot(psi_num, psi_exact) < 0:
            psi_num = -psi_num

        ax.plot(x, psi_exact, 'k--', alpha=0.5, linewidth=1)
        ax.plot(x, psi_num + (n-1)*3, label=f'n={n}', linewidth=2)

    ax.set_xlabel('x (bohr)')
    ax.set_ylabel('psi(x) (offset for clarity)')
    ax.set_title('Wavefunctions: Particle in a Box')
    ax.legend()

    # --- Right panel: Energy levels ---
    ax = axes[1]
    n_levels = 8
    n_arr = np.arange(1, n_levels + 1)
    E_exact = n_arr**2 * np.pi**2 / (2 * L**2)
    E_numerical = energies[:n_levels]

    ax.barh(n_arr - 0.15, E_exact, height=0.3, label='Exact', alpha=0.7)
    ax.barh(n_arr + 0.15, E_numerical, height=0.3, label='Numerical', alpha=0.7)
    ax.set_ylabel('Quantum number n')
    ax.set_xlabel('Energy (hartree)')
    ax.set_title('Energy Levels')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(_DIR, 'particle_in_box.png'), dpi=150)
    plt.show()

    # Print comparison
    print("=" * 60)
    print("Particle in a Box: Numerical vs Exact Energies")
    print("=" * 60)
    print(f"{'n':>3} {'E_exact':>15} {'E_numerical':>15} {'Error':>15}")
    print("-" * 60)
    for n in range(1, n_levels + 1):
        E_ex = n**2 * np.pi**2 / (2 * L**2)
        E_num = energies[n-1]
        print(f"{n:3d} {E_ex:15.8f} {E_num:15.8f} {abs(E_ex - E_num):15.2e}")


# ==========================================================================
# PART 2: Quantum Harmonic Oscillator
# ==========================================================================
#
# V(x) = 1/2 * omega^2 * x^2
#
# Exact energies: E_n = (n + 1/2) * omega
# This tests our method with a non-trivial potential.

def solve_harmonic_oscillator(omega=1.0, N=200, x_max=8.0):
    """
    Solve the 1D harmonic oscillator numerically.

    Parameters
    ----------
    omega : float
        Angular frequency
    N : int
        Number of grid points
    x_max : float
        Grid extends from -x_max to +x_max

    Returns
    -------
    x, energies, wavefunctions
    """
    x = np.linspace(-x_max, x_max, N)
    dx = x[1] - x[0]

    # Kinetic energy (same finite difference as before)
    T = np.zeros((N, N))
    for i in range(N):
        T[i, i] = 1.0 / dx**2
        if i > 0:
            T[i, i-1] = -0.5 / dx**2
        if i < N-1:
            T[i, i+1] = -0.5 / dx**2

    # Potential energy: V(x) = 1/2 * omega^2 * x^2
    V = np.diag(0.5 * omega**2 * x**2)

    # Total Hamiltonian
    H = T + V

    energies, wavefunctions = eigh(H)

    return x, energies, wavefunctions


def plot_harmonic_oscillator():
    """Visualize harmonic oscillator solutions."""
    omega = 1.0
    x, energies, psi = solve_harmonic_oscillator(omega)

    fig, ax = plt.subplots(figsize=(10, 8))

    n_states = 5
    for n in range(n_states):
        E_n = energies[n]
        wf = psi[:, n]
        # Normalize for display
        wf = wf / np.max(np.abs(wf)) * 0.4

        # Plot potential + energy level
        ax.axhline(y=(n + 0.5) * omega, color='gray', linestyle='--', alpha=0.3)
        ax.plot(x, wf + E_n, linewidth=2, label=f'n={n}, E={(n+0.5)*omega:.1f}')

    # Plot the potential
    V = 0.5 * omega**2 * x**2
    ax.plot(x, V, 'k-', linewidth=2, label='V(x)')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.5, n_states + 1)
    ax.set_xlabel('x (bohr)')
    ax.set_ylabel('Energy (hartree)')
    ax.set_title('Quantum Harmonic Oscillator')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(_DIR, 'harmonic_oscillator.png'), dpi=150)
    plt.show()

    print("=" * 60)
    print("Harmonic Oscillator: Numerical vs Exact Energies")
    print("=" * 60)
    print(f"{'n':>3} {'E_exact':>15} {'E_numerical':>15} {'Error':>15}")
    print("-" * 60)
    for n in range(6):
        E_ex = (n + 0.5) * omega
        E_num = energies[n]
        print(f"{n:3d} {E_ex:15.8f} {E_num:15.8f} {abs(E_ex - E_num):15.2e}")


# ==========================================================================
# PART 3: Hydrogen Atom (Radial Equation)
# ==========================================================================
#
# The hydrogen atom is the ONLY atom with an exact analytical solution.
# In spherical coordinates, the radial equation is:
#
#   [-1/2 * d^2/dr^2 + l(l+1)/(2r^2) - Z/r] * u(r) = E * u(r)
#
# where u(r) = r * R(r), R(r) is the radial wavefunction.
#
# Exact energies: E_n = -Z^2 / (2*n^2)
#
# For hydrogen (Z=1):
#   E_1 = -0.5 hartree = -13.6 eV  (ground state)
#   E_2 = -0.125 hartree
#   E_3 = -0.0556 hartree

def solve_hydrogen_atom(Z=1, l=0, N=500, r_max=50.0):
    """
    Solve the radial Schrodinger equation for hydrogen-like atoms.

    We solve for u(r) = r*R(r) on a uniform grid, where:
        [-1/2 * d^2/dr^2 + l(l+1)/(2r^2) - Z/r] u(r) = E u(r)

    Boundary conditions: u(0) = 0, u(r_max) = 0

    Parameters
    ----------
    Z : int
        Nuclear charge
    l : int
        Angular momentum quantum number
    N : int
        Number of grid points
    r_max : float
        Maximum radius

    Returns
    -------
    r, energies, u(r) wavefunctions
    """
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)  # avoid r=0 singularity

    # Kinetic energy matrix
    T = np.zeros((N, N))
    for i in range(N):
        T[i, i] = 1.0 / dr**2
        if i > 0:
            T[i, i-1] = -0.5 / dr**2
        if i < N-1:
            T[i, i+1] = -0.5 / dr**2

    # Effective potential: V_eff = l(l+1)/(2r^2) - Z/r
    V_eff = l * (l + 1) / (2 * r**2) - Z / r
    V = np.diag(V_eff)

    H = T + V
    energies, wavefunctions = eigh(H)

    return r, energies, wavefunctions


def plot_hydrogen_atom():
    """Solve and visualize hydrogen atom wavefunctions."""
    Z = 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: Radial wavefunctions for l=0 ---
    ax = axes[0]
    r, energies, u = solve_hydrogen_atom(Z=Z, l=0, N=500, r_max=50)

    for n in range(1, 4):  # n = 1, 2, 3 for l=0 (1s, 2s, 3s)
        idx = n - 1
        R = u[:, idx] / r  # R(r) = u(r)/r
        R = R / np.max(np.abs(R))  # normalize for display

        # Exact 1s for comparison
        if n == 1:
            R_exact = 2 * Z**1.5 * np.exp(-Z * r)
            R_exact = R_exact / np.max(np.abs(R_exact))
            ax.plot(r, R_exact, 'k--', alpha=0.5, label='1s exact')

        ax.plot(r, R, linewidth=2, label=f'{n}s (E={energies[idx]:.4f})')

    ax.set_xlim(0, 30)
    ax.set_xlabel('r (bohr)')
    ax.set_ylabel('R(r) (normalized)')
    ax.set_title(f'Hydrogen Radial Wavefunctions (l=0)')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    # --- Right: Energy level comparison ---
    ax = axes[1]
    print("=" * 60)
    print("Hydrogen Atom: Numerical vs Exact Energies")
    print("=" * 60)
    print(f"{'n':>3} {'l':>3} {'E_exact':>15} {'E_numerical':>15} {'Error':>15}")
    print("-" * 60)

    colors = ['C0', 'C1', 'C2']
    for li, l in enumerate([0, 1, 2]):
        r, energies, u = solve_hydrogen_atom(Z=Z, l=l, N=500, r_max=50)

        for n in range(l + 1, l + 4):  # principal quantum number
            idx = n - l - 1
            if idx >= len(energies) or idx < 0:
                continue
            E_exact = -Z**2 / (2.0 * n**2)
            E_num = energies[idx]
            print(f"{n:3d} {l:3d} {E_exact:15.8f} {E_num:15.8f} {abs(E_exact - E_num):15.2e}")

            ax.plot(n + li*0.1 - 0.1, E_num, 'o', color=colors[li], markersize=10)
            ax.plot(n + li*0.1 - 0.1, E_exact, 'x', color='black', markersize=8)

    ax.set_xlabel('Principal quantum number n')
    ax.set_ylabel('Energy (hartree)')
    ax.set_title('H atom: o=numerical, x=exact')

    plt.tight_layout()
    plt.savefig(os.path.join(_DIR, 'hydrogen_atom.png'), dpi=150)
    plt.show()


# ==========================================================================
# MAIN: Run all demonstrations
# ==========================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("  STEP 1: SOLVING THE SCHRODINGER EQUATION NUMERICALLY")
    print("="*70)

    print("\n--- 1.1 Particle in a 1D Box ---\n")
    plot_particle_in_box()

    print("\n--- 1.2 Quantum Harmonic Oscillator ---\n")
    plot_harmonic_oscillator()

    print("\n--- 1.3 Hydrogen Atom ---\n")
    plot_hydrogen_atom()

    print("\n" + "="*70)
    print("  KEY TAKEAWAYS")
    print("="*70)
    print("""
    1. The Schrodinger equation H|psi> = E|psi> is an eigenvalue problem.

    2. We can discretize it on a grid using finite differences,
       turning it into a matrix eigenvalue problem H @ psi = E * psi.

    3. For 1 particle, this works beautifully. But for N electrons,
       the wavefunction psi(r1, r2, ..., rN) lives in 3N-dimensional space.
       This is the "curse of dimensionality" that motivates DFT.

    4. Next step: What happens with MANY electrons?
       --> Step 2: The Many-Body Problem
    """)
