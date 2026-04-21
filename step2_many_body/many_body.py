"""
==============================================================================
STEP 2: The Many-Body Problem & Born-Oppenheimer Approximation
==============================================================================

For a system of N_e electrons and N_n nuclei, the full Hamiltonian is:

    H_total = T_n + T_e + V_nn + V_en + V_ee

where:
    T_n  = -sum_A 1/(2*M_A) * nabla_A^2     (nuclear kinetic energy)
    T_e  = -sum_i 1/2 * nabla_i^2            (electron kinetic energy)
    V_nn = sum_{A>B} Z_A*Z_B / |R_A - R_B|   (nuclear-nuclear repulsion)
    V_en = -sum_{i,A} Z_A / |r_i - R_A|      (electron-nuclear attraction)
    V_ee = sum_{i>j} 1/|r_i - r_j|           (electron-electron repulsion)

The BORN-OPPENHEIMER APPROXIMATION:
    Since nuclei are ~2000x heavier than electrons, they move much slower.
    We can separate the problem:
    1. Fix nuclear positions R = {R_A}
    2. Solve the ELECTRONIC Schrodinger equation at fixed R
    3. Use the electronic energy E(R) as a potential surface for nuclear motion

    The electronic Hamiltonian (what DFT actually solves):
        H_elec = T_e + V_en + V_ee
        H_elec |Psi> = E_elec |Psi>

THE FUNDAMENTAL PROBLEM:
    Psi(r1, r2, ..., rN) is a function of 3N variables.
    For N=100 electrons on a grid of 10 points per dimension:
        Storage needed = 10^300 numbers. More than atoms in the universe!

    This is WHY we need approximations like Hartree-Fock and DFT.
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from itertools import combinations


# ==========================================================================
# PART 1: Two electrons in a 1D box (exact diagonalization)
# ==========================================================================
#
# The simplest system that captures electron-electron interaction.
# Two electrons in a 1D box with Coulomb-like repulsion.
#
# H = -1/2 d^2/dx1^2 - 1/2 d^2/dx2^2 + V_ee(x1, x2)
#
# The wavefunction Psi(x1, x2) lives in 2D space.
# We'll see how the Hilbert space grows exponentially.

def solve_two_electrons_1d(N=40, L=1.0, interaction_strength=1.0):
    """
    Solve two interacting electrons in a 1D box exactly.

    This demonstrates:
    1. How the Hilbert space scales (N^2 for 2 particles)
    2. How electron-electron repulsion changes the physics
    3. Why exact methods fail for many electrons

    The Hamiltonian in the product basis {phi_i(x1) * phi_j(x2)} is:

        H = T1 (x) I + I (x) T2 + V_ee

    where (x) denotes Kronecker product.

    Parameters
    ----------
    N : int
        Grid points per electron (total basis = N^2)
    L : float
        Box length
    interaction_strength : float
        Prefactor for 1/|x1-x2| repulsion (0 = non-interacting)
    """
    dx = L / (N + 1)
    x = np.linspace(dx, L - dx, N)

    # Single-particle kinetic energy matrix
    T_1d = np.zeros((N, N))
    for i in range(N):
        T_1d[i, i] = 1.0 / dx**2
        if i > 0:
            T_1d[i, i-1] = -0.5 / dx**2
        if i < N-1:
            T_1d[i, i+1] = -0.5 / dx**2

    # Two-particle Hamiltonian in product space (N^2 x N^2)
    I = np.eye(N)
    T = np.kron(T_1d, I) + np.kron(I, T_1d)  # T1 + T2

    # Electron-electron repulsion: V_ee = lambda / |x1 - x2|
    # We use a softened Coulomb to avoid the singularity at x1=x2
    V_ee = np.zeros((N**2, N**2))
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            # Softened Coulomb: 1/sqrt((x1-x2)^2 + epsilon^2)
            r12 = np.sqrt((x[i] - x[j])**2 + 0.01)
            V_ee[idx, idx] = interaction_strength / r12

    H = T + V_ee

    print(f"Hilbert space dimension: {N}^2 = {N**2}")
    print(f"Hamiltonian matrix size: {H.shape}")
    print(f"Memory for H: {H.nbytes / 1e6:.1f} MB")

    energies, states = eigh(H)

    return x, energies, states


def demonstrate_scaling():
    """
    Show how the Hilbert space grows exponentially with particle number.

    This is THE fundamental motivation for DFT.
    """
    print("=" * 70)
    print("THE EXPONENTIAL WALL: Why exact methods fail")
    print("=" * 70)
    print()
    print(f"{'N_electrons':>12} {'Grid pts/dim':>14} {'Hilbert dim':>20} {'Memory (GB)':>15}")
    print("-" * 65)

    grid_pts = 20  # modest grid
    for n_elec in range(1, 8):
        dim = grid_pts ** n_elec
        # Each complex number = 16 bytes
        memory_gb = dim * 16 / 1e9
        dim_str = f"{dim:.2e}" if dim > 1e6 else str(dim)
        mem_str = f"{memory_gb:.2e}" if memory_gb > 1e3 else f"{memory_gb:.2f}"
        print(f"{n_elec:12d} {grid_pts:14d} {dim_str:>20} {mem_str:>15}")

    print()
    print("For N=10 electrons with 20 grid points per dimension:")
    print(f"  Hilbert space dimension = 20^10 = {20**10:.2e}")
    print(f"  Memory needed = {20**10 * 16 / 1e9:.2e} GB")
    print(f"  (More than all hard drives on Earth combined!)")
    print()
    print("This is why we CANNOT solve the Schrodinger equation exactly")
    print("for more than a few electrons. We need approximations.")
    print("  --> Hartree-Fock: mean-field approximation (Step 3)")
    print("  --> DFT: use density rho(r) instead of Psi(r1,...,rN) (Step 4-5)")


# ==========================================================================
# PART 2: Visualize electron correlation
# ==========================================================================

def visualize_correlation():
    """
    Compare non-interacting vs interacting two-electron wavefunctions.

    Key insight: electron-electron repulsion creates CORRELATION -
    the electrons avoid each other. This correlation is what makes
    quantum chemistry hard and is central to DFT.
    """
    N = 40
    L = 5.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (lam, title) in enumerate([
        (0.0, 'Non-interacting (V_ee=0)'),
        (1.0, 'Weakly interacting'),
        (5.0, 'Strongly interacting'),
    ]):
        x, energies, states = solve_two_electrons_1d(N, L, lam)

        # Ground state wavefunction |Psi(x1, x2)|^2
        psi_gs = states[:, 0].reshape(N, N)
        density = np.abs(psi_gs)**2

        ax = axes[idx]
        im = ax.imshow(density, extent=[0, L, 0, L], origin='lower',
                       cmap='hot', aspect='equal')
        ax.plot([0, L], [0, L], 'w--', alpha=0.5, label='x1=x2')
        ax.set_xlabel('x1 (bohr)')
        ax.set_ylabel('x2 (bohr)')
        ax.set_title(f'{title}\nE_0 = {energies[0]:.4f}')
        plt.colorbar(im, ax=ax, label='|Psi(x1,x2)|^2')
        ax.legend()

    plt.suptitle('Two Electrons in a Box: Effect of Electron-Electron Repulsion',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('step2_many_body/electron_correlation.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("""
    OBSERVATION:
    - Non-interacting: |Psi|^2 is symmetric, electrons don't care about each other
    - Interacting: |Psi|^2 is depleted along x1=x2 diagonal
      --> Electrons AVOID each other due to Coulomb repulsion
      --> This is "electron correlation"

    In DFT, capturing this correlation through the exchange-correlation
    functional is the central challenge.
    """)


# ==========================================================================
# PART 3: Born-Oppenheimer Approximation demonstration
# ==========================================================================

def born_oppenheimer_demo():
    """
    Demonstrate the Born-Oppenheimer approximation for H2+ (simplest molecule).

    H2+ = 2 protons + 1 electron

    1. Fix proton separation R
    2. Solve for electronic energy E_elec(R)
    3. E_total(R) = E_elec(R) + V_nn(R) gives the potential energy surface
    4. The minimum of E_total(R) gives the bond length
    """
    print("=" * 70)
    print("Born-Oppenheimer: H2+ Potential Energy Surface")
    print("=" * 70)

    N = 200
    x_max = 20.0

    R_values = np.linspace(0.5, 10.0, 40)  # proton-proton distances
    E_total = []
    E_electronic = []

    for R in R_values:
        dx = 2 * x_max / (N + 1)
        x = np.linspace(-x_max + dx, x_max - dx, N)

        # Kinetic energy
        T = np.zeros((N, N))
        for i in range(N):
            T[i, i] = 1.0 / dx**2
            if i > 0:
                T[i, i-1] = -0.5 / dx**2
            if i < N-1:
                T[i, i+1] = -0.5 / dx**2

        # Electron-nuclear attraction (two protons at +/- R/2)
        V_en = -1.0 / np.sqrt((x - R/2)**2 + 0.01) \
               -1.0 / np.sqrt((x + R/2)**2 + 0.01)

        H = T + np.diag(V_en)
        energies, _ = eigh(H)

        E_elec = energies[0]
        V_nn = 1.0 / R  # proton-proton repulsion
        E_tot = E_elec + V_nn

        E_electronic.append(E_elec)
        E_total.append(E_tot)

    E_total = np.array(E_total)
    E_electronic = np.array(E_electronic)

    # Find equilibrium bond length
    idx_min = np.argmin(E_total)
    R_eq = R_values[idx_min]
    E_eq = E_total[idx_min]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(R_values, E_electronic, 'b--', label='E_electronic(R)', linewidth=2)
    ax.plot(R_values, 1.0/R_values, 'r--', label='V_nn = 1/R', linewidth=2)
    ax.plot(R_values, E_total, 'k-', label='E_total(R)', linewidth=3)
    ax.plot(R_eq, E_eq, 'ro', markersize=12, label=f'R_eq = {R_eq:.2f} bohr')
    ax.axhline(y=E_total[-1], color='gray', linestyle=':', alpha=0.5,
               label='Dissociation limit')

    ax.set_xlabel('R (bohr)', fontsize=12)
    ax.set_ylabel('Energy (hartree)', fontsize=12)
    ax.set_title('Born-Oppenheimer Potential Energy Surface: H2+', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0.5, 10)

    plt.tight_layout()
    plt.savefig('step2_many_body/born_oppenheimer_h2plus.png', dpi=150)
    plt.show()

    print(f"\nEquilibrium bond length: R_eq = {R_eq:.2f} bohr")
    print(f"Equilibrium energy: E = {E_eq:.4f} hartree")
    print(f"(Exact H2+ bond length: ~2.0 bohr)")
    print(f"\nThe Born-Oppenheimer approximation:")
    print(f"  1. Fix nuclei at positions R")
    print(f"  2. Solve electronic problem --> E_elec(R)")
    print(f"  3. Total energy = E_elec(R) + V_nn(R)")
    print(f"  4. This gives the Potential Energy Surface (PES)")
    print(f"  5. Minimum of PES = equilibrium geometry")


# ==========================================================================
# PART 4: Electron density - preview of DFT's key variable
# ==========================================================================

def density_preview():
    """
    Show that the electron density rho(r) contains useful information
    even though it lives in 3D (not 3N-D like the wavefunction).

    rho(r) = N * integral |Psi(r, r2, ..., rN)|^2 dr2...drN

    For our 1D two-electron system:
    rho(x) = 2 * integral |Psi(x, x2)|^2 dx2

    Key insight: rho(r) is a 3D function regardless of N!
    """
    N = 50
    L = 5.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (lam, label) in enumerate([(0.0, 'Non-interacting'), (3.0, 'Interacting')]):
        x, energies, states = solve_two_electrons_1d(N, L, lam)
        dx = x[1] - x[0]

        psi_gs = states[:, 0].reshape(N, N)

        # Compute density: rho(x1) = 2 * integral |Psi(x1, x2)|^2 dx2
        # (factor 2 because both electrons contribute)
        rho = 2 * np.sum(np.abs(psi_gs)**2, axis=1) * dx

        axes[idx].plot(x, rho, 'b-', linewidth=2)
        axes[idx].fill_between(x, rho, alpha=0.3)
        axes[idx].set_xlabel('x (bohr)')
        axes[idx].set_ylabel('rho(x)')
        axes[idx].set_title(f'{label}\nTotal electrons = {np.trapz(rho, x):.3f}')

    plt.suptitle('Electron Density: From Wavefunction to Density',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('step2_many_body/electron_density_preview.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("""
    KEY INSIGHT for DFT:
    - Wavefunction Psi(r1,...,rN) lives in 3N dimensions --> intractable
    - Electron density rho(r) lives in 3 dimensions --> manageable!
    - Hohenberg-Kohn will prove: rho(r) contains ALL ground state info
    - This is the foundation of DFT (Step 4)
    """)


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("  STEP 2: THE MANY-BODY PROBLEM")
    print("="*70)

    print("\n--- 2.1 The Exponential Wall ---\n")
    demonstrate_scaling()

    print("\n--- 2.2 Electron Correlation ---\n")
    visualize_correlation()

    print("\n--- 2.3 Born-Oppenheimer Approximation ---\n")
    born_oppenheimer_demo()

    print("\n--- 2.4 Electron Density Preview ---\n")
    density_preview()
