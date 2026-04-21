"""
==============================================================================
STEP 3: Hartree-Fock Method
==============================================================================

The Hartree-Fock (HF) method is the foundation of molecular orbital theory
and provides the starting point for understanding DFT.

KEY IDEAS:
1. Approximate the N-electron wavefunction as a SINGLE Slater determinant
2. Each electron moves in the MEAN FIELD of all other electrons
3. Solve self-consistently (SCF)

THE SLATER DETERMINANT:
    Electrons are fermions --> wavefunction must be antisymmetric:
        Psi(r1, r2) = -Psi(r2, r1)

    For 2 electrons in orbitals phi_1, phi_2:
        Psi(r1, r2) = (1/sqrt(2)) * |phi_1(r1) phi_2(r1)|
                                     |phi_1(r2) phi_2(r2)|

    This automatically satisfies the Pauli exclusion principle!

THE HARTREE-FOCK EQUATIONS:
    f(i) |phi_a> = epsilon_a |phi_a>

    where f(i) is the FOCK OPERATOR:
        f(i) = h(i) + sum_b [J_b(i) - K_b(i)]

    h(i) = -1/2 nabla_i^2 - sum_A Z_A/|r_i - R_A|  (one-electron operator)
    J_b(i) = integral |phi_b(r')|^2 / |r-r'| dr'    (Coulomb operator)
    K_b(i) phi_a(r) = [integral phi_b*(r') phi_a(r') / |r-r'| dr'] phi_b(r)
                                                      (Exchange operator)

    The exchange operator K has NO classical analogue - it arises purely
    from the antisymmetry requirement (Pauli principle).

IN A BASIS SET (Roothaan-Hall equations):
    F C = S C epsilon

    F = Fock matrix
    S = overlap matrix
    C = MO coefficient matrix
    epsilon = orbital energies

TOTAL ENERGY:
    E_HF = sum_a epsilon_a - 1/2 sum_{a,b} (J_ab - K_ab) + V_nn

    or equivalently:
    E_HF = sum_a h_aa + 1/2 sum_{a,b} (2*J_ab - K_ab) + V_nn
==============================================================================
"""

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt


# ==========================================================================
# PART 1: 1D Hartree-Fock from scratch
# ==========================================================================
#
# We implement HF for electrons in a 1D potential using a real-space grid.
# This avoids basis set issues and shows the pure HF algorithm.

class HartreeFock1D:
    """
    1D Hartree-Fock solver on a real-space grid.

    Solves for N_elec electrons in an external potential V_ext(x).
    Uses a softened Coulomb interaction: 1/sqrt((x1-x2)^2 + eps^2)

    The SCF procedure:
    1. Guess initial orbitals (e.g., from non-interacting problem)
    2. Build Coulomb (J) and Exchange (K) potentials from current orbitals
    3. Build Fock operator: F = T + V_ext + J - K
    4. Diagonalize F to get new orbitals
    5. Check convergence; if not converged, go to step 2
    """

    def __init__(self, N_grid=100, x_max=10.0, N_elec=2, softening=0.5):
        self.N = N_grid
        self.x = np.linspace(-x_max, x_max, N_grid)
        self.dx = self.x[1] - self.x[0]
        self.N_elec = N_elec
        self.N_occ = N_elec // 2  # number of occupied spatial orbitals (RHF)
        self.softening = softening

        # Build kinetic energy matrix
        self.T = np.zeros((N_grid, N_grid))
        for i in range(N_grid):
            self.T[i, i] = 1.0 / self.dx**2
            if i > 0:
                self.T[i, i-1] = -0.5 / self.dx**2
            if i < N_grid - 1:
                self.T[i, i+1] = -0.5 / self.dx**2

        # Precompute softened Coulomb interaction on the grid
        # V_ee(x_i, x_j) = 1 / sqrt((x_i - x_j)^2 + eps^2)
        self.V_ee_matrix = np.zeros((N_grid, N_grid))
        for i in range(N_grid):
            for j in range(N_grid):
                r12 = np.sqrt((self.x[i] - self.x[j])**2 + self.softening**2)
                self.V_ee_matrix[i, j] = 1.0 / r12

    def set_potential(self, V_ext):
        """Set the external potential (e.g., nuclear attraction)."""
        self.V_ext = np.diag(V_ext)
        self.h_core = self.T + self.V_ext  # one-electron Hamiltonian

    def compute_coulomb(self, orbitals):
        """
        Compute the Coulomb potential (mean-field electron repulsion).

        J(x) = sum_a integral |phi_a(x')|^2 / |x - x'| dx'

        This is the classical electrostatic potential from the electron density.
        """
        J = np.zeros((self.N, self.N))
        for a in range(self.N_occ):
            phi_a = orbitals[:, a]
            # Density from orbital a: rho_a(x') = |phi_a(x')|^2
            rho_a = phi_a**2

            # J_a(x) = integral rho_a(x') / |x - x'| dx'
            # In matrix form for the grid representation
            v_J = self.V_ee_matrix @ (rho_a * self.dx)
            J += np.diag(v_J)

        return 2.0 * J  # factor 2 for spin (each spatial orbital has 2 electrons)

    def compute_exchange(self, orbitals):
        """
        Compute the Exchange potential.

        K_ab = integral phi_a*(x') phi_b(x') / |x - x'| dx' * phi_a(x)

        The exchange operator is NON-LOCAL: it depends on the orbital
        being acted upon. This is what makes HF different from DFT.
        K has no classical analogue - it's purely quantum mechanical.
        """
        K = np.zeros((self.N, self.N))
        for a in range(self.N_occ):
            phi_a = orbitals[:, a]
            # K_{ij} = phi_a(i) * [sum_x' phi_a(x') * V_ee(x_i, x') * dx] * delta
            # More precisely: K_{ij} = integral phi_a(x_i) phi_a(x_j) / |x_i - x_j| dx is wrong
            # K acts on an orbital phi_b: (K phi_b)(x) = sum_a [integral phi_a*(x')phi_b(x')/|x-x'| dx'] phi_a(x)
            # In grid representation:
            for i in range(self.N):
                for j in range(self.N):
                    K[i, j] += phi_a[i] * self.V_ee_matrix[i, j] * phi_a[j] * self.dx

        return K  # no factor of 2: exchange only between same-spin electrons

    def compute_density(self, orbitals):
        """Compute electron density from occupied orbitals."""
        rho = np.zeros(self.N)
        for a in range(self.N_occ):
            rho += 2.0 * orbitals[:, a]**2  # factor 2 for spin
        return rho

    def compute_energy(self, orbitals):
        """
        Compute total HF energy.

        E_HF = sum_a 2*h_aa + sum_{a,b} (2*J_ab - K_ab)
             = sum_a 2*epsilon_a - (J - 1/2 K)_total
             = Tr[D @ (h_core + F)]

        where D is the density matrix.
        """
        # One-electron energy
        E_1e = 0.0
        for a in range(self.N_occ):
            phi_a = orbitals[:, a]
            E_1e += 2.0 * (phi_a @ self.h_core @ phi_a) * self.dx

        # Two-electron energy
        J = self.compute_coulomb(orbitals)
        K = self.compute_exchange(orbitals)
        E_2e = 0.0
        for a in range(self.N_occ):
            phi_a = orbitals[:, a]
            # Coulomb contribution
            E_2e += (phi_a @ J @ phi_a) * self.dx
            # Exchange contribution (minus sign)
            E_2e -= (phi_a @ K @ phi_a) * self.dx

        # The two-electron part double-counts, so we take 1/2
        E_2e *= 0.5

        return E_1e + E_2e

    def solve(self, max_iter=100, tol=1e-7, mixing=0.3):
        """
        Run the Self-Consistent Field (SCF) procedure.

        Parameters
        ----------
        max_iter : int
            Maximum SCF iterations
        tol : float
            Convergence threshold for energy
        mixing : float
            Density mixing parameter (0 < mixing <= 1)
            new_orbs = mixing * new + (1-mixing) * old
        """
        print("=" * 60)
        print("Hartree-Fock SCF Procedure")
        print("=" * 60)

        # Step 1: Initial guess - solve non-interacting problem
        energies, orbitals = eigh(self.h_core)
        # Normalize
        for i in range(orbitals.shape[1]):
            norm = np.sqrt(np.sum(orbitals[:, i]**2) * self.dx)
            orbitals[:, i] /= norm

        E_old = 0.0
        E_history = []
        rho_old = self.compute_density(orbitals)

        print(f"\n{'Iter':>5} {'Energy':>18} {'dE':>15} {'Status':>10}")
        print("-" * 55)

        for iteration in range(max_iter):
            # Step 2: Build Fock operator
            J = self.compute_coulomb(orbitals)
            K = self.compute_exchange(orbitals)
            F = self.h_core + J - K

            # Step 3: Diagonalize Fock operator
            energies, new_orbitals = eigh(F)
            for i in range(new_orbitals.shape[1]):
                norm = np.sqrt(np.sum(new_orbitals[:, i]**2) * self.dx)
                new_orbitals[:, i] /= norm

            # Density mixing for stability
            rho_new = self.compute_density(new_orbitals)
            rho_mixed = mixing * rho_new + (1 - mixing) * rho_old

            # Reconstruct orbitals from mixed density is complex,
            # so we just use the new orbitals directly with energy check
            orbitals = new_orbitals
            rho_old = rho_mixed

            # Step 4: Compute energy
            E_total = self.compute_energy(orbitals)
            dE = abs(E_total - E_old)
            E_history.append(E_total)

            status = "converged" if dE < tol else ""
            print(f"{iteration:5d} {E_total:18.10f} {dE:15.2e} {status:>10}")

            if dE < tol and iteration > 0:
                print(f"\nSCF converged in {iteration} iterations!")
                break

            E_old = E_total
        else:
            print(f"\nWARNING: SCF did not converge in {max_iter} iterations")

        self.orbitals = orbitals
        self.orbital_energies = energies
        self.E_total = E_total
        self.E_history = E_history

        return E_total, orbitals, energies


# ==========================================================================
# PART 2: Demonstrate HF for model systems
# ==========================================================================

def demo_helium_1d():
    """
    Solve a 1D model of helium (2 electrons in a nuclear potential).

    V_ext(x) = -Z / sqrt(x^2 + eps^2)   (softened Coulomb from nucleus)
    """
    print("\n" + "=" * 70)
    print("  DEMO: 1D Helium Atom (2 electrons)")
    print("=" * 70)

    Z = 2  # nuclear charge
    hf = HartreeFock1D(N_grid=150, x_max=15.0, N_elec=2, softening=0.5)

    # Nuclear potential
    V_nuc = -Z / np.sqrt(hf.x**2 + 0.5**2)
    hf.set_potential(V_nuc)

    # Solve
    E_hf, orbitals, energies = hf.solve(max_iter=50, tol=1e-8, mixing=0.5)

    # Also solve non-interacting for comparison
    E_ni, psi_ni = eigh(hf.h_core)
    for i in range(psi_ni.shape[1]):
        norm = np.sqrt(np.sum(psi_ni[:, i]**2) * hf.dx)
        psi_ni[:, i] /= norm

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. SCF convergence
    ax = axes[0, 0]
    ax.plot(hf.E_history, 'bo-', linewidth=2)
    ax.set_xlabel('SCF Iteration')
    ax.set_ylabel('Total Energy (hartree)')
    ax.set_title('SCF Convergence')

    # 2. Orbitals
    ax = axes[0, 1]
    ax.plot(hf.x, orbitals[:, 0], 'b-', linewidth=2, label='HF orbital')
    ax.plot(hf.x, psi_ni[:, 0], 'r--', linewidth=2, label='Non-interacting')
    ax.set_xlabel('x (bohr)')
    ax.set_ylabel('phi(x)')
    ax.set_title('Ground State Orbital')
    ax.legend()

    # 3. Electron density
    ax = axes[1, 0]
    rho_hf = hf.compute_density(orbitals)
    rho_ni = 2 * psi_ni[:, 0]**2
    ax.plot(hf.x, rho_hf, 'b-', linewidth=2, label='HF density')
    ax.plot(hf.x, rho_ni, 'r--', linewidth=2, label='Non-interacting')
    ax.fill_between(hf.x, rho_hf, alpha=0.2)
    ax.set_xlabel('x (bohr)')
    ax.set_ylabel('rho(x)')
    ax.set_title('Electron Density')
    ax.legend()
    ax.text(0.95, 0.95, f'N_elec = {np.trapz(rho_hf, hf.x):.3f}',
            transform=ax.transAxes, ha='right', va='top')

    # 4. Potential landscape
    ax = axes[1, 1]
    ax.plot(hf.x, V_nuc, 'k-', linewidth=2, label='V_nuc')

    # Hartree potential
    v_H = np.zeros(hf.N)
    for i in range(hf.N):
        v_H[i] = np.sum(rho_hf * hf.V_ee_matrix[i, :]) * hf.dx
    ax.plot(hf.x, v_H, 'r-', linewidth=2, label='V_Hartree')
    ax.plot(hf.x, V_nuc + v_H, 'b-', linewidth=2, label='V_eff')
    ax.set_xlabel('x (bohr)')
    ax.set_ylabel('V(x) (hartree)')
    ax.set_title('Effective Potential')
    ax.legend()
    ax.set_ylim(-5, 3)

    plt.suptitle('1D Hartree-Fock: Helium Model', fontsize=14)
    plt.tight_layout()
    plt.savefig('step3_hartree_fock/hf_helium_1d.png', dpi=150)
    plt.show()

    print(f"\nHF total energy: {E_hf:.6f} hartree")
    print(f"Non-interacting energy: {2 * E_ni[0]:.6f} hartree")
    print(f"Electron-electron repulsion raises energy by: {E_hf - 2*E_ni[0]:.6f} hartree")


def demo_scf_components():
    """
    Illustrate the components of the Fock operator: h_core, J, K.

    This breaks down what each term does physically:
    - h_core: single electron in nuclear potential (simple)
    - J (Coulomb): classical electrostatic repulsion from electron cloud
    - K (Exchange): quantum mechanical, no classical analogue!
    """
    print("\n" + "=" * 70)
    print("  DEMO: Components of the Fock Operator")
    print("=" * 70)

    hf = HartreeFock1D(N_grid=150, x_max=15.0, N_elec=2, softening=0.5)
    V_nuc = -2.0 / np.sqrt(hf.x**2 + 0.5**2)
    hf.set_potential(V_nuc)
    E_hf, orbitals, _ = hf.solve(max_iter=50, tol=1e-8, mixing=0.5)

    # Extract diagonal of J and K matrices (local part for visualization)
    J = hf.compute_coulomb(orbitals)
    K = hf.compute_exchange(orbitals)

    J_diag = np.diag(J)
    K_diag = np.diag(K)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(hf.x, V_nuc, 'k-', linewidth=2, label='V_nuclear (attraction)')
    ax.plot(hf.x, J_diag, 'r-', linewidth=2, label='J (Coulomb repulsion)')
    ax.plot(hf.x, -K_diag, 'b-', linewidth=2, label='-K (Exchange, stabilizing)')
    ax.plot(hf.x, V_nuc + J_diag - K_diag, 'g--', linewidth=2, label='V_eff = V_nuc + J - K')

    ax.set_xlabel('x (bohr)', fontsize=12)
    ax.set_ylabel('Potential (hartree)', fontsize=12)
    ax.set_title('Fock Operator Components', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(-5, 3)

    plt.tight_layout()
    plt.savefig('step3_hartree_fock/fock_components.png', dpi=150)
    plt.show()

    print("""
    Physical interpretation:
    - V_nuclear: attracts electrons to nucleus (negative, stabilizing)
    - J (Coulomb): repulsion from electron cloud (positive, destabilizing)
    - K (Exchange): same-spin electrons avoid each other (stabilizing!)
      This is the Pauli exclusion principle in action.

    HF captures EXCHANGE exactly but misses CORRELATION entirely.
    The correlation energy is defined as:
        E_corr = E_exact - E_HF

    DFT aims to capture BOTH exchange and correlation through
    the exchange-correlation functional E_xc[rho].
    """)


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("  STEP 3: HARTREE-FOCK METHOD")
    print("="*70)

    print("\n--- 3.1 HF for 1D Helium ---\n")
    demo_helium_1d()

    print("\n--- 3.2 Fock Operator Components ---\n")
    demo_scf_components()

    print("\n" + "="*70)
    print("  KEY TAKEAWAYS")
    print("="*70)
    print("""
    1. Hartree-Fock approximates the wavefunction as a single Slater
       determinant, reducing the many-body problem to single-particle
       equations solved self-consistently (SCF).

    2. The Fock operator: F = h + J - K
       - h: one-electron (kinetic + nuclear attraction)
       - J: Coulomb repulsion (classical, local)
       - K: Exchange (quantum, NON-LOCAL) -- exact in HF

    3. HF captures ~99% of the total energy but misses CORRELATION.
       The missing correlation energy is small but chemically important
       (bond energies, reaction barriers, etc.)

    4. Two paths forward from HF:
       a) Post-HF methods (MP2, CCSD(T), ...) -- systematic but expensive
       b) DFT -- replace exchange+correlation with a functional of density

    Next: The theoretical foundation of DFT
    --> Step 4: Hohenberg-Kohn Theorems
    """)
