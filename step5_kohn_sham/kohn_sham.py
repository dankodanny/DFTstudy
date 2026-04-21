"""
==============================================================================
STEP 5: Kohn-Sham Equations and SCF Implementation
==============================================================================

The Kohn-Sham (KS) approach (1965) is the practical implementation of DFT.
It maps the interacting many-electron problem to a non-interacting system
that gives the SAME density.

THE KOHN-SHAM ANSATZ:
    Replace the interacting system with a FICTITIOUS non-interacting system
    that has the same ground-state density rho(r).

    The non-interacting electrons satisfy single-particle equations:

    [-1/2 nabla^2 + V_KS(r)] phi_i(r) = epsilon_i phi_i(r)

    where the Kohn-Sham effective potential is:

    V_KS(r) = V_ext(r) + V_H(r) + V_xc(r)

    V_ext(r) = -sum_A Z_A / |r - R_A|      (nuclear potential)
    V_H(r)   = integral rho(r')/|r-r'| dr'  (Hartree potential)
    V_xc(r)  = delta E_xc[rho] / delta rho(r)  (XC potential)

    The density is constructed from occupied KS orbitals:
    rho(r) = sum_{i=1}^{N/2} 2 * |phi_i(r)|^2

THE SELF-CONSISTENT FIELD (SCF) PROCEDURE:
    1. Guess initial density rho^(0)(r)
    2. Construct V_KS[rho] = V_ext + V_H[rho] + V_xc[rho]
    3. Solve KS equations: [-1/2 nabla^2 + V_KS] phi_i = epsilon_i phi_i
    4. Compute new density: rho^(new)(r) = sum_i 2|phi_i|^2
    5. Mix densities: rho^(n+1) = alpha * rho^(new) + (1-alpha) * rho^(n)
    6. If converged, stop. Otherwise go to 2.

TOTAL ENERGY:
    E = sum_i epsilon_i - E_H[rho] + E_xc[rho] - integral V_xc*rho dr
      = T_s + E_ext + E_H + E_xc

This file implements a COMPLETE Kohn-Sham DFT solver from scratch
using only NumPy/SciPy.
==============================================================================
"""

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt


class KohnShamDFT1D:
    """
    Complete 1D Kohn-Sham DFT solver.

    This implements the full KS-DFT algorithm on a real-space grid:
    - Kinetic energy via finite differences
    - Hartree potential via direct integration
    - Exchange-correlation via LDA (local density approximation)
    - Self-consistent field iteration with density mixing

    This is a minimal but COMPLETE DFT implementation!
    """

    def __init__(self, N_grid=200, x_max=15.0, N_elec=2, softening=0.1):
        """
        Parameters
        ----------
        N_grid : int
            Number of grid points
        x_max : float
            Grid extends from -x_max to x_max
        N_elec : int
            Number of electrons (must be even for RKS)
        softening : float
            Coulomb softening parameter to avoid 1D singularity
        """
        self.N = N_grid
        self.x = np.linspace(-x_max, x_max, N_grid)
        self.dx = self.x[1] - self.x[0]
        self.N_elec = N_elec
        self.N_occ = N_elec // 2  # occupied spatial orbitals
        self.softening = softening

        # Kinetic energy matrix: T = -1/2 d^2/dx^2
        self.T = np.zeros((N_grid, N_grid))
        for i in range(N_grid):
            self.T[i, i] = 1.0 / self.dx**2
            if i > 0:
                self.T[i, i-1] = -0.5 / self.dx**2
            if i < N_grid - 1:
                self.T[i, i+1] = -0.5 / self.dx**2

        # Precompute Coulomb kernel
        self.coulomb_kernel = np.zeros((N_grid, N_grid))
        for i in range(N_grid):
            for j in range(N_grid):
                r = np.sqrt((self.x[i] - self.x[j])**2 + softening**2)
                self.coulomb_kernel[i, j] = 1.0 / r

    def set_nuclear_potential(self, Z_list, R_list):
        """
        Set nuclear positions and charges.

        Parameters
        ----------
        Z_list : list of float
            Nuclear charges
        R_list : list of float
            Nuclear positions (1D)
        """
        self.Z_list = Z_list
        self.R_list = R_list

        self.V_ext = np.zeros(self.N)
        for Z, R in zip(Z_list, R_list):
            self.V_ext -= Z / np.sqrt((self.x - R)**2 + self.softening**2)

        self.V_nn = 0.0  # nuclear-nuclear repulsion
        for i in range(len(Z_list)):
            for j in range(i+1, len(Z_list)):
                R_ij = abs(R_list[i] - R_list[j])
                self.V_nn += Z_list[i] * Z_list[j] / R_ij

    def compute_hartree_potential(self, rho):
        """
        Compute the Hartree potential: V_H(x) = integral rho(x')/|x-x'| dx'

        This is the classical electrostatic potential from the electron density.
        """
        V_H = self.coulomb_kernel @ (rho * self.dx)
        return V_H

    def compute_hartree_energy(self, rho):
        """E_H = 1/2 integral integral rho(r)*rho(r')/|r-r'| dr dr'"""
        V_H = self.compute_hartree_potential(rho)
        return 0.5 * np.trapz(rho * V_H, self.x)

    # ------------------------------------------------------------------
    # LDA Exchange-Correlation (the simplest approximation)
    # ------------------------------------------------------------------
    # In 3D, the LDA exchange energy density is:
    #   e_x(rho) = -C_x * rho^(4/3),  C_x = 3/4 * (3/pi)^(1/3)
    #
    # For 1D systems, the exact exchange energy is different.
    # We use a 1D-adapted LDA:
    #   e_x(rho) = -C_1d * rho^2  (1D exchange)
    #   V_x(rho) = d(rho * e_x)/d(rho)
    #
    # For correlation, we use a simple parametrization.

    def lda_exchange(self, rho):
        """
        1D LDA exchange energy density and potential.

        For 1D: e_x = -pi/4 * rho  (per electron)
        E_x = integral rho * e_x dx = -pi/4 * integral rho^2 dx
        V_x = d(rho * e_x)/d rho = -pi/2 * rho
        """
        C_x = np.pi / 4.0
        safe_rho = np.maximum(rho, 1e-20)

        e_x = -C_x * safe_rho  # energy density (per electron)
        V_x = -2 * C_x * safe_rho  # exchange potential

        return e_x, V_x

    def lda_correlation(self, rho):
        """
        Simple 1D LDA correlation.

        We use a Chachiyo-inspired parametrization adapted for 1D:
        e_c(r_s) = a * ln(1 + b/r_s + b/r_s^2)

        where r_s = 1/(2*rho) is the 1D Wigner-Seitz radius.
        """
        safe_rho = np.maximum(rho, 1e-20)
        r_s = 1.0 / (2.0 * safe_rho)

        a = -0.04
        b = 1.0

        e_c = a * np.log(1.0 + b / r_s + b / r_s**2)

        # Potential: V_c = e_c + rho * de_c/drho
        # de_c/drho = de_c/dr_s * dr_s/drho = de_c/dr_s * (-1/(2*rho^2))
        de_c_drs = a * (-b/r_s**2 - 2*b/r_s**3) / (1.0 + b/r_s + b/r_s**2)
        drs_drho = -1.0 / (2.0 * safe_rho**2)
        V_c = e_c + safe_rho * de_c_drs * drs_drho

        return e_c, V_c

    def compute_xc(self, rho):
        """
        Compute total XC energy and potential.

        E_xc = integral rho(r) * e_xc(rho(r)) dr
        V_xc = d E_xc / d rho = e_xc + rho * de_xc/drho
        """
        e_x, V_x = self.lda_exchange(rho)
        e_c, V_c = self.lda_correlation(rho)

        e_xc = e_x + e_c
        V_xc = V_x + V_c
        E_xc = np.trapz(rho * e_xc, self.x)

        return E_xc, V_xc

    def compute_total_energy(self, orbitals, eigenvalues, rho):
        """
        Compute the total energy.

        E_total = sum_i f_i * epsilon_i    (band structure energy)
                  - E_H[rho]               (remove double-counting)
                  + E_xc[rho]              (add XC energy)
                  - integral V_xc*rho dr   (remove double-counting of XC)
                  + V_nn                   (nuclear repulsion)
        """
        # Band energy
        E_band = 2.0 * np.sum(eigenvalues[:self.N_occ])

        # Hartree energy (double-counting correction)
        E_H = self.compute_hartree_energy(rho)

        # XC energy and potential
        E_xc, V_xc = self.compute_xc(rho)
        E_vxc = np.trapz(rho * V_xc, self.x)

        E_total = E_band - E_H + E_xc - E_vxc + self.V_nn

        return E_total, {'E_band': E_band, 'E_H': E_H, 'E_xc': E_xc,
                         'E_vxc': E_vxc, 'V_nn': self.V_nn}

    def solve(self, max_iter=200, tol=1e-8, mixing=0.3, verbose=True):
        """
        Run the Kohn-Sham SCF procedure.

        This is the HEART of DFT:
        1. Guess density -> 2. Build V_KS -> 3. Solve KS eqs ->
        4. New density -> 5. Mix -> 6. Check convergence -> repeat

        Parameters
        ----------
        max_iter : int
            Maximum SCF iterations
        tol : float
            Energy convergence threshold (hartree)
        mixing : float
            Linear mixing parameter (0 < mixing <= 1)
        verbose : bool
            Print SCF progress
        """
        if verbose:
            print("=" * 65)
            print("  Kohn-Sham DFT: Self-Consistent Field Procedure")
            print("=" * 65)
            print(f"  Grid: {self.N} points, dx = {self.dx:.4f} bohr")
            print(f"  Electrons: {self.N_elec}")
            print(f"  XC functional: LDA (1D)")
            print(f"  Mixing: {mixing}")
            print("=" * 65)

        # Initial guess: solve non-interacting problem
        H_0 = self.T + np.diag(self.V_ext)
        eigenvalues, orbitals = eigh(H_0)
        for i in range(orbitals.shape[1]):
            orbitals[:, i] /= np.sqrt(np.sum(orbitals[:, i]**2) * self.dx)

        rho = self.compute_density(orbitals)
        E_old = 0.0
        E_history = []
        rho_history = [rho.copy()]

        if verbose:
            print(f"\n{'Iter':>5} {'Energy':>18} {'dE':>14} {'|drho|':>14}")
            print("-" * 55)

        for it in range(max_iter):
            # Build KS effective potential
            V_H = self.compute_hartree_potential(rho)
            _, V_xc = self.compute_xc(rho)
            V_KS = self.V_ext + V_H + V_xc

            # Solve KS equations
            H_KS = self.T + np.diag(V_KS)
            eigenvalues, orbitals = eigh(H_KS)
            for i in range(orbitals.shape[1]):
                orbitals[:, i] /= np.sqrt(np.sum(orbitals[:, i]**2) * self.dx)

            # New density
            rho_new = self.compute_density(orbitals)

            # Density mixing
            rho_mixed = mixing * rho_new + (1.0 - mixing) * rho
            drho = np.sqrt(np.trapz((rho_mixed - rho)**2, self.x))

            rho = rho_mixed

            # Compute energy
            E_total, E_components = self.compute_total_energy(
                orbitals, eigenvalues, rho)
            dE = abs(E_total - E_old)
            E_history.append(E_total)
            rho_history.append(rho.copy())

            if verbose:
                print(f"{it:5d} {E_total:18.10f} {dE:14.2e} {drho:14.2e}")

            if dE < tol and it > 0:
                if verbose:
                    print(f"\n  SCF CONVERGED in {it} iterations!")
                break

            E_old = E_total
        else:
            if verbose:
                print(f"\n  WARNING: SCF did not converge in {max_iter} iterations")

        # Store results
        self.orbitals = orbitals
        self.eigenvalues = eigenvalues
        self.rho = rho
        self.E_total = E_total
        self.E_components = E_components
        self.E_history = E_history
        self.rho_history = rho_history

        if verbose:
            self.print_results()

        return E_total

    def compute_density(self, orbitals):
        """rho(x) = 2 * sum_{i occupied} |phi_i(x)|^2"""
        rho = np.zeros(self.N)
        for i in range(self.N_occ):
            rho += 2.0 * orbitals[:, i]**2
        return rho

    def print_results(self):
        """Print final energy decomposition."""
        ec = self.E_components
        print(f"\n  === Energy Decomposition ===")
        print(f"  Band energy (sum epsilon_i):  {ec['E_band']:16.8f}")
        print(f"  Hartree energy E_H:           {ec['E_H']:16.8f}")
        print(f"  XC energy E_xc:               {ec['E_xc']:16.8f}")
        print(f"  XC potential energy:           {ec['E_vxc']:16.8f}")
        print(f"  Nuclear repulsion V_nn:        {ec['V_nn']:16.8f}")
        print(f"  -----------------------------------------")
        print(f"  TOTAL ENERGY:                  {self.E_total:16.8f} hartree")
        print(f"  Orbital energies: {self.eigenvalues[:self.N_occ+2]}")


# ==========================================================================
# DEMONSTRATIONS
# ==========================================================================

def demo_helium_atom():
    """DFT calculation for a 1D helium atom model."""
    print("\n" + "="*70)
    print("  DEMO: 1D Helium Atom with Kohn-Sham DFT")
    print("="*70)

    dft = KohnShamDFT1D(N_grid=200, x_max=15.0, N_elec=2, softening=0.5)
    dft.set_nuclear_potential(Z_list=[2.0], R_list=[0.0])

    E = dft.solve(max_iter=100, tol=1e-9, mixing=0.4)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. SCF convergence
    ax = axes[0, 0]
    ax.semilogy(np.abs(np.diff(dft.E_history)), 'bo-')
    ax.set_xlabel('SCF Iteration')
    ax.set_ylabel('|dE| (hartree)')
    ax.set_title('SCF Convergence')

    # 2. KS orbital
    ax = axes[0, 1]
    ax.plot(dft.x, dft.orbitals[:, 0], 'b-', linewidth=2, label='phi_1 (occupied)')
    ax.plot(dft.x, dft.orbitals[:, 1], 'r--', linewidth=2, label='phi_2 (virtual)')
    ax.set_xlabel('x (bohr)')
    ax.set_ylabel('phi(x)')
    ax.set_title('Kohn-Sham Orbitals')
    ax.legend()

    # 3. Electron density
    ax = axes[1, 0]
    ax.plot(dft.x, dft.rho, 'b-', linewidth=2)
    ax.fill_between(dft.x, dft.rho, alpha=0.3)
    ax.set_xlabel('x (bohr)')
    ax.set_ylabel('rho(x)')
    ax.set_title(f'Electron Density (N={np.trapz(dft.rho, dft.x):.3f})')

    # 4. Potentials
    ax = axes[1, 1]
    V_H = dft.compute_hartree_potential(dft.rho)
    _, V_xc = dft.compute_xc(dft.rho)
    V_KS = dft.V_ext + V_H + V_xc

    ax.plot(dft.x, dft.V_ext, 'k-', linewidth=1.5, label='V_ext (nuclear)')
    ax.plot(dft.x, V_H, 'r-', linewidth=1.5, label='V_H (Hartree)')
    ax.plot(dft.x, V_xc, 'g-', linewidth=1.5, label='V_xc (XC)')
    ax.plot(dft.x, V_KS, 'b--', linewidth=2, label='V_KS (total)')
    ax.set_xlabel('x (bohr)')
    ax.set_ylabel('V(x) (hartree)')
    ax.set_title('Kohn-Sham Effective Potential')
    ax.legend()
    ax.set_ylim(-6, 4)

    plt.suptitle(f'Kohn-Sham DFT: 1D Helium (E = {E:.6f} Ha)', fontsize=14)
    plt.tight_layout()
    plt.savefig('step5_kohn_sham/ks_helium.png', dpi=150)
    plt.show()


def demo_h2_molecule():
    """DFT calculation for a 1D H2 molecule - compute binding curve."""
    print("\n" + "="*70)
    print("  DEMO: 1D H2 Molecule Binding Curve with KS-DFT")
    print("="*70)

    R_values = np.linspace(0.5, 8.0, 30)
    E_values = []

    # Also compute atomic energy for binding energy
    dft_atom = KohnShamDFT1D(N_grid=200, x_max=15.0, N_elec=2, softening=0.5)
    dft_atom.set_nuclear_potential(Z_list=[1.0], R_list=[0.0])
    E_atom = dft_atom.solve(verbose=False)

    for R in R_values:
        dft = KohnShamDFT1D(N_grid=200, x_max=15.0, N_elec=2, softening=0.5)
        dft.set_nuclear_potential(Z_list=[1.0, 1.0], R_list=[-R/2, R/2])
        E = dft.solve(max_iter=150, tol=1e-8, mixing=0.3, verbose=False)
        E_values.append(E)

    E_values = np.array(E_values)
    E_binding = E_values - E_atom  # binding energy relative to H atom with 2e

    # Find equilibrium
    idx_min = np.argmin(E_values)
    R_eq = R_values[idx_min]
    E_eq = E_values[idx_min]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Total energy
    ax = axes[0]
    ax.plot(R_values, E_values, 'b-', linewidth=2)
    ax.plot(R_eq, E_eq, 'ro', markersize=10)
    ax.set_xlabel('Bond length R (bohr)')
    ax.set_ylabel('Total Energy (hartree)')
    ax.set_title(f'H2 Potential Energy Curve\nR_eq = {R_eq:.2f} bohr, E = {E_eq:.4f} Ha')

    # Right: Density at equilibrium
    ax = axes[1]
    dft_eq = KohnShamDFT1D(N_grid=200, x_max=15.0, N_elec=2, softening=0.5)
    dft_eq.set_nuclear_potential(Z_list=[1.0, 1.0], R_list=[-R_eq/2, R_eq/2])
    dft_eq.solve(max_iter=100, tol=1e-9, mixing=0.3, verbose=False)

    ax.plot(dft_eq.x, dft_eq.rho, 'b-', linewidth=2)
    ax.fill_between(dft_eq.x, dft_eq.rho, alpha=0.3)
    ax.axvline(x=-R_eq/2, color='red', linestyle=':', label='H nuclei')
    ax.axvline(x=R_eq/2, color='red', linestyle=':')
    ax.set_xlabel('x (bohr)')
    ax.set_ylabel('rho(x)')
    ax.set_title(f'Electron Density at R_eq = {R_eq:.2f}')
    ax.legend()

    plt.suptitle('Kohn-Sham DFT: H2 Molecule', fontsize=14)
    plt.tight_layout()
    plt.savefig('step5_kohn_sham/ks_h2_molecule.png', dpi=150)
    plt.show()

    print(f"\nEquilibrium bond length: {R_eq:.2f} bohr")
    print(f"Total energy at equilibrium: {E_eq:.6f} hartree")


def demo_scf_visualization():
    """Visualize the SCF process: how density evolves to self-consistency."""
    print("\n" + "="*70)
    print("  DEMO: Visualizing SCF Convergence")
    print("="*70)

    dft = KohnShamDFT1D(N_grid=200, x_max=15.0, N_elec=4, softening=0.5)
    dft.set_nuclear_potential(Z_list=[4.0], R_list=[0.0])
    dft.solve(max_iter=60, tol=1e-9, mixing=0.3, verbose=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Density evolution
    ax = axes[0]
    n_snapshots = min(8, len(dft.rho_history))
    indices = np.linspace(0, len(dft.rho_history)-1, n_snapshots, dtype=int)
    for idx in indices:
        alpha = 0.3 + 0.7 * idx / max(indices)
        ax.plot(dft.x, dft.rho_history[idx], alpha=alpha,
                label=f'iter {idx}')
    ax.set_xlabel('x (bohr)')
    ax.set_ylabel('rho(x)')
    ax.set_title('Density Evolution During SCF')
    ax.legend(fontsize=9)

    # Energy convergence
    ax = axes[1]
    ax.plot(dft.E_history, 'bo-', markersize=4)
    ax.set_xlabel('SCF Iteration')
    ax.set_ylabel('Total Energy (hartree)')
    ax.set_title('Energy Convergence')

    plt.tight_layout()
    plt.savefig('step5_kohn_sham/scf_convergence.png', dpi=150)
    plt.show()


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("  STEP 5: KOHN-SHAM DFT FROM SCRATCH")
    print("="*70)

    print("\n--- 5.1 Helium Atom ---")
    demo_helium_atom()

    print("\n--- 5.2 H2 Molecule ---")
    demo_h2_molecule()

    print("\n--- 5.3 SCF Visualization ---")
    demo_scf_visualization()

    print("\n" + "="*70)
    print("  KEY TAKEAWAYS")
    print("="*70)
    print("""
    1. Kohn-Sham maps the interacting problem to a non-interacting one
       with the SAME density, using an effective potential V_KS.

    2. V_KS = V_ext + V_H + V_xc
       - V_ext: given (nuclear potential)
       - V_H: computed from density (Hartree)
       - V_xc: THE approximation in DFT

    3. The SCF loop is the core algorithm:
       density -> potential -> orbitals -> density -> ...

    4. The choice of V_xc (exchange-correlation functional) determines
       the accuracy. This is the topic of Step 6.

    Next --> Step 6: Exchange-Correlation Functionals
    """)
