"""
==============================================================================
STEP 7: Basis Sets for Quantum Chemistry
==============================================================================

In Steps 1-5, we solved the Schrodinger/KS equations on a REAL-SPACE GRID.
In practice, most quantum chemistry codes expand orbitals in a BASIS SET:

    phi_i(r) = sum_mu C_{mu,i} * chi_mu(r)

This converts the differential equation into a matrix eigenvalue problem:
    F C = S C epsilon   (Roothaan-Hall equations)

where:
    F_{mu,nu} = <chi_mu|F|chi_nu>  (Fock/KS matrix)
    S_{mu,nu} = <chi_mu|chi_nu>     (overlap matrix)
    C = MO coefficient matrix
    epsilon = orbital energies

TYPES OF BASIS FUNCTIONS:

1. Slater-Type Orbitals (STO):
    chi(r) = N * r^(n-1) * exp(-zeta * r) * Y_lm(theta, phi)
    - Correct physics (cusp at nucleus, exponential decay)
    - But: 3/4-center integrals are VERY expensive

2. Gaussian-Type Orbitals (GTO):
    chi(r) = N * x^a * y^b * z^c * exp(-alpha * r^2)
    - Wrong physics (no cusp, wrong decay)
    - But: product of two Gaussians is another Gaussian!
    - All integrals can be computed analytically --> FAST

3. Contracted GTOs (CGTO):
    chi_mu(r) = sum_k d_k * g_k(alpha_k, r)
    - Linear combination of "primitive" Gaussians
    - Mimics the shape of STOs with the efficiency of GTOs
    - This is what STO-3G means: each STO approximated by 3 GTOs

COMMON BASIS SET FAMILIES:

    Minimal: STO-3G (1 function per orbital, very approximate)
    Split-valence: 3-21G, 6-31G (separate core/valence)
    Polarization: 6-31G(d), 6-31G(d,p) (add higher angular momentum)
    Diffuse: 6-31+G(d) (add diffuse functions for anions, weak interactions)
    Correlation-consistent: cc-pVDZ, cc-pVTZ, cc-pVQZ (systematic convergence)
==============================================================================
"""

import os
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt
from scipy.special import factorial2


# ==========================================================================
# PART 1: Gaussian vs Slater-Type Orbitals
# ==========================================================================

def compare_sto_gto():
    """
    Compare Slater-Type Orbitals (STO) with Gaussian-Type Orbitals (GTO).

    Show why we use GTOs despite STOs having the correct physics.
    """
    r = np.linspace(0, 8, 500)

    # 1s Slater-type orbital: chi(r) = N * exp(-zeta * r)
    zeta = 1.0  # for hydrogen
    N_sto = (2 * zeta)**1.5 / np.sqrt(4 * np.pi)
    sto_1s = N_sto * np.exp(-zeta * r)

    # Single Gaussian: chi(r) = N * exp(-alpha * r^2)
    # Optimize alpha to best fit the STO
    alpha = 0.2709  # optimized for H 1s
    N_gto = (2 * alpha / np.pi)**0.75
    gto_1s = N_gto * np.exp(-alpha * r**2)

    # STO-3G: 3 Gaussians contracted to approximate the STO
    # Coefficients for H 1s STO-3G (from standard tables)
    alphas_sto3g = [3.42525091, 0.62391373, 0.16885540]
    coeffs_sto3g = [0.15432897, 0.53532814, 0.44463454]

    sto3g = np.zeros_like(r)
    for a, c in zip(alphas_sto3g, coeffs_sto3g):
        N = (2 * a / np.pi)**0.75
        sto3g += c * N * np.exp(-a * r**2)

    # STO-6G for comparison
    alphas_sto6g = [35.52322122, 6.51314831, 1.82221400,
                    0.62595580, 0.24307858, 0.10011709]
    coeffs_sto6g = [0.00916360, 0.04936149, 0.16853830,
                    0.37056280, 0.41649153, 0.13033408]
    sto6g = np.zeros_like(r)
    for a, c in zip(alphas_sto6g, coeffs_sto6g):
        N = (2 * a / np.pi)**0.75
        sto6g += c * N * np.exp(-a * r**2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Radial functions
    ax = axes[0]
    ax.plot(r, sto_1s, 'k-', linewidth=3, label='Exact STO (1s)')
    ax.plot(r, gto_1s, 'r--', linewidth=2, label='Single GTO')
    ax.plot(r, sto3g, 'b-.', linewidth=2, label='STO-3G (3 GTOs)')
    ax.plot(r, sto6g, 'g:', linewidth=2, label='STO-6G (6 GTOs)')
    ax.set_xlabel('r (bohr)')
    ax.set_ylabel('chi(r)')
    ax.set_title('1s Orbital: STO vs GTO Approximations')
    ax.legend()
    ax.set_xlim(0, 5)

    # Log plot to see behavior at origin and tail
    ax = axes[1]
    ax.semilogy(r[1:], np.abs(sto_1s[1:]), 'k-', linewidth=3, label='STO')
    ax.semilogy(r[1:], np.abs(gto_1s[1:]), 'r--', linewidth=2, label='Single GTO')
    ax.semilogy(r[1:], np.abs(sto3g[1:]), 'b-.', linewidth=2, label='STO-3G')
    ax.set_xlabel('r (bohr)')
    ax.set_ylabel('|chi(r)| (log scale)')
    ax.set_title('Tail Behavior: exp(-zeta*r) vs exp(-alpha*r^2)')
    ax.legend()
    ax.set_xlim(0, 7)

    plt.tight_layout()
    plt.savefig(os.path.join(_DIR, 'sto_vs_gto.png'), dpi=150)
    plt.show()

    print("""
    KEY OBSERVATIONS:
    1. Single GTO is poor: wrong cusp at r=0, decays too fast
    2. STO-3G is much better: 3 contracted GTOs approximate the STO shape
    3. STO-6G is nearly perfect
    4. The LOG plot shows: GTO decays as exp(-r^2) vs STO as exp(-r)
       --> GTO falls off too fast at large r

    WHY USE GTOs DESPITE WRONG PHYSICS?
    Product of two Gaussians centered at A and B:
        exp(-a*|r-A|^2) * exp(-b*|r-B|^2) = K * exp(-(a+b)*|r-P|^2)
    where P = (a*A + b*B)/(a+b) and K is a constant.

    This means ALL electron repulsion integrals are analytically computable!
    For STOs, 3- and 4-center integrals require expensive numerical methods.
    """)


# ==========================================================================
# PART 2: Basis set convergence
# ==========================================================================

def demonstrate_basis_convergence():
    """
    Show how energies converge as the basis set grows.

    Using hydrogen atom as an example with Gaussian basis sets of
    increasing size.
    """
    print("=" * 70)
    print("Basis Set Convergence: Hydrogen Atom")
    print("=" * 70)

    # Exact hydrogen 1s energy
    E_exact = -0.5  # hartree

    # Even-tempered Gaussian basis sets: alpha_i = alpha_0 * beta^i
    # We'll show how the energy converges as we add more Gaussians

    results = []

    for N_basis in range(1, 16):
        # Even-tempered basis: exponents span from tight to diffuse
        alpha_0 = 0.1
        beta = 2.5
        alphas = [alpha_0 * beta**i for i in range(N_basis)]

        # Build overlap matrix S and Hamiltonian matrix H
        S = np.zeros((N_basis, N_basis))
        H_mat = np.zeros((N_basis, N_basis))

        for i in range(N_basis):
            for j in range(N_basis):
                ai, aj = alphas[i], alphas[j]

                # Overlap: <gi|gj> = (pi/(ai+aj))^(3/2)
                S[i, j] = (np.pi / (ai + aj))**1.5

                # Kinetic energy: <gi|T|gj> = ai*aj/(ai+aj) * 3*(pi/(ai+aj))^(3/2)
                # T = -1/2 nabla^2 --> <g|T|g> = 3*ai*aj/(ai+aj) * (pi/(ai+aj))^(3/2)
                H_mat[i, j] = 3.0 * ai * aj / (ai + aj) * (np.pi / (ai + aj))**1.5

                # Nuclear attraction: <gi|(-Z/r)|gj> = -Z * 2*pi/(ai+aj)
                H_mat[i, j] -= 2.0 * np.pi / (ai + aj)

        try:
            energies, _ = eigh(H_mat, S)
            E_gs = energies[0]
            error = abs(E_gs - E_exact)
            results.append((N_basis, E_gs, error))
        except np.linalg.LinAlgError:
            results.append((N_basis, np.nan, np.nan))

    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    N_vals = [r[0] for r in results]
    E_vals = [r[1] for r in results]
    err_vals = [r[2] for r in results]

    ax = axes[0]
    ax.plot(N_vals, E_vals, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=E_exact, color='r', linestyle='--', linewidth=2,
               label=f'Exact: {E_exact} Ha')
    ax.set_xlabel('Number of Basis Functions')
    ax.set_ylabel('Ground State Energy (hartree)')
    ax.set_title('Basis Set Convergence')
    ax.legend()

    ax = axes[1]
    valid = [(n, e) for n, _, e in results if not np.isnan(e) and e > 1e-15]
    if valid:
        ax.semilogy([v[0] for v in valid], [v[1] for v in valid], 'ro-',
                    linewidth=2, markersize=8)
    ax.set_xlabel('Number of Basis Functions')
    ax.set_ylabel('|E - E_exact| (hartree)')
    ax.set_title('Convergence Rate')
    ax.axhline(y=1.6e-3, color='green', linestyle=':', label='Chemical accuracy (1 kcal/mol)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(_DIR, 'basis_convergence.png'), dpi=150)
    plt.show()

    print(f"\n{'N_basis':>8} {'Energy':>15} {'Error':>15}")
    print("-" * 42)
    for N, E, err in results:
        if not np.isnan(E):
            print(f"{N:8d} {E:15.8f} {err:15.2e}")


def demonstrate_basis_set_hierarchy():
    """Show the hierarchy of standard basis sets."""
    print("=" * 70)
    print("Standard Basis Set Hierarchy")
    print("=" * 70)

    # Basis set data for Carbon atom
    basis_sets = {
        'STO-3G': {'N_functions': 5, 'description': '1s, 2s, 2p (minimal)'},
        '3-21G': {'N_functions': 9, 'description': 'Split valence'},
        '6-31G': {'N_functions': 9, 'description': 'Split valence (better core)'},
        '6-31G(d)': {'N_functions': 15, 'description': '+ d polarization on C'},
        '6-31G(d,p)': {'N_functions': 17, 'description': '+ p polarization on H'},
        '6-31+G(d)': {'N_functions': 19, 'description': '+ diffuse functions'},
        '6-311+G(2d,p)': {'N_functions': 30, 'description': 'Triple-zeta + pol + diff'},
        'cc-pVDZ': {'N_functions': 14, 'description': 'Correlation-consistent DZ'},
        'cc-pVTZ': {'N_functions': 30, 'description': 'Correlation-consistent TZ'},
        'cc-pVQZ': {'N_functions': 55, 'description': 'Correlation-consistent QZ'},
    }

    print(f"\n{'Basis Set':>20} {'N (for C)':>10} {'Description':>40}")
    print("-" * 75)
    for name, info in basis_sets.items():
        print(f"{name:>20} {info['N_functions']:>10} {info['description']:>40}")

    # Visual comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    names = list(basis_sets.keys())
    N_funcs = [basis_sets[n]['N_functions'] for n in names]

    bars = ax.barh(range(len(names)), N_funcs, color='steelblue',
                   edgecolor='black')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Number of Basis Functions (for Carbon)')
    ax.set_title('Basis Set Size Comparison')

    # Add count labels
    for bar, n in zip(bars, N_funcs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(n), va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(_DIR, 'basis_hierarchy.png'), dpi=150)
    plt.show()

    print("""
    NAMING CONVENTIONS:

    Pople-style (6-31G family):
    - First number (6): # primitives for core orbitals
    - After dash (31): split valence -- 3 primitives inner + 1 outer
    - (d,p): polarization functions (d on heavy atoms, p on H)
    - +: diffuse functions (for anions, long-range interactions)

    Dunning-style (cc-pVXZ):
    - cc: correlation-consistent
    - pV: polarized valence
    - XZ: X-zeta (DZ=double, TZ=triple, QZ=quadruple)
    - Designed for systematic convergence to the basis set limit

    PRACTICAL ADVICE:
    - DFT: 6-31G(d) or cc-pVDZ is usually sufficient
    - Production: 6-311+G(2d,p) or cc-pVTZ
    - Correlated methods (MP2, CCSD): need larger basis (cc-pVTZ+)
    - Basis Set Superposition Error (BSSE): use counterpoise correction
    """)


# ==========================================================================
# PART 3: Gaussian integrals tutorial
# ==========================================================================

def gaussian_integral_tutorial():
    """
    Show how electron repulsion integrals work with Gaussians.

    The key advantage of GTOs: the product of two Gaussians is a Gaussian!
    This makes all integrals analytically computable.
    """
    print("=" * 70)
    print("Gaussian Product Rule: Why GTOs Win")
    print("=" * 70)

    # Two Gaussian functions centered at different points
    r = np.linspace(-5, 5, 500)

    # g_A(r) = exp(-alpha * (r - A)^2)
    # g_B(r) = exp(-beta * (r - B)^2)
    # Product: g_A * g_B = K * exp(-gamma * (r - P)^2)
    # where gamma = alpha + beta, P = (alpha*A + beta*B)/(alpha+beta)
    # K = exp(-alpha*beta/(alpha+beta) * (A-B)^2)

    alpha, beta = 1.0, 0.5
    A, B = -1.0, 1.5

    g_A = np.exp(-alpha * (r - A)**2)
    g_B = np.exp(-beta * (r - B)**2)
    product = g_A * g_B

    # Analytical product
    gamma = alpha + beta
    P = (alpha * A + beta * B) / gamma
    K = np.exp(-alpha * beta / gamma * (A - B)**2)
    product_analytical = K * np.exp(-gamma * (r - P)**2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(r, g_A, 'b-', linewidth=2, label=f'g_A (center={A}, alpha={alpha})')
    ax.plot(r, g_B, 'r-', linewidth=2, label=f'g_B (center={B}, beta={beta})')
    ax.plot(r, product, 'g-', linewidth=3, label='g_A * g_B (numerical)')
    ax.plot(r, product_analytical, 'k--', linewidth=2, label='Analytical product')
    ax.set_xlabel('r (bohr)')
    ax.set_ylabel('Function value')
    ax.set_title('Gaussian Product Rule')
    ax.legend()

    # Two-center integral visualization
    ax = axes[1]
    # Show overlap integral as function of distance
    distances = np.linspace(0, 5, 100)
    overlaps = []
    for d in distances:
        # <g_A|g_B> = (pi/(alpha+beta))^(3/2) * exp(-alpha*beta/(alpha+beta) * d^2)
        S = (np.pi / gamma)**1.5 * np.exp(-alpha * beta / gamma * d**2)
        overlaps.append(S)

    ax.plot(distances, overlaps, 'b-', linewidth=2)
    ax.set_xlabel('Distance |A - B| (bohr)')
    ax.set_ylabel('Overlap Integral')
    ax.set_title('Overlap Integral vs Distance')

    plt.tight_layout()
    plt.savefig(os.path.join(_DIR, 'gaussian_product.png'), dpi=150)
    plt.show()

    print("""
    THE GAUSSIAN PRODUCT THEOREM:

    exp(-a|r-A|^2) * exp(-b|r-B|^2) = K * exp(-(a+b)|r-P|^2)

    where P = (a*A + b*B)/(a+b) and K = exp(-ab/(a+b) * |A-B|^2)

    This means:
    1. Overlap integrals: analytical (Gaussian integral formula)
    2. Kinetic integrals: analytical (derivative of Gaussian)
    3. Nuclear attraction: analytical (error function)
    4. Electron repulsion (2e): analytical (Boys function)

    ALL integrals in quantum chemistry with GTOs are computable
    in closed form. This is why GTOs dominate despite having the
    wrong physical behavior at the origin and at large distances.
    """)


from scipy.linalg import eigh


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("  STEP 7: BASIS SETS")
    print("="*70)

    print("\n--- 7.1 STO vs GTO ---\n")
    compare_sto_gto()

    print("\n--- 7.2 Basis Set Convergence ---\n")
    demonstrate_basis_convergence()

    print("\n--- 7.3 Basis Set Hierarchy ---\n")
    demonstrate_basis_set_hierarchy()

    print("\n--- 7.4 Gaussian Integrals ---\n")
    gaussian_integral_tutorial()

    print("\n" + "="*70)
    print("  KEY TAKEAWAYS")
    print("="*70)
    print("""
    1. Basis sets expand orbitals: phi_i = sum_mu C_{mu,i} * chi_mu
       Converting the differential equation to a matrix eigenvalue problem.

    2. GTOs are used because their products are analytically integrable,
       despite having wrong physics at the nucleus and at long range.

    3. Contracted GTOs (STO-3G, etc.) combine multiple primitive GTOs
       to approximate the correct STO shape.

    4. Larger basis sets (more functions) give more accurate results
       but cost more (N^4 for 2-electron integrals in HF/DFT).

    5. For DFT: polarized double-zeta (6-31G(d) or cc-pVDZ) is often
       sufficient. Triple-zeta for production calculations.

    Next --> Step 8: Practical DFT with PySCF
    """)
