"""
==============================================================================
STEP 4: Hohenberg-Kohn Theorems - The Foundation of DFT
==============================================================================

The two Hohenberg-Kohn (HK) theorems (1964) are the theoretical bedrock
of Density Functional Theory. They prove that the electron density rho(r)
is sufficient to determine ALL ground-state properties.

THEOREM 1 (Existence Theorem):
    The external potential V_ext(r) is (up to a constant) a unique
    functional of the ground-state electron density rho_0(r).

    In other words: rho_0(r) --> V_ext(r) --> H --> Psi --> everything
    The density uniquely determines the Hamiltonian!

    Proof sketch (by contradiction):
    Assume two different potentials V and V' give the same density rho.
    They give different Hamiltonians H, H' and different wavefunctions Psi, Psi'.
    By the variational principle:
        E  = <Psi |H |Psi > < <Psi'|H |Psi'>  = E'  + <Psi'|(V-V')|Psi'>
        E' = <Psi'|H'|Psi'> < <Psi |H'|Psi >  = E   + <Psi |(V'-V)|Psi >

    Adding these: E + E' < E + E'  --> CONTRADICTION!
    Therefore, two different V_ext cannot produce the same ground-state rho.

THEOREM 2 (Variational Principle):
    There exists a universal functional F[rho] such that the energy
    functional:
        E[rho] = F[rho] + integral V_ext(r) * rho(r) dr

    is minimized by the true ground-state density, and the minimum
    value is the true ground-state energy.

    F[rho] = T[rho] + V_ee[rho]  (universal: same for ALL systems!)
    F[rho] is the holy grail of DFT -- if we knew it exactly, DFT
    would give exact results. We don't, so we approximate.

THE PROBLEM:
    HK theorems are existence proofs -- they tell us F[rho] EXISTS
    but not what it IS. The challenge is finding good approximations.
    --> This leads to the Kohn-Sham approach (Step 5).
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.optimize import minimize


# ==========================================================================
# PART 1: Demonstrate the HK mapping: V_ext <--> rho
# ==========================================================================

def demonstrate_hk_theorem1():
    """
    Numerically verify HK Theorem 1: different V_ext --> different rho.

    We solve several 1D systems with different potentials and show
    that each gives a unique density.
    """
    print("=" * 70)
    print("Hohenberg-Kohn Theorem 1: V_ext(r) <--> rho(r) is one-to-one")
    print("=" * 70)

    N = 200
    x = np.linspace(-10, 10, N)
    dx = x[1] - x[0]

    # Build kinetic energy matrix
    T = np.zeros((N, N))
    for i in range(N):
        T[i, i] = 1.0 / dx**2
        if i > 0:
            T[i, i-1] = -0.5 / dx**2
        if i < N - 1:
            T[i, i+1] = -0.5 / dx**2

    # Different external potentials
    potentials = {
        'Harmonic: V=0.5*x^2': 0.5 * x**2,
        'Double well: V=0.1*(x^2-4)^2': 0.1 * (x**2 - 4)**2,
        'Asymmetric: V=0.5*x^2+0.1*x^3': 0.5 * x**2 + 0.1 * x**3,
        'Quartic: V=0.05*x^4': 0.05 * x**4,
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot potentials
    ax = axes[0]
    for name, V in potentials.items():
        ax.plot(x, V, linewidth=2, label=name)
    ax.set_xlabel('x (bohr)')
    ax.set_ylabel('V_ext(x) (hartree)')
    ax.set_title('Different External Potentials')
    ax.set_ylim(-0.5, 8)
    ax.legend()

    # Plot corresponding densities (2 electrons)
    ax = axes[1]
    for name, V in potentials.items():
        H = T + np.diag(V)
        energies, psi = eigh(H)

        # Normalize ground state
        psi_0 = psi[:, 0]
        psi_0 /= np.sqrt(np.trapz(psi_0**2, x))

        # Density for 2 electrons
        rho = 2 * psi_0**2
        ax.plot(x, rho, linewidth=2, label=name)

    ax.set_xlabel('x (bohr)')
    ax.set_ylabel('rho(x)')
    ax.set_title('Corresponding Ground-State Densities (unique!)')
    ax.legend()

    plt.tight_layout()
    plt.savefig('step4_hohenberg_kohn/hk_theorem1.png', dpi=150)
    plt.show()

    print("""
    Each external potential produces a UNIQUE ground-state density.
    This is Hohenberg-Kohn Theorem 1.

    The implication: if you know rho(r), you know V_ext(r),
    which means you know the entire Hamiltonian, and therefore
    ALL ground-state properties!
    """)


# ==========================================================================
# PART 2: Demonstrate the variational principle for density
# ==========================================================================

def demonstrate_hk_theorem2():
    """
    Numerically verify HK Theorem 2: E[rho] >= E_0, with equality at rho_0.

    For a simple 1D system, we:
    1. Solve exactly to get rho_0 and E_0
    2. Try different trial densities rho_trial
    3. Show E[rho_trial] >= E_0 always
    """
    print("=" * 70)
    print("Hohenberg-Kohn Theorem 2: Variational Principle for Density")
    print("=" * 70)

    N = 200
    x = np.linspace(-10, 10, N)
    dx = x[1] - x[0]

    # Harmonic potential, 2 electrons (non-interacting for simplicity)
    V_ext = 0.5 * x**2

    T = np.zeros((N, N))
    for i in range(N):
        T[i, i] = 1.0 / dx**2
        if i > 0:
            T[i, i-1] = -0.5 / dx**2
        if i < N - 1:
            T[i, i+1] = -0.5 / dx**2

    H = T + np.diag(V_ext)
    energies, psi = eigh(H)

    # Exact ground state
    psi_0 = psi[:, 0]
    psi_0 /= np.sqrt(np.trapz(psi_0**2, x))
    rho_exact = 2 * psi_0**2
    E_exact = 2 * energies[0]

    # For non-interacting electrons, we can compute T[rho] exactly
    # from the orbitals: T_s = -1/2 sum <phi_i|nabla^2|phi_i>
    T_exact = 2 * (psi_0 @ T @ psi_0) * dx

    # Trial densities (Gaussian with different widths)
    sigmas = np.linspace(0.5, 3.0, 30)
    E_trials = []

    for sigma in sigmas:
        # Trial density: Gaussian normalized to N_elec=2
        rho_trial = 2 * np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

        # External potential energy (exact functional)
        E_ext = np.trapz(V_ext * rho_trial, x)

        # Kinetic energy (use von Weizsacker approximation for 2-electron system):
        # T_vW[rho] = 1/8 * integral |nabla rho|^2 / rho dx
        drho = np.gradient(rho_trial, dx)
        # Avoid division by zero
        safe_rho = np.maximum(rho_trial, 1e-20)
        T_vW = (1.0 / 8.0) * np.trapz(drho**2 / safe_rho, x)

        E_trial = T_vW + E_ext
        E_trials.append(E_trial)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: trial densities vs exact
    ax = axes[0]
    for sigma in [0.8, 1.2, 2.0]:
        rho_trial = 2 * np.exp(-x**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
        ax.plot(x, rho_trial, '--', linewidth=1.5, label=f'trial (sigma={sigma})')
    ax.plot(x, rho_exact, 'k-', linewidth=3, label='Exact rho_0')
    ax.set_xlabel('x (bohr)')
    ax.set_ylabel('rho(x)')
    ax.set_title('Trial Densities vs Exact')
    ax.legend()

    # Right: energy functional
    ax = axes[1]
    ax.plot(sigmas, E_trials, 'b-', linewidth=2, label='E[rho_trial]')
    ax.axhline(y=E_exact, color='r', linestyle='--', linewidth=2, label=f'E_exact = {E_exact:.4f}')
    ax.set_xlabel('Trial density width sigma')
    ax.set_ylabel('Energy (hartree)')
    ax.set_title('HK Theorem 2: E[rho] >= E_0')
    ax.legend()

    # Mark minimum
    idx_min = np.argmin(E_trials)
    ax.plot(sigmas[idx_min], E_trials[idx_min], 'ro', markersize=10)
    ax.annotate(f'min E = {E_trials[idx_min]:.4f}\nsigma = {sigmas[idx_min]:.2f}',
                xy=(sigmas[idx_min], E_trials[idx_min]),
                xytext=(sigmas[idx_min] + 0.5, E_trials[idx_min] + 0.3),
                arrowprops=dict(arrowstyle='->'), fontsize=11)

    plt.tight_layout()
    plt.savefig('step4_hohenberg_kohn/hk_theorem2.png', dpi=150)
    plt.show()

    print(f"\n  Exact ground-state energy: {E_exact:.6f} hartree")
    print(f"  Best trial energy: {min(E_trials):.6f} hartree")
    print(f"  E[rho_trial] >= E_0 always? {all(e >= E_exact - 0.01 for e in E_trials)}")

    print("""
    HK Theorem 2 says: you can find the ground-state by MINIMIZING
    the energy functional E[rho] over all valid densities.

    But we need to know the functional F[rho] = T[rho] + V_ee[rho]!
    - T[rho]: kinetic energy as functional of density (VERY hard)
    - V_ee[rho]: electron-electron interaction functional

    The von Weizsacker approximation T_vW is OK for 2 electrons but
    terrible for many electrons. We need a better way.

    --> Kohn-Sham's brilliant idea: use ORBITALS to compute T_s,
        and put everything we don't know into E_xc[rho].
    """)


# ==========================================================================
# PART 3: The energy functional decomposition
# ==========================================================================

def energy_functional_decomposition():
    """
    Show the decomposition of the total energy functional:

    E[rho] = T_s[rho] + E_ext[rho] + E_H[rho] + E_xc[rho]

    where:
    T_s[rho]  = kinetic energy of non-interacting electrons (KS orbitals)
    E_ext[rho] = integral V_ext(r) * rho(r) dr
    E_H[rho]  = 1/2 integral integral rho(r)*rho(r')/|r-r'| dr dr' (Hartree)
    E_xc[rho] = exchange-correlation energy (THE unknown)

    E_xc = (T - T_s) + (V_ee - E_H)
         = T_c (correlation kinetic) + non-classical V_ee

    This decomposition is the KEY step from HK to Kohn-Sham!
    """
    print("=" * 70)
    print("Energy Functional Decomposition: The Road to Kohn-Sham")
    print("=" * 70)

    # Visual diagram using text
    diagram = """
    EXACT ENERGY:
    E[rho] = T[rho] + V_ext[rho] + V_ee[rho]
             ------   ----------   ---------
             exact      known       exact
             kinetic    trivially   but complex

    KOHN-SHAM DECOMPOSITION:
    E[rho] = T_s[rho] + E_ext[rho] + E_H[rho] + E_xc[rho]
             --------   ----------   ---------   ----------
             KS kinetic   known      Hartree     UNKNOWN!
             (from        trivially  (classical   (must be
              orbitals)              Coulomb)     approximated)

    WHERE DOES E_xc COME FROM?
    E_xc[rho] = (T[rho] - T_s[rho]) + (V_ee[rho] - E_H[rho])
              = T_c (correlation        + non-classical
                    kinetic energy)       electron-electron

    E_xc IS SMALL compared to T_s and E_H, but it contains ALL
    the difficult many-body physics. The genius of Kohn-Sham is
    that we only need to approximate the SMALL part!
    """
    print(diagram)

    # Pie chart showing energy contributions for a typical atom
    fig, ax = plt.subplots(figsize=(8, 8))

    # Typical magnitudes for Neon atom (in hartree, approximate)
    contributions = {
        'T_s (KS kinetic)': 128.5,
        'E_ext (nuclear)': -311.3,
        'E_H (Hartree)': 66.1,
        'E_x (exchange)': -12.1,
        'E_c (correlation)': -0.39,
    }

    # Bar chart instead (pie chart doesn't work well with negative values)
    names = list(contributions.keys())
    values = list(contributions.values())
    colors = ['steelblue', 'darkred', 'orange', 'green', 'purple']

    bars = ax.barh(names, values, color=colors, edgecolor='black')
    ax.set_xlabel('Energy (hartree)', fontsize=12)
    ax.set_title('Energy Contributions in Neon Atom\n(approximate magnitudes)', fontsize=14)
    ax.axvline(x=0, color='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        x_pos = val + (2 if val > 0 else -8)
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                f'{val:.1f} Ha', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('step4_hohenberg_kohn/energy_decomposition.png', dpi=150)
    plt.show()

    print("""
    KEY OBSERVATIONS:
    1. T_s and E_ext are LARGE but computed exactly in Kohn-Sham
    2. E_H is large and computed exactly (classical Coulomb integral)
    3. E_x is moderate (~10% of E_H) -- well approximated by LDA/GGA
    4. E_c is SMALL (~0.3% of total) but chemically crucial!

    The correlation energy determines:
    - Chemical bond strengths (a few kcal/mol matters!)
    - Reaction barriers
    - Van der Waals interactions
    - Magnetic properties

    Approximating E_xc is the art of DFT --> Step 6
    But first, we need the Kohn-Sham equations --> Step 5
    """)


# ==========================================================================
# PART 4: Demonstrate v-representability
# ==========================================================================

def v_representability_demo():
    """
    Show the concept of v-representability.

    Not every function that looks like a density actually IS a
    ground-state density of some potential. A valid density must be:
    1. rho(r) >= 0  (non-negative)
    2. integral rho(r) dr = N  (integrates to electron number)
    3. sqrt(rho) must be smooth (no cusps except at nuclei)
    4. v-representable: it must be the ground-state density of SOME V_ext

    The Kohn-Sham approach sidesteps this issue by working with
    orbitals that automatically produce valid densities.
    """
    print("=" * 70)
    print("V-Representability: Which densities are valid?")
    print("=" * 70)

    N = 200
    x = np.linspace(-10, 10, N)
    dx = x[1] - x[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Valid density: Gaussian (ground state of harmonic oscillator)
    rho_valid = 2 * np.exp(-x**2) / np.sqrt(np.pi)
    axes[0, 0].plot(x, rho_valid, 'b-', linewidth=2)
    axes[0, 0].fill_between(x, rho_valid, alpha=0.3)
    axes[0, 0].set_title(f'Valid: Gaussian\nN = {np.trapz(rho_valid, x):.2f}')
    axes[0, 0].set_ylabel('rho(x)')

    # Valid density: bimodal (ground state of double well)
    rho_bimodal = np.exp(-(x-2)**2) + np.exp(-(x+2)**2)
    rho_bimodal *= 2.0 / np.trapz(rho_bimodal, x)  # normalize to N=2
    axes[0, 1].plot(x, rho_bimodal, 'b-', linewidth=2)
    axes[0, 1].fill_between(x, rho_bimodal, alpha=0.3)
    axes[0, 1].set_title(f'Valid: Bimodal\nN = {np.trapz(rho_bimodal, x):.2f}')

    # Invalid: negative density
    rho_negative = 2 * np.exp(-x**2) / np.sqrt(np.pi) - 0.3 * np.exp(-(x-1)**2)
    axes[1, 0].plot(x, rho_negative, 'r-', linewidth=2)
    axes[1, 0].fill_between(x, rho_negative, where=(rho_negative < 0),
                            alpha=0.3, color='red', label='rho < 0!')
    axes[1, 0].fill_between(x, rho_negative, where=(rho_negative >= 0),
                            alpha=0.3, color='blue')
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 0].set_title('INVALID: Negative density')
    axes[1, 0].set_ylabel('rho(x)')
    axes[1, 0].legend()

    # Invalid: discontinuous density
    rho_discontinuous = np.where(np.abs(x) < 3, 1.0/3.0, 0.0)
    axes[1, 1].plot(x, rho_discontinuous, 'r-', linewidth=2)
    axes[1, 1].fill_between(x, rho_discontinuous, alpha=0.3, color='red')
    axes[1, 1].set_title(f'Problematic: Discontinuous\nN = {np.trapz(rho_discontinuous, x):.2f}')

    for ax in axes.flat:
        ax.set_xlabel('x (bohr)')

    plt.suptitle('Density Constraints for DFT', fontsize=14)
    plt.tight_layout()
    plt.savefig('step4_hohenberg_kohn/v_representability.png', dpi=150)
    plt.show()

    print("""
    For DFT to work, the trial density must be "v-representable":
    it must be the ground-state density of SOME external potential.

    The Kohn-Sham approach elegantly solves this by constructing
    the density from orbitals:  rho(r) = sum_i |phi_i(r)|^2

    This guarantees a valid, v-representable density automatically!
    """)


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("  STEP 4: HOHENBERG-KOHN THEOREMS")
    print("="*70)

    print("\n--- 4.1 HK Theorem 1: Unique Mapping ---\n")
    demonstrate_hk_theorem1()

    print("\n--- 4.2 HK Theorem 2: Variational Principle ---\n")
    demonstrate_hk_theorem2()

    print("\n--- 4.3 Energy Functional Decomposition ---\n")
    energy_functional_decomposition()

    print("\n--- 4.4 V-Representability ---\n")
    v_representability_demo()

    print("\n" + "="*70)
    print("  SUMMARY: HK Theorems -> Road to Kohn-Sham")
    print("="*70)
    print("""
    HK Theorem 1: rho(r) uniquely determines V_ext(r) and hence
                   the entire Hamiltonian and all ground-state properties.

    HK Theorem 2: The energy functional E[rho] is minimized by the
                   true ground-state density.

    The problem: We don't know the exact F[rho] = T[rho] + V_ee[rho].

    The solution (Kohn-Sham):
    - Compute T_s from orbitals (exact for non-interacting system)
    - Compute E_H from density (classical Coulomb)
    - Lump the rest into E_xc[rho] (to be approximated)

    Next --> Step 5: Kohn-Sham Equations
    """)
