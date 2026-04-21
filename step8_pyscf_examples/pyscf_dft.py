"""
==============================================================================
STEP 8: Practical DFT with PySCF
==============================================================================

Now we use a real quantum chemistry package (PySCF) to perform actual
DFT calculations on real molecules. PySCF is an open-source Python
package that implements all the concepts from Steps 1-7.

PySCF handles:
- Basis sets (GTO integrals, standard basis set libraries)
- SCF procedure (with advanced convergence algorithms like DIIS)
- Exchange-correlation functionals (via libxc)
- Analytical gradients (for geometry optimization)
- Many post-DFT methods (TDDFT, MP2, CCSD, CASSCF, ...)

In this step we will:
1. Single-point energy of H2
2. Compare functionals (LDA vs GGA vs hybrid)
3. Basis set convergence study
4. Geometry optimization of H2O
5. Compute molecular properties (dipole moment, orbital energies)
6. Potential energy surface scan
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    from pyscf import gto, scf, dft, lib
    from pyscf.geomopt import geometric_solver
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False
    print("PySCF not installed. Install with: pip install pyscf")
    print("Running in demo mode with pre-computed results.")


# ==========================================================================
# PART 1: Your first DFT calculation - H2 molecule
# ==========================================================================

def first_dft_calculation():
    """
    The simplest possible DFT calculation: H2 with PBE/cc-pVDZ.

    This walks through every step:
    1. Define the molecule (geometry + basis set)
    2. Set up the DFT calculation (functional)
    3. Run the SCF
    4. Analyze results
    """
    print("=" * 70)
    print("  Your First DFT Calculation: H2 Molecule")
    print("=" * 70)

    if not PYSCF_AVAILABLE:
        print("\n[PySCF not available - showing expected results]")
        print("E(H2, PBE/cc-pVDZ) ≈ -1.1665 hartree")
        return None

    # Step 1: Define the molecule
    # Geometry in Angstroms (PySCF default), converted internally to bohr
    mol = gto.M(
        atom='''
            H  0.0  0.0  0.0
            H  0.0  0.0  0.74
        ''',
        basis='cc-pvdz',      # Dunning's correlation-consistent basis
        charge=0,              # neutral molecule
        spin=0,                # singlet (N_alpha - N_beta = 0)
        verbose=4,             # print level (4 = detailed)
    )

    print(f"\nMolecule: H2")
    print(f"Basis set: {mol.basis}")
    print(f"Number of basis functions: {mol.nao}")
    print(f"Number of electrons: {mol.nelectron}")
    print(f"Nuclear repulsion energy: {mol.energy_nuc():.8f} hartree")

    # Step 2: Set up DFT calculation
    mf = dft.RKS(mol)         # Restricted Kohn-Sham
    mf.xc = 'pbe'             # PBE functional (GGA)

    # Step 3: Run the SCF
    E_total = mf.kernel()

    # Step 4: Analyze
    print(f"\n{'='*50}")
    print(f"  RESULTS")
    print(f"{'='*50}")
    print(f"Total energy:     {E_total:.8f} hartree")
    print(f"                = {E_total * 27.2114:.4f} eV")
    print(f"Orbital energies: {mf.mo_energy}")
    print(f"HOMO energy:      {mf.mo_energy[mol.nelectron//2 - 1]:.6f} Ha")
    print(f"LUMO energy:      {mf.mo_energy[mol.nelectron//2]:.6f} Ha")
    print(f"HOMO-LUMO gap:    {(mf.mo_energy[mol.nelectron//2] - mf.mo_energy[mol.nelectron//2-1]):.6f} Ha")
    print(f"                = {(mf.mo_energy[mol.nelectron//2] - mf.mo_energy[mol.nelectron//2-1]) * 27.2114:.4f} eV")

    return mf


# ==========================================================================
# PART 2: Compare functionals
# ==========================================================================

def compare_functionals():
    """
    Compare LDA, GGA, and hybrid functionals for H2O.

    This shows how different functional choices affect:
    - Total energy
    - Orbital energies
    - HOMO-LUMO gap
    """
    print("\n" + "=" * 70)
    print("  Functional Comparison: H2O")
    print("=" * 70)

    if not PYSCF_AVAILABLE:
        print("\n[PySCF not available - showing expected results]")
        print("LDA(SVWN)  ≈ -75.854 Ha")
        print("GGA(PBE)   ≈ -76.270 Ha")
        print("Hybrid(B3LYP) ≈ -76.389 Ha")
        return

    mol = gto.M(
        atom='''
            O   0.000   0.000   0.117
            H   0.000   0.757  -0.469
            H   0.000  -0.757  -0.469
        ''',
        basis='6-31g(d)',
        verbose=0,
    )

    functionals = {
        'HF':     None,      # Hartree-Fock (no correlation)
        'SVWN':   'svwn',    # LDA
        'PBE':    'pbe',     # GGA
        'BLYP':   'blyp',   # GGA
        'B3LYP':  'b3lyp',  # Hybrid (20% HF exchange)
        'PBE0':   'pbe0',   # Hybrid (25% HF exchange)
    }

    results = {}
    for name, xc in functionals.items():
        if name == 'HF':
            mf = scf.RHF(mol)
        else:
            mf = dft.RKS(mol)
            mf.xc = xc

        mf.verbose = 0
        E = mf.kernel()

        n_occ = mol.nelectron // 2
        homo = mf.mo_energy[n_occ - 1]
        lumo = mf.mo_energy[n_occ]
        gap = (lumo - homo) * 27.2114  # convert to eV

        results[name] = {'E': E, 'HOMO': homo, 'LUMO': lumo, 'gap': gap}

    # Print table
    print(f"\n{'Functional':>12} {'Energy (Ha)':>15} {'HOMO (Ha)':>12} {'LUMO (Ha)':>12} {'Gap (eV)':>10}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:>12} {r['E']:15.8f} {r['HOMO']:12.6f} {r['LUMO']:12.6f} {r['gap']:10.4f}")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = list(results.keys())
    energies = [results[n]['E'] for n in names]
    gaps = [results[n]['gap'] for n in names]

    ax = axes[0]
    ax.bar(names, energies, color=['gray', 'blue', 'green', 'green', 'red', 'red'],
           edgecolor='black')
    ax.set_ylabel('Total Energy (hartree)')
    ax.set_title('H2O: Total Energy by Functional')
    ax.tick_params(axis='x', rotation=45)

    ax = axes[1]
    ax.bar(names, gaps, color=['gray', 'blue', 'green', 'green', 'red', 'red'],
           edgecolor='black')
    ax.set_ylabel('HOMO-LUMO Gap (eV)')
    ax.set_title('H2O: HOMO-LUMO Gap by Functional')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('step8_pyscf_examples/functional_comparison.png', dpi=150)
    plt.show()

    print("""
    OBSERVATIONS:
    - HF gives the highest energy (no correlation)
    - LDA (SVWN) overbinds (too negative)
    - GGA (PBE, BLYP) improves on LDA
    - Hybrids (B3LYP, PBE0) give larger HOMO-LUMO gaps
      (closer to experimental ionization potential / electron affinity)
    - The HOMO-LUMO gap from KS-DFT is NOT the true band gap,
      but hybrid functionals give better estimates
    """)


# ==========================================================================
# PART 3: Basis set convergence
# ==========================================================================

def basis_set_convergence():
    """Demonstrate basis set convergence for H2O with PBE."""
    print("\n" + "=" * 70)
    print("  Basis Set Convergence: H2O / PBE")
    print("=" * 70)

    if not PYSCF_AVAILABLE:
        print("\n[PySCF not available - showing expected results]")
        return

    mol_template = '''
        O   0.000   0.000   0.117
        H   0.000   0.757  -0.469
        H   0.000  -0.757  -0.469
    '''

    basis_sets = ['sto-3g', '3-21g', '6-31g', '6-31g(d)', '6-31g(d,p)',
                  '6-311g(d,p)', 'cc-pvdz', 'cc-pvtz', 'cc-pvqz']

    results = []
    for basis in basis_sets:
        try:
            mol = gto.M(atom=mol_template, basis=basis, verbose=0)
            mf = dft.RKS(mol)
            mf.xc = 'pbe'
            mf.verbose = 0
            E = mf.kernel()
            results.append((basis, mol.nao, E))
            print(f"  {basis:>15}  N_ao={mol.nao:4d}  E={E:.8f} Ha")
        except Exception as e:
            print(f"  {basis:>15}  FAILED: {e}")

    if len(results) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        names = [r[0] for r in results]
        n_ao = [r[1] for r in results]
        energies = [r[2] for r in results]
        E_best = energies[-1]

        ax = axes[0]
        ax.plot(range(len(names)), energies, 'bo-', linewidth=2, markersize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Total Energy (hartree)')
        ax.set_title('Basis Set Convergence: H2O/PBE')

        ax = axes[1]
        errors = [(e - E_best) * 627.509 for e in energies]  # kcal/mol
        ax.semilogy(n_ao, [max(abs(e), 1e-3) for e in errors], 'ro-',
                    linewidth=2, markersize=8)
        ax.axhline(y=1.0, color='green', linestyle='--', label='1 kcal/mol')
        ax.set_xlabel('Number of Basis Functions')
        ax.set_ylabel('Error vs largest basis (kcal/mol)')
        ax.set_title('Convergence Rate')
        ax.legend()

        plt.tight_layout()
        plt.savefig('step8_pyscf_examples/basis_convergence.png', dpi=150)
        plt.show()


# ==========================================================================
# PART 4: Geometry optimization
# ==========================================================================

def geometry_optimization():
    """Optimize the geometry of H2O using DFT."""
    print("\n" + "=" * 70)
    print("  Geometry Optimization: H2O / B3LYP / 6-31G(d)")
    print("=" * 70)

    if not PYSCF_AVAILABLE:
        print("\n[PySCF not available - showing expected results]")
        print("Optimized O-H bond length ≈ 0.969 Angstrom")
        print("Optimized H-O-H angle ≈ 103.6 degrees")
        return

    # Initial (slightly distorted) geometry
    mol = gto.M(
        atom='''
            O   0.000   0.000   0.120
            H   0.000   0.800  -0.500
            H   0.000  -0.800  -0.500
        ''',
        basis='6-31g(d)',
        verbose=0,
    )

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.verbose = 0

    try:
        mol_eq = geometric_solver.optimize(mf, maxsteps=50)
        print(f"\nOptimized geometry (Angstrom):")
        coords = mol_eq.atom_coords(unit='Angstrom')
        atoms = [mol_eq.atom_symbol(i) for i in range(mol_eq.natm)]
        for i, (atom, coord) in enumerate(zip(atoms, coords)):
            print(f"  {atom}  {coord[0]:10.6f} {coord[1]:10.6f} {coord[2]:10.6f}")

        # Compute bond lengths and angle
        O = coords[0]
        H1 = coords[1]
        H2 = coords[2]
        r_OH1 = np.linalg.norm(H1 - O)
        r_OH2 = np.linalg.norm(H2 - O)

        v1 = H1 - O
        v2 = H2 - O
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        print(f"\n  O-H bond lengths: {r_OH1:.4f}, {r_OH2:.4f} Angstrom")
        print(f"  H-O-H angle: {angle:.2f} degrees")
        print(f"\n  Experimental: O-H = 0.958 A, H-O-H = 104.5 deg")

        # Final energy
        mf_final = dft.RKS(mol_eq)
        mf_final.xc = 'b3lyp'
        mf_final.verbose = 0
        E_final = mf_final.kernel()
        print(f"  Optimized energy: {E_final:.8f} hartree")

    except Exception as e:
        print(f"Geometry optimization failed: {e}")
        print("This may require the 'geometric' package: pip install geometric")


# ==========================================================================
# PART 5: Potential energy surface
# ==========================================================================

def potential_energy_surface():
    """Compute the PES for H2 bond stretching."""
    print("\n" + "=" * 70)
    print("  Potential Energy Surface: H2 Dissociation")
    print("=" * 70)

    if not PYSCF_AVAILABLE:
        print("\n[PySCF not available - showing expected results]")
        return

    distances = np.linspace(0.4, 4.0, 30)  # in Angstrom
    functionals = {'PBE': 'pbe', 'B3LYP': 'b3lyp'}
    results = {name: [] for name in functionals}

    # Also do HF for comparison
    results['HF'] = []

    for R in distances:
        mol = gto.M(
            atom=f'H 0 0 0; H 0 0 {R}',
            basis='cc-pvtz',
            verbose=0,
        )

        # HF
        mf = scf.RHF(mol)
        mf.verbose = 0
        results['HF'].append(mf.kernel())

        # DFT
        for name, xc in functionals.items():
            mf = dft.RKS(mol)
            mf.xc = xc
            mf.verbose = 0
            results[name].append(mf.kernel())

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {'HF': 'gray', 'PBE': 'blue', 'B3LYP': 'red'}
    for name, energies in results.items():
        # Shift so dissociation limit is 0
        E = np.array(energies)
        ax.plot(distances, (E - E[-1]) * 627.509, color=colors[name],
                linewidth=2, label=name)

    ax.set_xlabel('H-H distance (Angstrom)', fontsize=12)
    ax.set_ylabel('Relative Energy (kcal/mol)', fontsize=12)
    ax.set_title('H2 Dissociation: PES Comparison', fontsize=14)
    ax.legend(fontsize=12)
    ax.set_ylim(-120, 50)
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('step8_pyscf_examples/h2_pes.png', dpi=150)
    plt.show()

    # Find equilibrium for each
    for name, energies in results.items():
        E = np.array(energies)
        idx = np.argmin(E)
        D_e = (E[-1] - E[idx]) * 627.509  # dissociation energy in kcal/mol
        print(f"  {name:>6}: R_eq = {distances[idx]:.2f} A, D_e = {D_e:.1f} kcal/mol")

    print(f"\n  Experimental: R_eq = 0.74 A, D_e = 109.5 kcal/mol")


# ==========================================================================
# PART 6: Molecular properties
# ==========================================================================

def molecular_properties():
    """Compute various molecular properties for H2O."""
    print("\n" + "=" * 70)
    print("  Molecular Properties: H2O / B3LYP / cc-pVTZ")
    print("=" * 70)

    if not PYSCF_AVAILABLE:
        print("\n[PySCF not available - showing expected results]")
        return

    mol = gto.M(
        atom='''
            O   0.000   0.000   0.117
            H   0.000   0.757  -0.469
            H   0.000  -0.757  -0.469
        ''',
        basis='cc-pvtz',
        verbose=0,
    )

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.verbose = 0
    E = mf.kernel()

    # Dipole moment
    dm = mf.make_rdm1()
    dip = mf.dip_moment(verbose=0)
    dip_total = np.linalg.norm(dip)

    print(f"\n  Total energy: {E:.8f} hartree")
    print(f"  Dipole moment: ({dip[0]:.4f}, {dip[1]:.4f}, {dip[2]:.4f}) Debye")
    print(f"  |dipole|: {dip_total:.4f} Debye (experimental: 1.855 D)")

    # Orbital energies
    n_occ = mol.nelectron // 2
    print(f"\n  Orbital Energies (hartree):")
    print(f"  {'Orbital':>10} {'Energy':>12} {'Occ':>5}")
    print(f"  {'-'*30}")
    for i in range(min(n_occ + 3, len(mf.mo_energy))):
        occ = '2' if i < n_occ else '0'
        label = 'HOMO' if i == n_occ - 1 else ('LUMO' if i == n_occ else '')
        print(f"  {i+1:>10} {mf.mo_energy[i]:12.6f} {occ:>5}  {label}")

    gap = (mf.mo_energy[n_occ] - mf.mo_energy[n_occ-1]) * 27.2114
    print(f"\n  HOMO-LUMO gap: {gap:.2f} eV")

    # Mulliken population analysis
    print(f"\n  Mulliken Charges:")
    pop = mf.mulliken_pop(verbose=0)


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("  STEP 8: PRACTICAL DFT WITH PySCF")
    print("="*70)

    print("\n--- 8.1 First DFT Calculation ---")
    first_dft_calculation()

    print("\n--- 8.2 Functional Comparison ---")
    compare_functionals()

    print("\n--- 8.3 Basis Set Convergence ---")
    basis_set_convergence()

    print("\n--- 8.4 Geometry Optimization ---")
    geometry_optimization()

    print("\n--- 8.5 Potential Energy Surface ---")
    potential_energy_surface()

    print("\n--- 8.6 Molecular Properties ---")
    molecular_properties()

    print("\n" + "="*70)
    print("  CONGRATULATIONS!")
    print("="*70)
    print("""
    You've completed the full journey from the Schrodinger equation
    to practical DFT calculations:

    Step 1: Schrodinger equation (the foundation)
    Step 2: Many-body problem (why it's hard)
    Step 3: Hartree-Fock (first approximation: mean-field)
    Step 4: Hohenberg-Kohn (density is sufficient!)
    Step 5: Kohn-Sham (practical DFT framework)
    Step 6: XC functionals (the art of DFT)
    Step 7: Basis sets (practical implementation)
    Step 8: PySCF (real calculations!)

    NEXT STEPS:
    - Try different molecules and properties
    - Explore TDDFT for excited states
    - Try periodic DFT for solids (PySCF has PBC support)
    - Compare DFT with wave function methods (MP2, CCSD(T))
    - Learn about modern developments: machine-learned functionals,
      embedding methods, DFT+U, etc.
    """)
