"""
==============================================================================
STEP 6: Exchange-Correlation Functionals
==============================================================================

The exchange-correlation (XC) functional E_xc[rho] is where ALL the
approximations in DFT live. Choosing the right functional is the
key practical decision in any DFT calculation.

JACOB'S LADDER OF XC FUNCTIONALS (Perdew, 2001):

    Rung 5: Double hybrids (B2PLYP) -- include MP2 correlation
    Rung 4: Hybrid functionals (B3LYP, PBE0) -- include exact exchange
    Rung 3: Meta-GGA (TPSS, SCAN) -- include kinetic energy density
    Rung 2: GGA (PBE, BLYP) -- include density gradient
    Rung 1: LDA -- only uses local density
    -------
    Earth: Hartree (no XC at all)

    Higher rungs are generally more accurate but more expensive.

EXCHANGE vs CORRELATION:
    E_xc = E_x + E_c

    Exchange (E_x):
    - Arises from Pauli exclusion principle (antisymmetry)
    - ~90% of E_xc
    - Well described even by LDA

    Correlation (E_c):
    - Arises from electron-electron Coulomb interaction beyond mean-field
    - ~10% of E_xc but chemically crucial
    - Much harder to approximate
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ==========================================================================
# PART 1: The Uniform Electron Gas and LDA
# ==========================================================================

def lda_exchange_3d(rho):
    """
    3D LDA exchange functional (Dirac, 1930).

    For the uniform electron gas:
        e_x(rho) = -C_x * rho^(1/3)
        C_x = 3/4 * (3/pi)^(1/3) = 0.7386

    E_x[rho] = integral rho * e_x(rho) dr = -C_x integral rho^(4/3) dr

    V_x(rho) = d(rho * e_x)/d rho = -4/3 * C_x * rho^(1/3)

    Parameters
    ----------
    rho : array
        Electron density

    Returns
    -------
    e_x : energy density per electron
    V_x : exchange potential
    """
    C_x = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
    safe_rho = np.maximum(rho, 1e-30)

    e_x = -C_x * safe_rho ** (1.0 / 3.0)
    V_x = -(4.0 / 3.0) * C_x * safe_rho ** (1.0 / 3.0)

    return e_x, V_x


def vwn_correlation_3d(rho):
    """
    VWN correlation functional (Vosko, Wilk, Nusair, 1980).

    Parametrization of the uniform electron gas correlation energy
    based on quantum Monte Carlo data of Ceperley and Alder (1980).

    Uses the Wigner-Seitz radius: r_s = (3/(4*pi*rho))^(1/3)

    The VWN-5 parametrization:
        e_c(r_s) = A/2 * {ln(x^2/X(x)) + 2b/Q * arctan(Q/(2x+b))
                         - bx_0/X(x_0) * [ln((x-x_0)^2/X(x)) + 2(b+2x_0)/Q * arctan(Q/(2x+b))]}

    where x = sqrt(r_s), X(x) = x^2 + bx + c, Q = sqrt(4c - b^2)

    Parameters for the paramagnetic case (unpolarized):
    """
    safe_rho = np.maximum(rho, 1e-30)
    r_s = (3.0 / (4.0 * np.pi * safe_rho)) ** (1.0 / 3.0)

    # VWN parameters (paramagnetic)
    A = 0.0310907
    b = 3.72744
    c = 12.9352
    x_0 = -0.10498

    x = np.sqrt(r_s)
    X = x**2 + b * x + c
    X_0 = x_0**2 + b * x_0 + c
    Q = np.sqrt(4 * c - b**2)

    e_c = A * (np.log(x**2 / X)
               + 2 * b / Q * np.arctan(Q / (2*x + b))
               - b * x_0 / X_0 * (np.log((x - x_0)**2 / X)
                                    + 2 * (b + 2*x_0) / Q * np.arctan(Q / (2*x + b))))

    # Potential: V_c = e_c - r_s/3 * de_c/dr_s
    dx_drs = 0.5 / x
    dX_dx = 2 * x + b
    de_c_dx = A * (2.0/x - dX_dx/X - 2*b*2/(Q**2 + (2*x+b)**2) * Q * (-1)
                   - b*x_0/X_0 * (2*(x-x_0)**(-1) - (2*x+b)/X
                                   - 2*(b+2*x_0)*2/(Q**2+(2*x+b)**2) * Q * (-1)))
    # Simplified potential using chain rule
    V_c = e_c - (r_s / 3.0) * de_c_dx * dx_drs

    return e_c, V_c


def pw92_correlation_3d(rho):
    """
    Perdew-Wang 1992 correlation functional.

    Simpler parametrization than VWN, often used with PBE:
        e_c(r_s) = -2A(1 + alpha1*r_s) * ln(1 + 1/(2A*(beta1*r_s^0.5 + ...)))
    """
    safe_rho = np.maximum(rho, 1e-30)
    r_s = (3.0 / (4.0 * np.pi * safe_rho)) ** (1.0 / 3.0)

    # PW92 parameters (unpolarized)
    A = 0.031091
    alpha1 = 0.21370
    beta1 = 7.5957
    beta2 = 3.5876
    beta3 = 1.6382
    beta4 = 0.49294

    rs12 = np.sqrt(r_s)
    rs32 = r_s * rs12
    rs2 = r_s * r_s

    denom = 2.0 * A * (beta1 * rs12 + beta2 * r_s + beta3 * rs32 + beta4 * rs2)
    e_c = -2.0 * A * (1.0 + alpha1 * r_s) * np.log(1.0 + 1.0 / denom)

    # Numerical potential
    drho = 1e-6 * safe_rho
    r_s_plus = (3.0 / (4.0 * np.pi * (safe_rho + drho))) ** (1.0 / 3.0)
    rs12_p = np.sqrt(r_s_plus)
    rs32_p = r_s_plus * rs12_p
    rs2_p = r_s_plus * r_s_plus
    denom_p = 2.0*A*(beta1*rs12_p + beta2*r_s_plus + beta3*rs32_p + beta4*rs2_p)
    e_c_plus = -2.0*A*(1.0 + alpha1*r_s_plus) * np.log(1.0 + 1.0/denom_p)

    # V_c = d(rho * e_c)/d rho (numerical derivative)
    V_c = (safe_rho + drho) * e_c_plus - safe_rho * e_c
    V_c = e_c + safe_rho * (e_c_plus - e_c) / drho

    return e_c, V_c


# ==========================================================================
# PART 2: GGA - Gradient Corrections
# ==========================================================================

def pbe_exchange(rho, grad_rho):
    """
    PBE exchange functional (Perdew, Burke, Ernzerhof, 1996).

    Adds a gradient correction to LDA exchange:
        E_x^PBE = integral rho * e_x^LDA * F_x(s) dr

    where s = |grad rho| / (2 * k_F * rho) is the reduced gradient
    and F_x(s) = 1 + kappa - kappa / (1 + mu*s^2/kappa)

    Parameters:
    kappa = 0.804 (from Lieb-Oxford bound)
    mu = 0.21951 (recovers LDA linear response)
    """
    C_x = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
    safe_rho = np.maximum(rho, 1e-30)

    # LDA exchange
    e_x_lda = -C_x * safe_rho ** (1.0 / 3.0)

    # Fermi wavevector: k_F = (3*pi^2*rho)^(1/3)
    k_F = (3.0 * np.pi**2 * safe_rho) ** (1.0 / 3.0)

    # Reduced gradient: s = |grad rho| / (2 * k_F * rho)
    s = np.abs(grad_rho) / (2.0 * k_F * safe_rho + 1e-30)

    # PBE enhancement factor
    kappa = 0.804
    mu = 0.21951
    F_x = 1.0 + kappa - kappa / (1.0 + mu * s**2 / kappa)

    e_x_pbe = e_x_lda * F_x

    return e_x_pbe, F_x


# ==========================================================================
# PART 3: Visualization and comparison
# ==========================================================================

def plot_jacobs_ladder():
    """Visualize Jacob's ladder of DFT functionals."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Ladder rungs
    rungs = [
        (0, 'Earth: Hartree\n(no XC)', 'gray', 'No XC, enormous errors'),
        (1, 'Rung 1: LDA\n(rho)', '#2196F3', 'Local density only\nEx: SVWN, SPW92'),
        (2, 'Rung 2: GGA\n(rho, |grad rho|)', '#4CAF50', 'Adds density gradient\nEx: PBE, BLYP, PW91'),
        (3, 'Rung 3: Meta-GGA\n(rho, |grad rho|, tau)', '#FF9800',
         'Adds kinetic energy density\nEx: TPSS, SCAN, r2SCAN'),
        (4, 'Rung 4: Hybrid\n(+ exact exchange)', '#F44336',
         'Mix in HF exchange\nEx: B3LYP, PBE0, HSE06'),
        (5, 'Rung 5: Double Hybrid\n(+ MP2 correlation)', '#9C27B0',
         'Include perturbative correlation\nEx: B2PLYP, XYG3'),
    ]

    for level, name, color, desc in rungs:
        y = level * 1.5
        # Rung (horizontal bar)
        ax.barh(y, 8, height=1.0, left=1, color=color, alpha=0.7,
                edgecolor='black', linewidth=2)
        # Label on rung
        ax.text(5, y, name, ha='center', va='center', fontsize=11,
                fontweight='bold', color='white' if level > 0 else 'gray')
        # Description
        ax.text(10, y, desc, ha='left', va='center', fontsize=10)

    # Vertical lines (ladder sides)
    ax.plot([1, 1], [-0.5, 8.5], 'k-', linewidth=3)
    ax.plot([9, 9], [-0.5, 8.5], 'k-', linewidth=3)

    # Arrow showing "chemical accuracy"
    ax.annotate('', xy=(0.3, 7.5), xytext=(0.3, 0),
                arrowprops=dict(arrowstyle='->', linewidth=2, color='darkblue'))
    ax.text(0.3, 3.75, 'Increasing\naccuracy\n(usually)', ha='center',
            va='center', fontsize=10, rotation=90, color='darkblue')

    ax.annotate('', xy=(0.7, 0), xytext=(0.7, 7.5),
                arrowprops=dict(arrowstyle='->', linewidth=2, color='darkred'))
    ax.text(0.7, 3.75, 'Increasing\ncost', ha='center',
            va='center', fontsize=10, rotation=90, color='darkred')

    ax.set_xlim(-0.5, 20)
    ax.set_ylim(-1, 9)
    ax.axis('off')
    ax.set_title("Jacob's Ladder of DFT Functionals", fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('step6_xc_functionals/jacobs_ladder.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_lda_exchange():
    """Visualize the LDA exchange energy density and potential."""
    rho = np.linspace(0.001, 1.0, 500)

    e_x, V_x = lda_exchange_3d(rho)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(rho, e_x, 'b-', linewidth=2)
    ax.set_xlabel('rho (electrons/bohr^3)')
    ax.set_ylabel('e_x (hartree/electron)')
    ax.set_title('LDA Exchange Energy Density')

    ax = axes[1]
    ax.plot(rho, V_x, 'r-', linewidth=2)
    ax.set_xlabel('rho (electrons/bohr^3)')
    ax.set_ylabel('V_x (hartree)')
    ax.set_title('LDA Exchange Potential')

    plt.tight_layout()
    plt.savefig('step6_xc_functionals/lda_exchange.png', dpi=150)
    plt.show()


def plot_pbe_enhancement():
    """
    Plot the PBE exchange enhancement factor F_x(s).

    F_x(s) tells us how much the exchange energy differs from LDA
    at a given reduced gradient s.
    """
    s = np.linspace(0, 5, 500)

    # PBE enhancement factor
    kappa = 0.804
    mu = 0.21951
    F_x_pbe = 1.0 + kappa - kappa / (1.0 + mu * s**2 / kappa)

    # B88 enhancement factor (for comparison)
    beta = 0.0042
    F_x_b88 = 1.0 + beta * s**2 / (1 + 6 * beta * s * np.arcsinh(s))

    # revPBE
    kappa_rev = 1.245
    F_x_revpbe = 1.0 + kappa_rev - kappa_rev / (1.0 + mu * s**2 / kappa_rev)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(s, np.ones_like(s), 'k--', linewidth=1, label='LDA (F_x = 1)')
    ax.plot(s, F_x_pbe, 'b-', linewidth=2, label='PBE')
    ax.plot(s, F_x_revpbe, 'r-', linewidth=2, label='revPBE')

    ax.axhline(y=1 + kappa, color='b', linestyle=':', alpha=0.5)
    ax.text(4.5, 1 + kappa + 0.02, f'PBE limit: {1+kappa}', color='b')
    ax.axhline(y=1 + kappa_rev, color='r', linestyle=':', alpha=0.5)

    ax.set_xlabel('Reduced gradient s', fontsize=12)
    ax.set_ylabel('Enhancement factor F_x(s)', fontsize=12)
    ax.set_title('GGA Exchange Enhancement Factors', fontsize=14)
    ax.legend(fontsize=12)
    ax.set_ylim(0.8, 3.0)

    plt.tight_layout()
    plt.savefig('step6_xc_functionals/pbe_enhancement.png', dpi=150)
    plt.show()

    print("""
    The enhancement factor F_x(s) modifies the LDA exchange:
        e_x^GGA = e_x^LDA * F_x(s)

    - s = 0 (uniform density): F_x = 1 (recover LDA)
    - s >> 0 (rapidly varying): F_x increases (more negative exchange)
    - PBE: F_x bounded by Lieb-Oxford condition (kappa = 0.804)
    - revPBE: Less strict bound, better for weak interactions

    Different GGA functionals differ mainly in F_x(s) and F_c(r_s, t).
    """)


def plot_correlation_energy():
    """Visualize the correlation energy as a function of density."""
    rho = np.linspace(0.001, 1.0, 500)

    e_c_pw92, _ = pw92_correlation_3d(rho)
    e_x_lda, _ = lda_exchange_3d(rho)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(rho, e_c_pw92, 'r-', linewidth=2, label='PW92 correlation')
    ax.plot(rho, e_x_lda, 'b-', linewidth=2, label='LDA exchange')
    ax.set_xlabel('rho (electrons/bohr^3)')
    ax.set_ylabel('Energy density (Ha/electron)')
    ax.set_title('Exchange vs Correlation Energy Density')
    ax.legend()

    ax = axes[1]
    ratio = np.abs(e_c_pw92) / (np.abs(e_x_lda) + 1e-30) * 100
    ax.plot(rho, ratio, 'g-', linewidth=2)
    ax.set_xlabel('rho (electrons/bohr^3)')
    ax.set_ylabel('|E_c / E_x| (%)')
    ax.set_title('Correlation as % of Exchange')
    ax.set_ylim(0, 20)

    plt.tight_layout()
    plt.savefig('step6_xc_functionals/correlation_vs_exchange.png', dpi=150)
    plt.show()

    print("""
    KEY OBSERVATION:
    Correlation is only ~5-15% of exchange in magnitude,
    but it's the harder part to approximate and is crucial for:
    - Bond energies (a few kcal/mol makes or breaks a prediction)
    - Reaction barriers
    - Dispersion/van der Waals interactions
    - Strongly correlated systems
    """)


def compare_functionals_summary():
    """Print a summary of when to use which functional."""
    print("=" * 70)
    print("  PRACTICAL GUIDE: Which Functional to Use?")
    print("=" * 70)
    print("""
    | Functional | Type      | Best for                        | Cost    |
    |------------|-----------|--------------------------------|---------|
    | SVWN       | LDA       | Metals, quick estimates         | Lowest  |
    | PBE        | GGA       | Solids, general purpose         | Low     |
    | BLYP       | GGA       | Organic molecules               | Low     |
    | PW91       | GGA       | Surfaces, adsorption            | Low     |
    | TPSS       | Meta-GGA  | Diverse chemistry               | Medium  |
    | SCAN       | Meta-GGA  | Solids + molecules              | Medium  |
    | B3LYP      | Hybrid    | Organic chemistry (most popular)| High    |
    | PBE0       | Hybrid    | General molecular properties    | High    |
    | HSE06      | Hybrid    | Band gaps in solids             | High    |
    | wB97X-D    | Hybrid+D  | Non-covalent interactions       | High    |
    | B2PLYP     | Double-H  | Thermochemistry benchmark       | Highest |

    RULES OF THUMB:
    1. Start with PBE for solids, B3LYP for molecules
    2. Add dispersion correction (-D3, -D4) for non-covalent interactions
    3. Use hybrid for band gaps and reaction barriers
    4. Meta-GGA (SCAN) is a good all-around modern choice
    5. Double hybrids for benchmark accuracy (if you can afford it)

    COMMON PITFALLS:
    - LDA overbinds (bonds too short, energies too negative)
    - GGA underbinds slightly (better than LDA)
    - Standard functionals MISS van der Waals interactions
    - Band gaps: LDA/GGA underestimate; hybrids are much better
    - Strongly correlated systems: ALL standard functionals struggle
    """)


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("  STEP 6: EXCHANGE-CORRELATION FUNCTIONALS")
    print("="*70)

    print("\n--- 6.1 Jacob's Ladder ---\n")
    plot_jacobs_ladder()

    print("\n--- 6.2 LDA Exchange ---\n")
    plot_lda_exchange()

    print("\n--- 6.3 GGA Enhancement Factor ---\n")
    plot_pbe_enhancement()

    print("\n--- 6.4 Correlation vs Exchange ---\n")
    plot_correlation_energy()

    print("\n--- 6.5 Practical Guide ---\n")
    compare_functionals_summary()

    print("\n" + "="*70)
    print("  KEY TAKEAWAYS")
    print("="*70)
    print("""
    1. LDA: Uses only the local density. Exact for uniform electron gas.
       Simple but surprisingly good for many properties.

    2. GGA: Adds the density gradient. PBE is the most widely used.
       Better for molecules and bond lengths.

    3. Hybrid: Mixes in exact (HF) exchange. B3LYP is the most cited
       functional in history. Better barriers and band gaps.

    4. The functional hierarchy (Jacob's ladder) trades cost for accuracy.

    5. No single functional is best for everything -- the choice depends
       on the property and system you're studying.

    Next --> Step 7: Basis Sets
    """)
