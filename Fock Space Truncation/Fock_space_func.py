from math import sqrt

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import qutip as q
q.settings.colorblind_safe = True

from tqdm import tqdm

def init_values(N, M):
    """Compute ladder operations"""
    a = q.tensor(q.destroy(N), q.qeye(M))
    ad = q.tensor(q.create(N), q.qeye(M))
    
    b = q.tensor(q.qeye(N), q.destroy(M))
    bd = q.tensor(q.qeye(N), q.create(M))
    
    return a, ad, b, bd

def solve_qed(N, M, om_m=2*np.pi*1, om_c=2*np.pi*1, Om=2*np.pi*1, gamma=0, kappa=0, alpha=2, beta=2, n_th_a=0, t=np.linspace(0, 2*np.pi, 100)):
    """Solve master equation."""
    a, ad, b, bd = init_values(N, M)

    H = om_c * ad * a + om_m * bd * b - Om/2 * ad * a * (b + bd)
    
    rho0 = q.tensor(q.coherent(N, alpha), q.coherent(M, beta))

    # Decay/dissipation
    c_ops = []

        # Cavity decay
    if gamma > 0: 
        c_ops.append(np.sqrt(gamma * (1 + n_th_a)) * a)
        if n_th_a > 0:
            c_ops.append(np.sqrt(gamma * n_th_a) * ad)
    
        # Mirror decay
    if kappa > 0: 
        c_ops.append(np.sqrt(kappa * (1 + n_th_a)) * b)
        if n_th_a > 0:
            c_ops.append(np.sqrt(kappa * n_th_a) * bd)

    # Master equation
    options = q.Options(store_states=True, rtol=1e-5, atol=1e-7, nsteps=10000)
    result = q.mesolve(H, rho0, t, c_ops, [], options=options)
    return result

def result_qed(N_vals, M_vals, Ns, Ms):
    results = np.empty((len(N_vals), len(M_vals)), dtype=object)
    for i in tqdm(range(len(N_vals)), desc="Solving master equations"):
        for j in range(len(M_vals)):
            results[i, j] = solve_qed(int(Ns[i, j]), int(Ms[i, j]), 
                                     t=np.linspace(0, np.pi, 50))
    return results

def compute_entropy(results):
    """Compute entropy matrix with precomputed valid indices."""
    n, m = results.shape
    entropy_matrix = np.zeros((n, m))
    
    # Precompute all valid indices
    valid_indices = [(i, j) for i in range(n) for j in range(m) if results[i, j] is not None]
    
    for i, j in tqdm(valid_indices, desc="Progress"):
        states = results[i, j].states
        n_states = len(states)
        
        # Vectorized partial trace calculation
        ptrace_cavity = [q.ptrace(rho, 0) for rho in states]
        ptrace_mirror = [q.ptrace(rho, 1) for rho in states]
        
        # Vectorized entropy calculation
        entropy_cavity = np.array([q.entropy_linear(pt) for pt in ptrace_cavity])
        entropy_mirror = np.array([q.entropy_linear(pt) for pt in ptrace_mirror])
        
        # Einsum for efficient condition checking
        valid_cavity = (entropy_cavity >= 0).astype(int)
        valid_mirror = (entropy_mirror >= 0).astype(int)
        
        # Check if ALL entropies are non-negative
        condition = np.einsum('i,i->', valid_cavity, valid_mirror) == n_states
        entropy_matrix[i, j] = int(condition)
    
    return entropy_matrix

def plot_pos_entropy(Ns, Ms, entropy_matrix):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    contour1 = ax.contourf(Ns, Ms, entropy_matrix, levels=[-0.5, 0.5, 1.5], cmap='RdBu_r')
    ax.set_title("Positive Entropy Check")
    ax.set_xlabel("Cavity Dimension (N)")
    ax.set_ylabel("Mirror Dimension (M)")
    plt.colorbar(contour1, ax=ax, ticks=[0, 1])
    
    plt.tight_layout()
    plt.show()
    return

def compute_truncation(Ns, Ms, results, tolerance):
    """Compute truncation matrix with precomputed operators."""
    n, m = results.shape
    truncation_matrix_a = np.zeros((n, m))
    truncation_matrix_b = np.zeros((n, m))

    bound_lower, bound_upper = [1-tolerance, 1+tolerance]
    
    # Precompute all valid indices AND operators
    valid_data = []
    for i in range(n):
        for j in range(m):
            if results[i, j] is not None:
                N_dim, M_dim = int(Ns[i, j]), int(Ms[i, j])
                a_temp, ad_temp, b_temp, bd_temp = init_values(N_dim, M_dim)
                comm_a = q.commutator(a_temp, ad_temp)
                comm_b = q.commutator(b_temp, bd_temp)
                valid_data.append((i, j, comm_a, comm_b, results[i, j].states))
    
    for i, j, comm_a, comm_b, states in tqdm(valid_data, desc="Progress"):
        # Vectorized expectation calculation
        exp_comm_a = np.array([q.expect(comm_a, s) for s in states])
        exp_comm_b = np.array([q.expect(comm_b, s) for s in states])
        
        # Einsum counting
        count_a = np.einsum('i->',
                            ((bound_upper > exp_comm_a) & (exp_comm_a > bound_lower)).astype(int))
        count_b = np.einsum('i->', 
                            ((bound_upper > exp_comm_b) & (exp_comm_b > bound_lower)).astype(int))

        
        truncation_matrix_a[i, j] = count_a
        truncation_matrix_b[i, j] = count_b
    
    return [truncation_matrix_a, truncation_matrix_b]


def plot_fock_space_trunc(Ns, Ms, N_vals, M_vals, truncation_matrix):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

    N_last, M_last = int(len(N_vals)), int(len(M_vals))
    
    contour1 = ax1.contourf(Ns, Ms, truncation_matrix[0], levels=np.arange(0,50+10,10), cmap='RdBu_r')
    ax1.set_title("For Cavity", fontsize=14)
    ax1.set_yticks(np.linspace(0, M_last+1, int(M_last/10) + 1))
    ax1.set_xticks(np.linspace(0, N_last+1, int(N_last/10) + 1))
    ax1.set_xlabel("Cavity Dimension (N)")
    ax1.set_ylabel("Mirror Dimension (M)")
    plt.colorbar(contour1, ax=ax1)
    
    contour2 = ax2.contourf(Ns, Ms, truncation_matrix[1], levels=np.arange(0,50+10,10), cmap='RdBu_r')
    ax2.set_title("For Mirror", fontsize=14)
    ax2.set_yticks(np.linspace(0, M_last+1, int(M_last/10) + 1))
    ax2.set_xticks(np.linspace(0, N_last+1, int(N_last/10) + 1))
    ax2.set_xlabel("Cavity Dimension (N)")
    ax2.set_ylabel("Mirror Dimension (M)")
    plt.colorbar(contour2, ax=ax2)
    
    fig.suptitle('Fock Space Truncation Check', fontsize=16)
    
    plt.tight_layout()
    plt.show()
    return

def compute_truncation_both(Ns, Ms, results, tolerance):
    """Compute truncation matrix with precomputed operators."""
    n, m = results.shape
    truncation_matrix = np.zeros((n, m))

    bound_lower, bound_upper = [1-tolerance, 1+tolerance]
    
    # Precompute all valid indices AND operators
    valid_data = []
    for i in range(n):
        for j in range(m):
            if results[i, j] is not None:
                N_dim, M_dim = int(Ns[i, j]), int(Ms[i, j])
                a_temp, ad_temp, b_temp, bd_temp = init_values(N_dim, M_dim)
                comm_a = q.commutator(a_temp, ad_temp)
                comm_b = q.commutator(b_temp, bd_temp)
                valid_data.append((i, j, comm_a, comm_b, results[i, j].states))
    
    for i, j, comm_a, comm_b, states in tqdm(valid_data, desc="Progress"):
        # Vectorized expectation calculation
        exp_comm_a = np.array([q.expect(comm_a, s) for s in states])
        exp_comm_b = np.array([q.expect(comm_b, s) for s in states])
        
        # Einsum counting
        count_a = np.einsum('i->',
                            ((bound_upper > exp_comm_a) & (exp_comm_a > bound_lower)).astype(int))
        count_b = np.einsum('i->', 
                            ((bound_upper > exp_comm_b) & (exp_comm_b > bound_lower)).astype(int))

        
        truncation_matrix[i, j] = count_a + count_b
    
    return truncation_matrix


def plot_fock_space_trunc_both(Ns, Ms, N_vals, M_vals, truncation_matrix):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    N_last, M_last = int(len(N_vals)), int(len(M_vals))
    
    contour = ax.contourf(Ns, Ms, truncation_matrix, levels=np.arange(0,100+10,10), cmap='RdBu_r')
    ax.set_title("For Cavity & Mirror", fontsize=14)
    ax.set_yticks(np.linspace(0, M_last+1, int(M_last/10) + 1))
    ax.set_xticks(np.linspace(0, N_last+1, int(N_last/10) + 1))
    ax.set_xlabel("Cavity Dimension (N)")
    ax.set_ylabel("Mirror Dimension (M)")
    plt.colorbar(contour, ax=ax)
    
    fig.suptitle('Fock Space Truncation Check', fontsize=16)
    
    plt.tight_layout()
    plt.show()
    return

    
def find_optimal_truncation(truncation_matrix, entropy_matrix, N_vals, M_vals, min_value):
    """
    Find optimal N and M values with multiple criteria options.
    
    Parameters:
        truncation_matrix: 2D array of truncation values
        entropy_matrix: 2D array of entropy validation results
        N_vals: Array of N values (cavity dimensions)
        M_vals: Array of M values (mirror dimensions)
        min_value: Minimum acceptable truncation value
        
    Returns:
        optimal_points: List of tuples (N, M, truncation_value, entropy_value)
        optimal_indices: List of tuple indices (i, j)
    """
    
    # Find indices where BOTH conditions are satisfied
    valid_indices = np.where((truncation_matrix >= min_value) & (entropy_matrix == 0))
    
    # Get the corresponding values
    optimal_points = []
    optimal_indices = []
    
    for i, j in zip(valid_indices[0], valid_indices[1]):
        N_val = N_vals[i]
        M_val = M_vals[j]
        trunc_val = truncation_matrix[i, j]
        entropy_val = entropy_matrix[i, j]
        optimal_points.append((N_val, M_val, trunc_val, entropy_val))
        optimal_indices.append((i, j))
    
    return optimal_points, optimal_indices

def find_optimal_threshold(truncation_matrix, N_vals, M_vals, target_coverage=0.8):
    """
    Find the truncation threshold that covers target_coverage of points.
    """
    flat_trunc = truncation_matrix.flatten()
    flat_trunc.sort()
    
    # Find threshold that covers target_coverage of points
    threshold_index = int(len(flat_trunc) * (1 - target_coverage))
    optimal_threshold = flat_trunc[threshold_index]
    
    return optimal_threshold

def print_truncation_statistics(truncation_matrix, N_vals, M_vals):
    """Print beautiful statistics about the truncation matrix."""
    stats = {
        'Mean': np.mean(truncation_matrix),
        'Median': np.median(truncation_matrix),
        'Maximum': np.max(truncation_matrix),
        'Minimum': np.min(truncation_matrix),
        'Standard Deviation': np.std(truncation_matrix),
        'Points ≥ 40': np.sum(truncation_matrix >= 40),
        'Points ≥ 50': np.sum(truncation_matrix >= 50),
        'Points ≥ 60': np.sum(truncation_matrix >= 60),
        'Points ≥ 70': np.sum(truncation_matrix >= 70),
        'Total Points': truncation_matrix.size,
        'Percentage ≥ 40': f"{100 * np.mean(truncation_matrix >= 40):.1f}%",
        'Percentage ≥ 50': f"{100 * np.mean(truncation_matrix >= 50):.1f}%",
        'N Range': f"{N_vals.min():.1f} - {N_vals.max():.1f}",
        'M Range': f"{M_vals.min():.1f} - {M_vals.max():.1f}",
        'Grid Size': f"{len(N_vals)} × {len(M_vals)}"
    }
    
    print("┌" + "─" * 48 + "┐")
    print("│           TRUNCATION MATRIX STATISTICS         │")
    print("├" + "─" * 48 + "┤")
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"│ {key:<25} {value:>20.3f} │")
        else:
            print(f"│ {key:<25} {value:>20} │")
    
    print("└" + "─" * 48 + "┘")
    
    return stats

def print_optimal_points_summary(optimal_points, min_value):
    """Print a summary of optimal points."""
    if not optimal_points:
        print("No optimal points found!")
        return
    
    print("\n" + "="*60)
    print(f"OPTIMAL POINTS SUMMARY (Truncation ≥ {min_value})")
    print("="*60)
    
    # Sort by N+M for better presentation
    optimal_points.sort(key=lambda x: x[0] + x[1])
    
    print(f"Total optimal points found: {len(optimal_points)}")
    print("\nTop 5 smallest dimension combinations:")
    print("-" * 50)
    print("      N       M    Truncation    N+M       Pos. Ent.")
    print("-" * 50)
    
    for i, (N, M, trunc, ent) in enumerate(optimal_points[:5]):
        print(f"{i+1:2d}. {M:5.1f}   {N:5.1f}   {trunc:9.1f}   {N+M:5.1f}   {ent:9.1}")
    
    # Show statistics about optimal points
    M_vals = [point[0] for point in optimal_points]
    N_vals = [point[1] for point in optimal_points]
    trunc_vals = [point[2] for point in optimal_points]
    
    print("\nOptimal Points Statistics:")
    print("-" * 30)
    print(f"Min N:          {min(N_vals):.1f}")
    print(f"Max N:          {max(N_vals):.1f}")
    print(f"Min M:          {min(M_vals):.1f}")
    print(f"Max M:          {max(M_vals):.1f}")
    print(f"Avg Truncation: {np.mean(trunc_vals):.1f} ± {np.std(trunc_vals):.1f}")

def print_detailed_threshold_analysis(truncation_matrix):
    """Print detailed analysis of different thresholds."""
    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS")
    print("="*60)
    
    thresholds = [30, 40, 50, 60, 70, 80]
    total_points = truncation_matrix.size
    
    print("Threshold   Count   Percentage")
    print("-" * 30)
    
    for threshold in thresholds:
        count = np.sum(truncation_matrix >= threshold)
        percentage = 100 * count / total_points
        print(f"   ≥{threshold:2d}     {count:5d}     {percentage:6.1f}%")
    
    # Find the threshold that covers 80% of points
    flat_trunc = np.sort(truncation_matrix.flatten())
    threshold_80 = flat_trunc[int(0.2 * len(flat_trunc))]  # 80th percentile
    print(f"\nThreshold for 80% coverage: {threshold_80:.1f}")

def print_recommendations(truncation_matrix, N_vals, M_vals, min_value):
    """Print practical recommendations based on the analysis."""
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    # Find minimal dimensions that work
    optimal_points = find_optimal_truncation(truncation_matrix, N_vals, M_vals, min_value)[0]
    
    if optimal_points:
        # Find the point with smallest N+M
        optimal_points.sort(key=lambda x: x[0] + x[1])
        min_N, min_M, min_trunc = optimal_points[0]
        
        # Find the most balanced point
        balanced_points = sorted(optimal_points, key=lambda x: abs(x[0] - x[1]))
        bal_N, bal_M, bal_trunc = balanced_points[0]
        
        print("Recommended dimension choices:")
        print(f"1. Minimal dimensions:    N = {min_N:.1f}, M = {min_M:.1f}")
        print(f"   (Total dimension: {min_N + min_M:.1f}, Truncation: {min_trunc:.1f})")
        print(f"2. Balanced dimensions:   N = {bal_N:.1f}, M = {bal_M:.1f}")
        print(f"   (Difference: {abs(bal_N - bal_M):.1f}, Truncation: {bal_trunc:.1f})")
        
        # Safety margin recommendation
        safety_margin = min_trunc - 40
        if safety_margin < 5:
            print("\n⚠️  Warning: Minimal choice has low safety margin!")
            print("   Consider using slightly larger dimensions.")
        else:
            print(f"\n✓ Good safety margin: {safety_margin:.1f} above threshold")
    else:
        print("❌ No dimensions found with truncation ≥ 40!")
        print("   Consider increasing the dimension range.")











