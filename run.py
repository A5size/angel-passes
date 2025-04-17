# This code was refactored by ChatGPT (OpenAI).

import numpy as np
import ctypes
import math
import scipy.special as sp
import matplotlib.pyplot as plt 

# Load the external Fortran library (libfort.so)
lib = np.ctypeslib.load_library("./libfort.so", ".")

def update(alpha, beta, mat, states, n):
    """
    Calls the Fortran function to update the state of all agents by one time step.
    Each agent's state (0: silence, 1: speaking) is updated based on a Markov transition.
    """
    lib.update.argtypes = [ctypes.POINTER(ctypes.c_double),
                           ctypes.POINTER(ctypes.c_double),
                           np.ctypeslib.ndpointer(dtype=np.float64),
                           np.ctypeslib.ndpointer(dtype=np.float64),
                           ctypes.POINTER(ctypes.c_int32)]
    lib.update.restype = ctypes.c_void_p
    lib.update(ctypes.byref(ctypes.c_double(alpha)),
               ctypes.byref(ctypes.c_double(beta)),
               mat,
               states,
               ctypes.byref(ctypes.c_int32(n)))

def batch_update(alpha, beta, mat, states, n, steps):
    """
    Calls the Fortran function to update the state of all agents over multiple time steps.
    Used for equilibration (burn-in phase) before collecting data.
    """
    lib.batch_update.argtypes = [ctypes.POINTER(ctypes.c_double),
                                 ctypes.POINTER(ctypes.c_double),
                                 np.ctypeslib.ndpointer(dtype=np.float64),
                                 np.ctypeslib.ndpointer(dtype=np.float64),
                                 ctypes.POINTER(ctypes.c_int32),
                                 ctypes.POINTER(ctypes.c_int32)]
    lib.batch_update.restype = ctypes.c_void_p
    lib.batch_update(ctypes.byref(ctypes.c_double(alpha)),
                     ctypes.byref(ctypes.c_double(beta)),
                     mat,
                     states,
                     ctypes.byref(ctypes.c_int32(n)),
                     ctypes.byref(ctypes.c_int32(steps)))

def calculate_ha(N, alpha, beta, theta):
    """Computes the interpolating function h_alpha(theta)."""
    h = ((alpha + beta) * theta) / (1.0 - theta + alpha + beta)
    return N * alpha * (2.0 * (1.0 - h) + h / (alpha + beta))

def calculate_hb(N, alpha, beta, theta):
    """Computes the interpolating function h_beta(theta)."""
    h = ((alpha + beta) * theta) / (1.0 - theta + alpha + beta)
    return N * beta * (2.0 * (1.0 - h) + h / (alpha + beta))

def find_theta_for_ha_equals_1(N, alpha, beta, tol=1e-6):
    """Finds theta such that h_alpha(theta) = 1 via bisection search."""
    low, high = 0.0, 1.0
    while high - low > tol:
        mid = (low + high) / 2.0
        if calculate_ha(N, alpha, beta, mid) < 1.0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0

def find_theta_for_hb_equals_1(N, alpha, beta, tol=1e-6):
    """Finds theta such that h_beta(theta) = 1 via bisection search."""
    low, high = 0.0, 1.0
    while high - low > tol:
        mid = (low + high) / 2.0
        if calculate_hb(N, alpha, beta, mid) < 1.0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0

def get_f_theta(N, alpha, beta, theta):
    """
    Computes the steady-state distribution f_theta(n1),
    representing the probability of n1 agents speaking in the Café θ model.
    """
    h = ((alpha + beta) * theta) / (1.0 - theta + alpha + beta)
    a = N * alpha * (2.0 * (1.0 - h) + h / (alpha + beta))
    b = N * beta  * (2.0 * (1.0 - h) + h / (alpha + beta))     
    p_list = []
    for i in range(N + 1):
        x = (1.0 - i / N) * alpha + (i / N) * (1.0 - beta)
        ly = (a - 1.0) * np.log(x) + (b - 1.0) * np.log(1.0 - x) - (np.log(N) + sp.betaln(a, b) - np.log(1.0 - alpha - beta))
        p_list.append(np.exp(ly))
    p_array = np.array(p_list)
    return p_array / np.sum(p_array)

def main():
    """Main simulation loop: initializes, equilibrates, runs, and visualizes the model."""
    np.random.seed(42)

    # === Simulation parameters ===
    sec2step = 100
    eq_steps = 5 * 60 * sec2step
    pro_steps = 60 * 60 * sec2step

    N = 64
    alpha = 1.0 / (5.0 * sec2step)
    beta  = 1.0 / (5.0 * sec2step)
    theta = 0.80

    # === Compute theoretical distribution first (needed for P₀) ===
    f_theta = get_f_theta(N, alpha, beta, theta)

    # === Compute transition threshold values ===
    theta_silence_onset_alpha = find_theta_for_ha_equals_1(N, alpha, beta)
    theta_silence_onset_beta  = find_theta_for_hb_equals_1(N, alpha, beta)

    # === Print parameters and derived values ===
    print("=== Simulation Parameters ===")
    print(f"Number of agents                        : {N}")
    print(f"Alpha (silence → talk)                  : {alpha:.6f}")
    print(f"Beta  (talk → silence)                  : {beta:.6f}")
    print(f"Theta (used)                            : {theta:.4f}")
    print(f"Theta for h_alpha = 1                   : {theta_silence_onset_alpha:.6f}")
    print(f"Theta for h_beta  = 1                   : {theta_silence_onset_beta:.6f}")
    print("--- Derived Quantities ---")
    print(f"Expected silence duration (*)           : {1/alpha/sec2step:.2f} sec")
    print(f"Expected speaking duration (*)          : {1/beta/sec2step:.2f} sec")
    print(f"Expected number of speakers             : {N * alpha / (alpha + beta):.2f}")
    print(f"Expected silence segment length         : {1 / (alpha * N) / sec2step:.6f} sec")
    print(f"Silence probability (P₀)                : {f_theta[0]:.6f}")
    print("(*) Idealized per-agent expectation assuming no interactions")
    print("===============================")

    # === Generate interaction matrix based on 2D spatial layout ===
    x, y = np.random.rand(N), np.random.rand(N)
    influence_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (x[j] - x[i])**2 + (y[j] - y[i])**2
            influence_matrix[i, j] = influence_matrix[j, i] = 1.0 / d2
    for i in range(N):
        influence_matrix[i, :] = (1.0 - theta) * influence_matrix[i, :] / np.sum(influence_matrix[i, :])
        influence_matrix[i, i] = theta

    # === Initialize and equilibrate ===
    states = np.random.choice([0.0, 1.0], size=N)
    print("Equilibrating...")
    batch_update(alpha, beta, influence_matrix, states, N, eq_steps)
    print("Equilibration complete.")

    # === Main simulation ===
    print("Starting main simulation...")
    trajectory = []
    for _ in range(pro_steps):
        update(alpha, beta, influence_matrix, states, N)
        trajectory.append(np.sum(states))
    print("Simulation complete.")

    # === Visualization ===
    trajectory = np.array(trajectory)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].plot([v / sec2step for v in range(pro_steps)], trajectory, color='blue', alpha=0.6, lw=0.5)
    axs[0].set_title("Number of Speaking Agents Over Time")
    axs[0].set_xlabel("Time (seconds)")
    axs[0].set_ylabel("Number of speakers")

    axs[1].hist(trajectory, bins=N+1, range=(-0.5, N + 0.5), density=True, alpha=0.6, label="Simulation")
    axs[1].plot(range(N + 1), f_theta, color='red', alpha=0.6, label="Theoretical $f_\\theta(n_1)$", linewidth=2)
    axs[1].set_title("Empirical vs Theoretical Distribution")
    axs[1].set_xlabel("Number of speakers")
    axs[1].set_ylabel("Probability")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
