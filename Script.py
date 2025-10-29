
import random
import sympy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Parameters ----------
N_VALUES = list(range(2, 8))          # matrix sizes n = 2..7
ENTRY_MIN, ENTRY_MAX = -2, 2          # entries from -2 to 2 inclusive
TRIALS_PER_N = 500                    # number of random matrices per n
RANDOM_SEED = 123

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------- Helper Functions ----------
def random_int_matrix(n, a, b):
    """Generate a random n×n SymPy Matrix with integer entries in [a,b]."""
    data = np.random.randint(a, b+1, size=(n, n))
    return sp.Matrix(data.tolist())

def is_invertible(M: sp.Matrix) -> bool:
    """Check if matrix determinant is nonzero (invertible over ℝ)."""
    return M.det() != 0

def has_all_distinct_eigenvalues_numeric(M: sp.Matrix, tol=1e-8) -> bool:
    """Check if all eigenvalues are distinct using numeric computation."""
    A = np.array(M.tolist(), dtype=float)
    try:
        vals = np.linalg.eigvals(A)
    except np.linalg.LinAlgError:
        return False
    unique = []
    for v in vals:
        if not any(abs(v - u) < tol for u in unique):
            unique.append(v)
    return len(unique) == M.shape[0]

# ---------- Main Experiment ----------
records = []
for n in N_VALUES:
    inv_count = 0
    distinct_count = 0
    for _ in range(TRIALS_PER_N):
        A = random_int_matrix(n, ENTRY_MIN, ENTRY_MAX)
        if is_invertible(A):
            inv_count += 1
        if has_all_distinct_eigenvalues_numeric(A):
            distinct_count += 1
    records.append({
        "n": n,
        "trials": TRIALS_PER_N,
        "invertible_prop": inv_count / TRIALS_PER_N,
        "distinct_eigenvalues_prop": distinct_count / TRIALS_PER_N,
    })

# ---------- Results ----------
df = pd.DataFrame(records)
print("\n=== Random Integer Matrix Properties ===")
print(df)

# Save results
csv_path = "random_integer_matrix_results.csv"
df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")

# ---------- Plots ----------
# Invertibility plot
plt.figure()
plt.plot(df["n"], df["invertible_prop"], marker="o")
plt.title("Proportion of invertible n×n integer matrices\n(entries uniform in [-2,2])")
plt.xlabel("Matrix size n")
plt.ylabel("Proportion invertible")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("invertible_proportion.png", bbox_inches="tight")
plt.show()

# Distinct eigenvalues plot
plt.figure()
plt.plot(df["n"], df["distinct_eigenvalues_prop"], marker="o")
plt.title("Proportion with all distinct eigenvalues (numeric)\n(entries uniform in [-2,2])")
plt.xlabel("Matrix size n")
plt.ylabel("Proportion (all eigenvalues distinct)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("distinct_eigenvalues_proportion.png", bbox_inches="tight")
plt.show()

print("\nPlots saved as 'invertible_proportion.png' and 'distinct_eigenvalues_proportion.png'")
