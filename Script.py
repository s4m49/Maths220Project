import random
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt


N_VALUES = list(range(2, 7))      
ENTRY_MIN, ENTRY_MAX = -2, 2      
TRIALS_PER_N = 300                
RANDOM_SEED = 123

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def random_int_matrix(n, a, b) -> sp.Matrix:
    """Random n×n SymPy Matrix with integer entries in [a,b]."""
    data = np.random.randint(a, b + 1, size=(n, n))
    return sp.Matrix(data.tolist())

def random_int_vector(n, a, b) -> sp.Matrix:
    """Random n×1 SymPy column vector with integer entries in [a,b]."""
    return sp.Matrix(np.random.randint(a, b + 1, size=(n, 1)).tolist())

def is_invertible(M: sp.Matrix) -> bool:
    """Invertible over ℚ (and hence over ℝ) iff exact determinant ≠ 0."""
    return M.det() != 0

def has_all_distinct_eigenvalues_exact(M: sp.Matrix) -> bool:
    """True iff algebraic multiplicity of every eigenvalue is 1 (exact, symbolic)."""
    mults = M.eigenvals()  
    n = M.shape[0]
    return (sum(mults.values()) == n) and all(m == 1 for m in mults.values())

def sequence_is_independent(n: int, k: int, a: int, b: int) -> bool:
    """Draw k random vectors in Z^n with entries in [a,b]; check if they are LI."""
    cols = [random_int_vector(n, a, b) for _ in range(k)]
    M = sp.Matrix.hstack(*cols)
    return M.rank() == k


matrix_records = []
for n in N_VALUES:
    inv_count = 0
    distinct_count = 0
    for _ in range(TRIALS_PER_N):
        A = random_int_matrix(n, ENTRY_MIN, ENTRY_MAX)
        if is_invertible(A):
            inv_count += 1
        if has_all_distinct_eigenvalues_exact(A):
            distinct_count += 1
    matrix_records.append({
        "n": n,
        "trials": TRIALS_PER_N,
        "invertible_prop": inv_count / TRIALS_PER_N,
        "distinct_eigenvalues_prop": distinct_count / TRIALS_PER_N,
    })

df_mats = pd.DataFrame(matrix_records)


vector_records = []
for n in N_VALUES:
    for k in range(1, n + 1):
        indep_count = 0
        for _ in range(TRIALS_PER_N):
            if sequence_is_independent(n, k, ENTRY_MIN, ENTRY_MAX):
                indep_count += 1
        vector_records.append({
            "n": n,
            "k": k,
            "trials": TRIALS_PER_N,
            "independent_prop": indep_count / TRIALS_PER_N,
        })

df_vecs = pd.DataFrame(vector_records)


print("\n=== Matrix properties (entries uniform in [{},{}]) ===".format(ENTRY_MIN, ENTRY_MAX))
print(df_mats)

print("\n=== Sequence independence probabilities ===")
print(df_vecs.pivot(index="n", columns="k", values="independent_prop").round(3))


df_mats.to_csv("matrix_results.csv", index=False)
df_vecs.to_csv("vector_sequence_results.csv", index=False)
print("\nSaved 'matrix_results.csv' and 'vector_sequence_results.csv'.")

plt.figure()
plt.plot(df_mats["n"], df_mats["invertible_prop"], marker="o")
plt.title("Proportion of invertible n×n integer matrices\n(entries uniform in [{},{}])".format(ENTRY_MIN, ENTRY_MAX))
plt.xlabel("Matrix size n")
plt.ylabel("Proportion invertible")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("plot_invertible_vs_n.png", bbox_inches="tight")

plt.figure()
plt.plot(df_mats["n"], df_mats["distinct_eigenvalues_prop"], marker="o")
plt.title("Proportion with all distinct eigenvalues (exact)\n(entries uniform in [{},{}])".format(ENTRY_MIN, ENTRY_MAX))
plt.xlabel("Matrix size n")
plt.ylabel("Proportion (all eigenvalues distinct)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("plot_distinct_eigs_vs_n.png", bbox_inches="tight")

plt.figure()
for k in sorted(df_vecs["k"].unique()):
    df_k = df_vecs[df_vecs["k"] == k].sort_values("n")
    plt.plot(df_k["n"], df_k["independent_prop"], marker="o", label=f"k={k}")
plt.title("P(sequence of k integer vectors in ℤⁿ is linearly independent)\n(entries uniform in [{},{}])".format(ENTRY_MIN, ENTRY_MAX))
plt.xlabel("n")
plt.ylabel("Proportion independent")
plt.legend(title="Sequence length k")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("plot_vector_independence.png", bbox_inches="tight")

plt.show()
print("\nPlots saved: 'plot_invertible_vs_n.png', 'plot_distinct_eigs_vs_n.png', 'plot_vector_independence.png'")
