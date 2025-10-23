import numpy as np
import math
import sympy as sp
from typing import Optional

def integer_orthogonal_eigenbasis(
    A: np.ndarray,
    make_primitive: bool = True,
    return_object_dtype: bool = False
) -> np.ndarray:
    """
    Given a square integer symmetric matrix A (numpy array), compute an integer
    eigenbasis whose columns are mutually orthogonal (standard dot product).
    Assumes all eigenvalues are integers and A is diagonalizable.

    Parameters
    ----------
    A : np.ndarray
        Square integer numpy array (symmetric).
    make_primitive : bool
        If True (default), each integer eigenvector is divided by the gcd of its
        entries so it is primitive.
    return_object_dtype : bool
        If True, returns a numpy array with dtype=object (Python ints) even if
        entries exceed np.int64. If False (default), will raise OverflowError
        when int64 cannot hold values.

    Returns
    -------
    Q : np.ndarray
        n x n matrix whose columns are integer, mutually orthogonal eigenvectors.
        dtype is np.int64 unless return_object_dtype is True (or OverflowError).
    """
    # --- basic checks
    if not (isinstance(A, np.ndarray) and A.ndim == 2 and A.shape[0] == A.shape[1]):
        raise ValueError("A must be a square numpy array.")
    n = A.shape[0]

    if not np.issubdtype(A.dtype, np.integer):
        raise ValueError("A must have integer dtype.")
    if not np.array_equal(A, A.T):
        raise ValueError("This routine assumes A is symmetric (undirected adjacency).")

    # Convert to sympy Matrix for exact arithmetic
    M = sp.Matrix(A.tolist())
    I = sp.eye(n)

    # Get eigenvalues (exact) and sort deterministically (ascending)
    ev_mult = M.eigenvals()  # dict: eigen -> multiplicity
    eigenvalues = sorted(ev_mult.keys(), key=lambda x: int(sp.nsimplify(x)))

    def exact_gram_schmidt(vs):
        """Exact Gram-Schmidt over rationals for list of sympy column vectors."""
        orth = []
        for v in vs:
            w = sp.Matrix(v)  # copy
            for u in orth:
                num = u.dot(w)        # rational
                den = u.dot(u)        # rational (nonzero)
                if den == 0:
                    continue
                w = w - (num / den) * u
                w = sp.Matrix([sp.simplify(x) for x in w])
            # keep nonzero
            if any(x != 0 for x in w):
                orth.append(w)
        return orth

    integer_vectors = []  # list of Python-int column lists

    for ev in eigenvalues:
        nullbasis = (M - ev * I).nullspace()
        if len(nullbasis) == 0:
            continue
        orth_rational = exact_gram_schmidt(nullbasis)

        for vr in orth_rational:
            # vr entries are rational sympy expressions. Convert to integers by clearing denominators.
            nums = []
            dens = []
            for entry in vr:
                num, den = sp.together(entry).as_numer_denom()
                # make sure they are sympy Integers (or convertible)
                nums.append(int(sp.Integer(num)))
                dens.append(int(sp.Integer(den)))
            # lcm of denominators
            lcm_den = 1
            for d in dens:
                lcm_den = math.lcm(lcm_den, d)
            # multiply each entry by lcm_den carefully: entry = num/den -> num * (lcm_den // den)
            int_entries = [nums[k] * (lcm_den // dens[k]) for k in range(len(nums))]
            # make primitive if requested
            if make_primitive:
                g = 0
                for x in int_entries:
                    g = math.gcd(g, abs(int(x)))
                if g > 1:
                    int_entries = [int(x // g) for x in int_entries]
            else:
                int_entries = [int(x) for x in int_entries]
            integer_vectors.append(int_entries)

    # sanity: produced n vectors?
    if len(integer_vectors) != n:
        raise ValueError(f"Could not produce full eigenbasis (got {len(integer_vectors)} vectors, expected {n}). "
                         "Matrix may be defective or not diagonalizable over rationals.")

    # Build numpy matrix
    # check for int64 overflow
    max_abs = max(max(abs(x) for x in col) for col in integer_vectors)
    INT64_MAX = 2**63 - 1

    if return_object_dtype:
        Q = np.column_stack([np.array(col, dtype=object) for col in integer_vectors])
        return Q
    else:
        if max_abs > INT64_MAX:
            raise OverflowError("Integer eigenvectors exceed int64 range; set return_object_dtype=True to get Python ints.")
        Q = np.column_stack([np.array(col, dtype=np.int64) for col in integer_vectors])

    # final orthogonality check (exact integer dot-products)
    for i in range(n):
        for j in range(i+1, n):
            if int(np.dot(Q[:, i].astype(np.int64), Q[:, j].astype(np.int64))) != 0:
                raise AssertionError("Orthogonality check failed: produced vectors are not orthogonal.")

    return Q


# ----------------------------
# Example (K3)
if __name__ == "__main__":
    A = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=np.int64)
    Q = integer_orthogonal_eigenbasis(A)
    print(f'Q:\n{Q}')
