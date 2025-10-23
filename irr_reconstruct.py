import numpy as np
from scipy.linalg import qr, lu_factor, lu_solve
import itertools
from typing import Iterator
import networkx as nx


def select_invertible_submatrix_with_solver(A, cond_threshold=1e8, max_swaps=50, verbose=False):
    """
    Select an invertible n×n submatrix B from a full-column-rank m×n matrix A (m>n),
    using QR with column pivoting and optional max-volume refinement. Returns B,
    selected row indices, its condition number, and a solver function solve_B(b).

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Full column rank matrix (m>n).
    cond_threshold : float, optional
        If cond(B) > cond_threshold, attempt max-volume swaps to improve B.
    max_swaps : int, optional
        Maximum number of refinement swaps in the max-volume heuristic.
    verbose : bool, optional
        Print diagnostic messages.

    Returns
    -------
    B : ndarray, shape (n, n)
        Chosen invertible submatrix of A.
    row_idx : ndarray, shape (n,)
        Indices of the rows of A used to form B (in Python 0-based indexing).
    condB : float
        Condition number of B (2-norm).
    solve_B : callable
        Function solve_B(b) that returns x solving B x = b. Accepts b of shape (n,) or (n, k).
    """
    m, n = A.shape
    if m <= n:
        raise ValueError("Matrix must have more rows than columns (m > n).")

    # Step 1: QR with column pivoting on A^T to choose candidate rows
    Q, R, piv = qr(A.T, pivoting=True)
    row_idx = np.array(piv[:n], dtype=int)
    B = A[row_idx, :]
    condB = np.linalg.cond(B)
    if verbose:
        print(f"[QRCP] initial cond(B) = {condB:.2e}")

    # Step 2: Max-volume style refinement if needed
    if condB > cond_threshold:
        if verbose:
            print("[maxvol] starting refinement...")
        # We'll use SVD to form a stable pseudoinverse of B during iteration.
        for swap_iter in range(max_swaps):
            # Compute B_inv-like object via SVD (stable even if some s are small)
            U, s, Vt = np.linalg.svd(B, full_matrices=False)
            tol = np.finfo(float).eps * max(B.shape) * s[0]
            s_inv = np.array([1 / si if si > tol else 0.0 for si in s])
            B_inv = (Vt.T * s_inv) @ U.T  # pseudo-inverse

            # projection matrix P = A @ B_inv  (m x n)
            P = A @ B_inv
            absP = np.abs(P)

            # find largest magnitude entry of P
            i, j = np.unravel_index(np.argmax(absP), absP.shape)
            maxval = absP[i, j]

            # stopping criterion: locally maximal volume when no entry > 1
            if maxval <= 1.0 + 1e-12:
                if verbose:
                    print(f"[maxvol] reached local max (iter {swap_iter}), max |P| = {maxval:.3g}")
                break

            # else swap row: bring row i into the j-th selected slot
            if verbose:
                print(f"[maxvol] swap iter {swap_iter + 1}: swapping in row {i} for slot {j} (|P|={maxval:.3g})")
            row_idx[j] = i
            B = A[row_idx, :]
            condB = np.linalg.cond(B)
            if verbose:
                print(f"  cond(B) after swap = {condB:.2e}")

            # stop early if we reached desired conditioning
            if condB < cond_threshold:
                if verbose:
                    print("[maxvol] cond threshold reached; stopping refinement.")
                break
        else:
            if verbose:
                print("[maxvol] reached maximum swap iterations.")

    # Final diagnostics
    condB = np.linalg.cond(B)
    if verbose:
        print(f"[final] cond(B) = {condB:.2e}")

    # Prepare solver using LU factorization (fast & stable for many solves)
    try:
        lu_and_piv = lu_factor(B)  # will raise if B is singular
    except Exception as e:
        # As a fallback, try SVD-based least-squares solve; but normally B is invertible.
        raise RuntimeError("Failed to factor final B (may be singular).") from e

    def solve_B(b):
        """
        Solve B x = b for x. Accepts b shape (n,) or (n, k).
        Returns x with shape (n,) or (n, k) respectively.
        """
        b_arr = np.asarray(b)
        # allow column vector shapes
        if b_arr.ndim == 1:
            if b_arr.shape[0] != n:
                raise ValueError(f"b must have length {n} for B x = b.")
            return lu_solve(lu_and_piv, b_arr)
        elif b_arr.ndim == 2:
            if b_arr.shape[0] != n:
                raise ValueError(f"b must have {n} rows for B x = b (got {b_arr.shape[0]}).")
            return lu_solve(lu_and_piv, b_arr)
        else:
            raise ValueError("b must be a 1D or 2D array.")

    return B, row_idx, condB, solve_B


def generate_b_vectors(rows, n: int, permissible_values=[0,1]) -> Iterator[np.ndarray]:
    """
    Generate all n-dimensional vectors b (dtype int, entries 0/1) such that:
      - b[i] == 0 whenever rows[i] < n (forced zero)
      - for positions with rows[i] >= n, b[i] iterates through all 0/1 combinations

    Parameters
    ----------
    rows : array-like of shape (n,)
        Row indices returned by your selection routine (0-based integers).
    n : int
        The threshold/number of columns (same `n` as matrix A has columns).

    Yields
    ------
    b : np.ndarray of shape (n,), dtype=int
        A vector of 0/1 values meeting the rule above.
    """
    rows = np.asarray(rows, dtype=int)
    if rows.shape[0] != n:
        raise ValueError("rows must have length n")

    # positions that are free to be 0/1
    var_positions = [i for i, r in enumerate(rows) if r >= n]
    k = len(var_positions)

    # base vector with forced zeros
    base = np.zeros(n, dtype=int)

    # iterate over all 2^k assignments
    for bits in itertools.product(permissible_values, repeat=k):
        b = base.copy()
        for pos, bit in zip(var_positions, bits):
            b[pos] = bit
        yield b


def binary_mask_real(A, tol=1e-8):
    """
    Return a boolean mask of the same shape as A: True where entry is close to 0 or 1.
    Assumes A is real-valued. Uses absolute tolerance tol.
    """
    A = np.asarray(A)
    return np.isclose(A, 0, atol=tol) | np.isclose(A, 1, atol=tol)


def is_binary_matrix_real(A, tol=1e-8, repair=False):
    """
    Check whether all entries of a real array A belong to {0,1} up to absolute tolerance tol.

    Parameters
    ----------
    A : array-like (real)
        Input array (any shape). Must be real-valued.
    tol : float
        Absolute tolerance for closeness to 0 or 1.
    repair : bool
        If True, and all entries are within tol of {0,1}, return a repaired integer
        array (dtype=int) with values rounded to 0/1. If there are entries outside tol,
        raise ValueError.

    Returns
    -------
    all_binary : bool
        True if every entry is within tol of 0 or 1.
    (optional) bad_idx : ndarray of shape (k, ndim)
        Indices of offending entries (only if return_bad_indices True).
    (optional) repaired : ndarray
        Integer array (0/1) when repair=True and repair succeeded.
    """
    A = np.asarray(A)

    mask = binary_mask_real(A, tol=tol)
    all_ok = bool(mask.all())
    out = (all_ok,)

    if repair:
        # choose closest of {0,1} for each entry
        repaired = (np.abs(A - 1) < np.abs(A - 0)).astype(int)
        out += (repaired,)

    if len(out) == 1:
        return out[0]
    else:
        return out


def multiset_equal_sort(a, b, *, atol=1e-8, rtol=0.0, equal_nan=False):
    """
    Return True if arrays a and b contain the same elements (as a multiset),
    up to rounding error, and possibly in different orders.
    Uses sorting + elementwise comparison with np.allclose.

    Parameters
    ----------
    a, b : array-like
        Input arrays (any shape). They are flattened before comparison.
    atol : float
        Absolute tolerance for closeness.
    rtol : float
        Relative tolerance for closeness (passed to np.allclose).
    equal_nan : bool
        If True, treat NaNs as equal.

    Returns
    -------
    bool
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size != b.size:
        return False
    # stable numeric sort
    asort = np.sort(a)
    bsort = np.sort(b)
    return np.allclose(asort, bsort, atol=atol, rtol=rtol, equal_nan=equal_nan)


def irr_reconstruct(n, x, permissible_values=[0,1],
                    thr01=1e-6,
                    skip_disconnected=False,
                    skip_regular=False):
    # auxiliary command to print full numpy matrices without truncation...
    # np.set_printoptions(threshold=np.inf)

    # initialize empty list of graph (dictionaries) found for this matrix
    graphs = []

    # a row of squared norms of eigenvectors - we'll need that to reconstruct eigenvalues
    norms = np.array([np.dot(x[0:n, i], x[0:n, i]) for i in range(n)], dtype=float)

    # create matrix of the linear system
    # [X^=]
    # [X^<]
    n_over_2 = n*(n-1)//2
    s = np.zeros((n + n_over_2, n), dtype=float)

    # X^= up
    s[0:n, 0:n] = x**2

    # products x_{i,k} x_{i,l} in the first n columns for each k<l
    # here x_{i,k} is the k-th entry of eigenvector x_i
    # since the eigenvectors are taken as columns of the matrix x,
    # x_{i,k} is actually x[k, i] in Python terminology
    # the same ordering will be respected later to create an adjacency matrix from the free and pivot variables

    # X^< down
    row_counter = n
    for k in range(n):
        for l in range(k+1, n):
            s[row_counter, 0:n] = x[k, 0:n] * x[l, 0:n]

            # move to the next row
            row_counter = row_counter + 1

    # system matrix s has full rank - find its invertible submatrix
    B, rows, condB, solve_B = select_invertible_submatrix_with_solver(s, cond_threshold=1e6, verbose=False)

    # now go through all possible choices for adjacency matrix entries
    # that correspond to selected rows whose indices are at least n
    # (the first n rows correspond to diagonal entries of adjacency matrix, which are all 0)
    for b in generate_b_vectors(rows, n, permissible_values=permissible_values):
        L = solve_B(b)
        A = x @ np.diag(L) @ x.T
        eigs = L * norms

        # check that A represents adjacency matrix
        ok_A, A_repaired = is_binary_matrix_real(A, tol=thr01, repair=True)

        if ok_A:
            # check also that all the diagonal entries are zeros - don't want no loops!
            if bool(np.all(np.abs(np.diag(A)) <= thr01)):

                # Construct the graph using networkX
                g = nx.from_numpy_array(A_repaired)

                # Skipping disconnected graphs?
                if not skip_disconnected or nx.is_connected(g):

                    # Skipping regular graphs?
                    degrees = A_repaired.sum(axis=1)
                    if not skip_regular or not(np.allclose(degrees, degrees[0], atol=thr01)):

                        # Is it isomorphic to a previous graph corresponding to this matrix?
                        same_old = False
                        for previous in graphs:
                            if nx.vf2pp_is_isomorphic(g, previous['graph']):
                                same_old = True
                                break

                        if not same_old:
                            # finally, we can add this as a new graph corresponding to the matrix x
                            # keep track of other useful data as well
                            graphs.append({'graph': g,
                                           'adjacency': A_repaired,
                                           'eigenvalues': eigs,
                                           'eigenvectors': x})

    return graphs
