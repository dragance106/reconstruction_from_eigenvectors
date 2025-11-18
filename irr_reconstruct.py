import numpy as np
from scipy.linalg import qr, lu_factor, lu_solve, svd
import itertools
from typing import Iterator, List, Optional, Tuple, Union
import networkx as nx


def select_invertible_submatrix_maximize_min_singular(
    A: np.ndarray,
    preferred_rows: Optional[List[int]] = None,
    min_sigma_threshold: float = 1e-12,
    max_swaps: int = 200,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, float, callable]:
    """
    Select an invertible n x n submatrix B from full-column-rank A (m>n),
    prioritizing inclusion of preferred_rows while maximizing (or preserving)
    the smallest singular value sigma_min(B) for numerical stability.

    Returns:
      B:    (n,n) ndarray, the selected submatrix
      row_idx: (n,) ndarray, indices of rows chosen from A
      sigma_min: float, smallest singular value of B
      solve_B: callable to solve B x = b (uses LU factorization)

    Parameters:
      A: (m,n) with m > n and full column rank
      preferred_rows: list of row indices to prefer (defaults to [0..n-1])
      min_sigma_threshold: minimal acceptable sigma_min; if sigma_min < this,
                           the algorithm still returns best found but prints
                           warning (if verbose).
      max_swaps: max number of single-row swap attempts in greedy phase
      verbose: print diagnostics
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    if m <= n:
        raise ValueError("A must have more rows than columns (m > n).")

    if preferred_rows is None:
        preferred_set = set(range(n))
    else:
        preferred_set = set(int(r) for r in preferred_rows)

    # Step 1: initial selection by QR with column pivoting on A^T
    Q, R, piv = qr(A.T, pivoting=True)
    row_idx = np.array(piv[:n], dtype=int)
    B = A[row_idx, :]

    # function to compute smallest singular value robustly
    def sigma_min_of(mat: np.ndarray) -> float:
        # For small n this is fine; for larger n consider using specialized routine.
        s = svd(mat, compute_uv=False)
        return float(np.min(s)) if s.size > 0 else 0.0

    sigma_min = sigma_min_of(B)
    if verbose:
        print(f"[init] selected rows {row_idx.tolist()}; sigma_min = {sigma_min:.3e}; preferred_count = {sum(1 for r in row_idx if r in preferred_set)}/{n}")

    # Helper: count preferred rows selected
    def preferred_count(sel_idx):
        return sum(1 for r in sel_idx if r in preferred_set)

    # Greedy single-row swaps: try to increase preferred rows without decreasing sigma_min
    swaps = 0
    improved = True
    while improved and swaps < max_swaps:
        improved = False
        current_pref = preferred_count(row_idx)

        # candidate preferred rows not currently selected
        not_sel_pref = [r for r in preferred_set if r not in row_idx]
        # positions in row_idx that are non-preferred (candidates to replace)
        sel_nonpref_pos = [j for j, r in enumerate(row_idx) if r not in preferred_set]

        if not not_sel_pref or not sel_nonpref_pos:
            break  # nothing to do

        best_candidate = None  # (sigma_new, j, p, new_row_idx)
        for p in not_sel_pref:
            for j in sel_nonpref_pos:
                trial_idx = row_idx.copy()
                trial_idx[j] = p
                B_trial = A[trial_idx, :]
                try:
                    sigma_trial = sigma_min_of(B_trial)
                except Exception:
                    sigma_trial = -1.0
                # accept only if sigma_trial >= sigma_min (no degradation)
                if sigma_trial >= sigma_min - 1e-18:
                    # prefer increases in preferred_count first, then larger sigma
                    new_pref = preferred_count(trial_idx)
                    if new_pref > current_pref:
                        # immediate best: prioritize any swap that increases preferred_count
                        if best_candidate is None or (new_pref > best_candidate[4]) or (sigma_trial > best_candidate[0]):
                            best_candidate = (sigma_trial, j, p, trial_idx, new_pref)
                    else:
                        # if no increase in pref_count we still may pick sigma increase
                        if best_candidate is None:
                            best_candidate = (sigma_trial, j, p, trial_idx, new_pref)
                        else:
                            # keep the one with larger sigma_trial
                            if sigma_trial > best_candidate[0]:
                                best_candidate = (sigma_trial, j, p, trial_idx, new_pref)

        if best_candidate is not None:
            sigma_new, slot_j, pref_row, new_idx, new_pref = best_candidate
            # Apply swap
            row_idx = new_idx
            B = A[row_idx, :]
            sigma_min = sigma_new
            swaps += 1
            improved = True
            if verbose:
                print(f"[greedy-swap #{swaps}] inserted preferred row {pref_row} into slot {slot_j}; sigma_min={sigma_min:.3e}; preferred_count={new_pref}/{n}")
        else:
            # nothing acceptable found that does not degrade sigma_min
            break

    # Optional max-volume-style refinement but only accept swaps that increase sigma_min
    # (Compute projection via pseudo-inverse and attempt to swap the max entry as before,
    #  but only if sigma_min improves.)
    if verbose:
        print("[maxvol-like] attempting refinement that increases sigma_min (no degradation allowed).")
    maxvol_swaps = 0
    while maxvol_swaps < max_swaps:
        # compute SVD-based pseudo-inverse of current B
        U, svals, Vt = np.linalg.svd(B, full_matrices=False)
        tol = np.finfo(float).eps * max(B.shape) * (svals[0] if svals.size else 0.0)
        s_inv = np.array([1.0/s if s > tol else 0.0 for s in svals])
        B_pinv = (Vt.T * s_inv) @ U.T                 # pseudo-inverse (n x n)
        P = A @ B_pinv                                # (m x n)
        absP = np.abs(P)
        i, j = np.unravel_index(np.argmax(absP), absP.shape)
        maxval = absP[i, j]
        if maxval <= 1.0 + 1e-12:
            if verbose:
                print(f"[maxvol-like] local optimum (max|P|={maxval:.3g}).")
            break

        # candidate swap: bring row i into slot j
        candidate_idx = row_idx.copy()
        candidate_idx[j] = int(i)
        B_candidate = A[candidate_idx, :]
        try:
            sigma_candidate = sigma_min_of(B_candidate)
        except Exception:
            sigma_candidate = -1.0

        # Accept only if sigma_candidate > sigma_min (strict improvement)
        if sigma_candidate > sigma_min + 1e-18:
            row_idx = candidate_idx
            B = A[row_idx, :]
            sigma_min = sigma_candidate
            maxvol_swaps += 1
            if verbose:
                print(f"[maxvol-like swap #{maxvol_swaps}] swapped in row {i} for slot {j}; sigma_min improved to {sigma_min:.3e}; preferred_count={preferred_count(row_idx)}/{n}")
            # continue refinement
            continue
        else:
            # no improvement in sigma_min possible via this max-entry swap
            if verbose:
                print(f"[maxvol-like] proposed swap (row {i}->slot {j}) does not improve sigma_min ({sigma_candidate:.3e} <= {sigma_min:.3e}); stopping refinement.")
            break

    # Final diagnostics
    final_pref = preferred_count(row_idx)
    if verbose:
        print(f"[final] sigma_min = {sigma_min:.3e}; preferred_count = {final_pref}/{n}; selected rows: {row_idx.tolist()}")
    if sigma_min < min_sigma_threshold and verbose:
        print(f"[warning] sigma_min ({sigma_min:.3e}) below threshold ({min_sigma_threshold:.3e}).")

    # Prepare solver: using LU (works well when B is well conditioned)
    try:
        lu_and_piv = lu_factor(B)
    except Exception as e:
        raise RuntimeError("Final B appears singular or LU failed.") from e

    def solve_B(b: Union[np.ndarray, List[float]]):
        b_arr = np.asarray(b)
        if b_arr.ndim == 1:
            if b_arr.shape[0] != n:
                raise ValueError(f"b must have length {n}.")
            return lu_solve(lu_and_piv, b_arr)
        elif b_arr.ndim == 2:
            if b_arr.shape[0] != n:
                raise ValueError(f"b must have {n} rows.")
            return lu_solve(lu_and_piv, b_arr)
        else:
            raise ValueError("b must be 1D or 2D array.")

    return B, row_idx, float(sigma_min), solve_B


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
    B, rows, condB, solve_B = select_invertible_submatrix_maximize_min_singular(s, verbose=True)

    print(f'for path_{n} we got {(rows>=n).sum()} variable rows')

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
