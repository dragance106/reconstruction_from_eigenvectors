import numpy as np
import networkx as nx
from scipy.linalg import qr as _qr, svd as _svd

# ---------------------------
# Numeric RREF-like helpers
# ---------------------------
def numeric_rref_like(A, precision_digits=None, eps=None, rrqr_safety=10.0, reg_alpha=1.0):
    """
    Numerically-stable RREF-like decomposition for possibly noisy/irrational data.

    Returns dict with keys:
      - 'rank': numeric rank r
      - 'pivots': list of pivot column indices (in original column indexing)
      - 'free': list of free column indices
      - 'Rperm': r x n matrix representing [I_r | X] in original column order
      - 'X': r x (n-r) matrix mapping free vars -> pivot coeffs (in pivot-first permuted order)
      - 'perm': permutation array mapping new order to original indices
      - 'inv_perm': inverse permutation
      - 'tol_rank': rank tolerance used
      - 'tol_entry': entry tolerance used (for "effectively zero")
      - 'svals': singular values of A
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape

    # determine eps from precision_digits if not provided
    if eps is None:
        if precision_digits is not None:
            eps = 10.0 ** (-precision_digits)
        else:
            eps = np.finfo(float).eps

    # SVD for singular values and numeric rank
    U, s, Wt = _svd(A, full_matrices=False)
    s0 = s[0] if s.size > 0 else 0.0
    tol_rank = max(m, n) * s0 * eps * rrqr_safety
    r = int(np.sum(s > tol_rank))

    # RRQR to pick stable pivot columns (column permutation 'piv')
    if n == 0:
        piv = np.array([], dtype=int)
    else:
        _, _, piv = _qr(A, pivoting=True, mode='economic')
    piv = np.asarray(piv, dtype=int)

    basic_cols = list(piv[:r])
    free_cols = list(piv[r:])

    # permutation that brings basic_cols first
    perm = np.concatenate([basic_cols, free_cols]) if n > 0 else np.array([], dtype=int)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(n)

    # permuted matrix
    Aperm = A[:, perm] if n > 0 else A.copy()

    # partition
    if r > 0:
        A_B = Aperm[:, :r]
        A_F = Aperm[:, r:]
    else:
        A_B = np.zeros((m, 0))
        A_F = Aperm.copy()

    # compute X = -A_B^+ A_F via Tikhonov-regularized pseudoinverse
    if r == 0:
        X = np.zeros((0, n))
        Rperm = np.zeros((0, n), dtype=float)
        tol_entry = 0.0
    else:
        Ub, sb, Wtb = _svd(A_B, full_matrices=False)
        sb0 = sb[0] if sb.size > 0 else 0.0
        # regularization parameter
        lam = reg_alpha * max(A_B.shape) * sb0 * eps * rrqr_safety
        # compute regularized pseudoinverse action
        d = sb / (sb**2 + lam)           # shape (r,)
        UtA_F = (Ub.T @ A_F)             # shape r x (n-r)
        X_reg = - (Wtb.T * d) @ UtA_F    # shape r x (n-r)
        X = X_reg

        # RREF-like block in permuted order
        rref_perm = np.hstack([np.eye(r), X])   # r x n
        # Map back to original column order
        Rperm = np.zeros((r, n), dtype=float)
        Rperm[:, perm] = rref_perm

        scale = max(1.0, np.max(np.abs(rref_perm)))
        tol_entry = scale * eps * rrqr_safety

    return {
        'rank': r,
        'pivots': basic_cols,
        'free': free_cols,
        'Rperm': Rperm,
        'X': X,
        'perm': perm,
        'inv_perm': inv_perm,
        'tol_rank': tol_rank,
        'tol_entry': tol_entry,
        'svals': s
    }


def decide_rref_zero_one(Rperm, tol_zero=None, tol_one=None, thr01=None):
    """
    Given Rperm (r x n) matrix (RREF-like) decide which entries are:
      - effectively zero (abs <= tol_zero)
      - effectively one  (abs(val-1) <= tol_one)
      - ambiguous otherwise
    thr01 can be used to set both tolerances at once.
    Returns dict with boolean masks 'is_zero','is_one','is_ambig' and used tols.
    """
    R = np.asarray(Rperm, dtype=float)
    if thr01 is not None:
        tol_zero = tol_zero or thr01
        tol_one = tol_one or thr01
    if tol_zero is None:
        eps = np.finfo(float).eps
        scale = max(1.0, np.max(np.abs(R)))
        tol_zero = scale * eps * 100.0
    if tol_one is None:
        eps = np.finfo(float).eps
        scale = max(1.0, np.max(np.abs(R)))
        tol_one = scale * eps * 100.0

    is_zero = np.abs(R) <= tol_zero
    is_one = np.abs(R - 1.0) <= tol_one
    is_ambig = ~(is_zero | is_one)
    return {
        'is_zero': is_zero,
        'is_one': is_one,
        'is_ambig': is_ambig,
        'tol_zero': tol_zero,
        'tol_one': tol_one
    }


def decide_binary_from_expressions(expr_values,
                                   thr01=None,
                                   tol_rank=None,
                                   eps=None,
                                   rrqr_safety=10.0,
                                   scale=None):
    """
    Decide binary assignment (0 / 1 / ambiguous) for expression values that
    ideally should be 0 or 1. Uses scale-aware tolerances and optionally
    incorporates tol_rank (numeric-rank tolerance) into the decision.

    Parameters
    ----------
    expr_values : array-like
        Floating-point values (linear combinations of RREF entries).
    thr01 : float or None
        Optional user-controlled threshold. If provided it participates in
        tolerance selection (max with other sources).
    tol_rank : float or None
        Numeric-rank tolerance (absolute). If provided it is used as a lower
        bound on the zero/one tolerances (so that rank decisions are respected).
    eps : float or None
        Base machine/noise epsilon. If None, uses np.finfo(float).eps.
    rrqr_safety : float
        Safety multiplier for scale-based tolerance (default 10.0).
    scale : float or None
        Characteristic scale of the expressions (e.g. max abs of RREF block).
        If None the function uses max(1.0, max(abs(expr_values))).

    Returns
    -------
    dict with keys:
      'decision' : ndarray of shape expr_values -> values 0.0, 1.0, or np.nan
      'is_zero'  : boolean mask where decided zero
      'is_one'   : boolean mask where decided one
      'is_ambig' : boolean mask where ambiguous
      'tol_zero' : tolerance used for zero decision
      'tol_one'  : tolerance used for one decision
      'eps'      : eps used
      'scale'    : scale used
    """
    vals = np.asarray(expr_values, dtype=float)
    if eps is None:
        eps = np.finfo(float).eps

    # determine scale if not provided
    if scale is None:
        scale = max(1.0, np.max(np.abs(vals))) if vals.size > 0 else 1.0

    # scale-based tolerance
    tol_scale = scale * eps * rrqr_safety

    # determine tolerances: combine scale-based, numeric-rank, and user thr01
    candidates = [tol_scale]
    if thr01 is not None:
        candidates.append(float(thr01))
    if tol_rank is not None:
        # tol_rank may be a small absolute value; include it directly
        candidates.append(float(tol_rank))

    tol_zero = max(candidates)
    tol_one = tol_zero  # symmetric choice; can be adapted if needed

    # decisions
    is_zero = np.abs(vals) <= tol_zero
    is_one = np.abs(vals - 1.0) <= tol_one
    is_ambig = ~(is_zero | is_one)

    # assemble decision array: prefer zero if both tests true (rare)
    decision = np.full_like(vals, np.nan, dtype=float)
    decision[is_zero] = 0.0
    decision[is_one & (~is_zero)] = 1.0

    return {
        'decision': decision,
        'is_zero': is_zero,
        'is_one': is_one,
        'is_ambig': is_ambig,
        'tol_zero': tol_zero,
        'tol_one': tol_one,
        'eps': eps,
        'scale': scale
    }

# ---------------------------
# Main function (modified)
# ---------------------------
def irr_reconstruct(n, x, permissible_values=[0, 1],
                    precision_digits=None, thr01=1e-6,
                    rrqr_safety=10.0, reg_alpha=1.0):
    """
    Reconstruct solutions for adjacency (permissible_values default [0,1])
    from (possibly irrational or noisy) eigenvector matrix x.

    New optional arguments:
      - precision_digits: number of decimal digits of precision in x (used to set eps)
      - thr01: threshold for deciding values to be close to 0 or 1
      - rrqr_safety, reg_alpha: numeric routine tuning parameters
    """

    # initialize results list
    graphs = []

    # norms (squared norms of eigenvectors) - use float
    norms = np.array([np.dot(x[0:n, i], x[0:n, i]) for i in range(n)], dtype=float)

    # create matrix of the linear system: size (n + n_over_2) x (n + n_over_2)
    n_over_2 = n * (n - 1) // 2
    s = np.zeros((n + n_over_2, n + n_over_2), dtype=float)

    # upper-left block X^= : squared entries (columns are eigenvectors)
    # x is expected with eigenvectors as columns: x[0:n, i] is i-th eigenvector
    s[0:n, 0:n] = (x**2)[0:n, 0:n]

    # X^< block: products x_{i,k} x_{i,l} for k<l
    row_counter = n
    for k in range(n):
        for l in range(k + 1, n):
            s[row_counter, 0:n] = x[k, 0:n] * x[l, 0:n]
            # -I on the right block
            s[row_counter, row_counter] = -1.0
            row_counter += 1

    # Use numeric RREF-like routine instead of sympy.rref (works with floats/noisy input)
    res = numeric_rref_like(s, precision_digits=precision_digits,
                            rrqr_safety=rrqr_safety, reg_alpha=reg_alpha)

    n_rref = res['Rperm']   # shape (num_pivots, total_vars) in original column order
    tol_entry = res['tol_entry']
    tol_rank = res['tol_rank']

    pivot_vars = np.array(tuple(res['pivots']))
    num_pivots = res['rank']

    # Determine a scale for RREF-related expressions (used below in decisions)
    try:
        scale_R = max(1.0, np.max(np.abs(n_rref)))
    except Exception:
        scale_R = 1.0

    # Determine free variables as those not in pivot_vars
    free_vars = np.setdiff1d(np.arange(n_over_2+n), pivot_vars)
    num_free_vars = len(free_vars)

    # If no pivots found (degenerate), return empty list early
    if num_pivots == 0:
        return graphs

    # Extract pivot_coeffs: coefficients mapping free vars to pivot vars
    pivot_coeffs = n_rref[0:num_pivots, free_vars]  # shape (num_pivots, num_free_vars)
    # MAYBE WE DO NOT NEED MINUS SIGN IN FRONT OF n_rref IN FLOATING ARITHMETIC?

    # Treat tiny coefficients as zero using tol_entry
    small_mask = np.abs(pivot_coeffs) <= tol_entry
    pivot_coeffs[small_mask] = 0.0

    # pivot dependency counts (number of free vars that pivot depends on)
    unmarked_depends = np.count_nonzero(pivot_coeffs != 0.0, axis=1)

    # Set unmarked_depends for first n pivots (eigenvalues) to zero -- they are fixed
    unmarked_depends[0:n] = 0

    # mark pivots that already depend on no free variable with phase -1
    pivot_var_phase = np.zeros(num_pivots, dtype=int)
    pivot_var_phase[np.where(unmarked_depends == 0)] = -1

    num_pivots_unmarked = np.count_nonzero(pivot_var_phase != -1)

    # free variable phases and pivot phases arrays
    free_var_phase = np.zeros(num_free_vars, dtype=int)

    current_phase = 1
    # main phase assignment loop
    while num_pivots_unmarked > 0:
        # scoring: pick unmarked pivot with smallest unmarked_depends
        select_pivot = np.where(pivot_var_phase == 0, unmarked_depends, np.inf).argmin()

        # mark unmarked free vars that this pivot depends on
        for j in range(num_free_vars):
            if free_var_phase[j] == 0:
                if pivot_coeffs[select_pivot, j] != 0.0:
                    free_var_phase[j] = current_phase

                    # update unmarked_depends
                    for i in range(num_pivots):
                        if pivot_var_phase[i] == 0 and pivot_coeffs[i, j] != 0.0:
                            unmarked_depends[i] -= 1

        # assign current phase to pivot vars that now have unmarked_depends == 0
        for i in range(num_pivots):
            if pivot_var_phase[i] == 0 and unmarked_depends[i] == 0:
                pivot_var_phase[i] = current_phase
                num_pivots_unmarked -= 1

        current_phase += 1

    total_phases = current_phase

    # Prepare phased ordering arrays
    phased_pivot_vars = np.zeros(num_pivots, dtype=int)
    phased_free_vars = np.zeros(num_free_vars, dtype=int)

    phased_pivot_var_limit = np.zeros(total_phases, dtype=int)
    phased_free_var_limit = np.zeros(total_phases, dtype=int)

    current_pivot = 0
    current_free = 0
    for p in range(1, total_phases):
        for i in range(num_pivots):
            if pivot_var_phase[i] == p:
                phased_pivot_vars[current_pivot] = i
                current_pivot += 1

        phased_pivot_var_limit[p] = current_pivot

        for j in range(num_free_vars):
            if free_var_phase[j] == p:
                phased_free_vars[current_free] = j
                current_free += 1

        phased_free_var_limit[p] = current_free

    # add unreachable pivots (phase -1) at the end
    for i in range(num_pivots):
        if pivot_var_phase[i] == -1:
            phased_pivot_vars[current_pivot] = i
            current_pivot += 1

    # free variables always have a nonzero entry in their column,
    # so they will always obtain a positive phase

    # reorder pivot_coeffs according to phased orders
    phased_pivot_coeffs = pivot_coeffs[phased_pivot_vars, :]
    phased_pivot_coeffs = phased_pivot_coeffs[:, phased_free_vars]

    # Setup enumeration data structures
    phase_len = np.zeros(total_phases, dtype=int)
    max_phase_value = np.zeros(total_phases, dtype=int)

    base = len(permissible_values)
    for p in range(1, total_phases):
        phase_len[p] = phased_free_var_limit[p] - phased_free_var_limit[p - 1]
        max_phase_value[p] = base ** phase_len[p] - 1

    phase_value = np.zeros(total_phases + 1, dtype=int)
    phased_free_var_value = np.zeros(num_free_vars, dtype=int)  # free variables correspond to edges

    # enumeration / backtracking
    p = 1
    phase_value[p] = 0
    go_back = False
    total_nodes = 1

    # Precompute permissible values as numpy array
    perm_vals = np.array(permissible_values, dtype=float)

    while True:
        if p == 0:
            break

        if p == total_phases:
            # found a complete assignment of free variables
            # compute pivot values for this assignment
            phased_pivot_var_value = phased_pivot_coeffs @ phased_free_var_value

            # reconstruct full_solution vector (size n + n_over_2)
            full_solution = np.zeros(n + n_over_2, dtype=float)

            for i in range(num_pivots):
                full_solution[pivot_vars[phased_pivot_vars[i]]] = phased_pivot_var_value[i]

            for j in range(num_free_vars):
                full_solution[free_vars[phased_free_vars[j]]] = phased_free_var_value[j]

            # reconstruct eigenvalues (first n variables scaled by norms)
            eigs = np.zeros(n, dtype=float)
            eigs[0:n] = full_solution[0:n] * norms[0:n]

            # reconstruct adjacency by mapping free/pivot variables to nearest permissible values
            A = np.zeros((n, n), dtype=int)
            entry_counter = n
            ambiguous_flag = False
            for k in range(n):
                for l in range(k + 1, n):
                    val = full_solution[entry_counter]
                    # decide using scale-aware routine: returns 0/1/NaN
                    dec = decide_binary_from_expressions([val],
                                                         thr01=thr01,
                                                         tol_rank=tol_rank,
                                                         eps=None,
                                                         rrqr_safety=rrqr_safety,
                                                         scale=scale_R)
                    chosen = dec['decision'][0]
                    if np.isnan(chosen):
                        ambiguous_flag = True
                        break
                    # chosen is 0.0 or 1.0
                    A[k, l] = int(chosen)
                    A[l, k] = int(chosen)
                    entry_counter += 1
                if ambiguous_flag:
                    break

            if ambiguous_flag:
                # reject this candidate and backtrack
                p = p - 1
                go_back = True
                continue

            # Construct graph
            g = nx.from_numpy_array(A)

            # check isomorphism to previously found graphs
            same_old = False
            for previous in graphs:
                if nx.vf2pp_is_isomorphic(g, previous['graph']):
                    same_old = True
                    break

            if not same_old:
                graphs.append({'graph': g,
                               'adjacency': A,
                               'eigenvalues': eigs,
                               'eigenvectors': x})

            # backtrack
            p = p - 1
            go_back = True
            continue

        if go_back:
            phase_value[p] += 1
            if phase_value[p] > max_phase_value[p]:
                p -= 1
                go_back = True
            else:
                go_back = False
                total_nodes += 1
        else:
            # enter new node: decode phase_value into digits base 'base'
            s_val = f'{np.base_repr(phase_value[p], base=base)}'.zfill(phase_len[p])
            for i in range(phase_len[p]):
                phased_free_var_value[phased_free_var_limit[p - 1] + i] = permissible_values[int(s_val[i])]

            # compute pivot values for this phase (only those pivots in the phase)
            start_p = phased_pivot_var_limit[p - 1]
            end_p = phased_pivot_var_limit[p]
            num_piv_in_phase = end_p - start_p
            if num_piv_in_phase > 0:
                rows = phased_pivot_coeffs[start_p:end_p, 0:phased_free_var_limit[p]]
                cols = phased_free_var_value[0:phased_free_var_limit[p]]
                phased_pivot_var_value = rows @ cols
            else:
                phased_pivot_var_value = np.array([], dtype=float)

            # Decide using scale-aware routine whether pivot values are close to permissible values
            if phased_pivot_var_value.size > 0:
                dec = decide_binary_from_expressions(phased_pivot_var_value,
                                                     thr01=thr01,
                                                     tol_rank=tol_rank,
                                                     eps=None,
                                                     rrqr_safety=rrqr_safety,
                                                     scale=scale_R)
                # node acceptable if none ambiguous and all decisions are in permissible set
                if np.any(dec['is_ambig']):
                    node_acceptable = False
                else:
                    # ensure that the decided values are among permissible_values (they should be 0/1)
                    node_acceptable = np.all(np.isin(dec['decision'], perm_vals))
            else:
                node_acceptable = True

            if node_acceptable:
                p += 1
                phase_value[p] = 0
                go_back = False
                total_nodes += 1
            else:
                phase_value[p] += 1
                if phase_value[p] > max_phase_value[p]:
                    p -= 1
                    go_back = True
                else:
                    go_back = False
                    total_nodes += 1

    return graphs
