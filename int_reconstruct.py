import numpy as np
import sympy as sp
import networkx as nx


def int_reconstruct(n, x, permissible_values=[0, 1],
                    skip_disconnected=False,
                    skip_regular=False):
    """
    The method reconstructs all solutions to the system
    [X^=  O][Lambda] = [0]
    [X^< -I][A     ] = [0]
    where the entries of A have to belong to the set of permissible values.
    The matrix X^= consists of n rows
    where the entries of row k are equal to the squares x_{i,k}^2.
    The matrix X^<*> consists of n columns
    where the entries of column i are equal to the products
    x_{i,k} x_{i,l} for a given ordering of all pairs (k,l) with k<l.

    When permissible_values=[0,1] these solutions represent
    adjacency matrices of all simple graphs whose adjacency eigenvectors are given by x.
    The lambda part of the solution describes the eigenvalues corresponding to x.

    :param n: the number of vertices.
    :param x: the set of eigenvectors.
    :param permissible_values: the values allowed for the entries of A.
    :return: the list of all graphs whose adjacency eigenvectors are given by x.
             each graph is represented by a dictionaries containing
             the graph as a networkX graph ("graph"),
             the adjacency matrix ("adjacency"),
             the eigenvalues ("eigenvalues"),
             and a copy of the eigenvectors ("eigenvectors")
    """

    # auxiliary command to print full numpy matrices without truncation...
    # np.set_printoptions(threshold=np.inf)

    # initialize empty list of graph (dictionaries) found for this matrix
    graphs = []

    # a row of squared norms of eigenvectors - we'll need that to reconstruct eigenvalues
    norms = np.array([np.dot(x[0:n, i], x[0:n, i]) for i in range(n)], dtype=np.intp)

    # create matrix of the linear system
    # [X^=  O]
    # [X^< -I]
    n_over_2 = n*(n-1)//2
    s = np.zeros((n + n_over_2, n+n_over_2), dtype=np.intp)

    # X^= up left
    s[0:n, 0:n] = x**2

    # products x_{i,k} x_{i,l} in the first n columns for each k<l
    # here x_{i,k} is the k-th entry of eigenvector x_i
    # since the eigenvectors are taken as columns of the matrix x,
    # x_{i,k} is actually x[k, i] in Python terminology
    # the same ordering will be respected later to create an adjacency matrix from the free and pivot variables

    # X^< down left
    row_counter = n
    for k in range(n):
        for l in range(k+1, n):
            s[row_counter, 0:n] = x[k, 0:n] * x[l, 0:n]

            # -I down right
            s[row_counter, row_counter] = -1

            # move to the next row
            row_counter = row_counter + 1

    # A VERY IMPORTANT STEP:
    # convert to sympy and find the reduced row-echelon form
    s_rref, pivot_vars = sp.Matrix(s).rref()    # s_rref entries are sympy.Rational objects
    n_rref = sp.matrix2numpy(s_rref)

    # sympy also has .echelon_form(with_pivots=True),
    # which gives the same pivot_vars,
    # but produces enormously large entries in the echelon form...

    # matrix of the system has rank at least 1+n_over_2,
    # so there will be at least 1+n_over_2 pivot_vars and at most n-1 free vars
    # still, pivot_vars indices need not necessarily be equal to 0,...,n_over_2,...
    # since rref will occasionally skip some columns
    # however, eigenvalue variables are always among pivot_vars,
    # since the first n columns of s contain multiples of the eigenvectors,
    # hence we have a full rank there - which has to be covered by pivots all along

    # free variables are those which are not pivot_vars
    free_vars = np.setdiff1d(np.arange(n_over_2 + n), pivot_vars)

    # pivot_vars and free_vars are now the indices of variables in the system
    num_pivots = len(pivot_vars)
    num_free_vars = len(free_vars)

    # GREEDY FAIL-FAST ENUMERATION OF INDEP.VAR.CHOICES:
    # 1. GREEDY DERIVATION OF GENERATION PHASES:
    # - Initialize phase_counter to 0,
    #   set all free_vars as unmarked,
    #   set all pivot_vars as unmarked
    # - Repeat:
    #   - select unmarked pivot depending on the smallest number of unmarked free_vars
    #   - mark these free_vars with phase_counter
    #   - mark with the same phase_counter
    #     all those unmarked pivot_vars that depend only on currently and previously marked free_vars
    #   - increase phase_counter by 1,
    #     and repeat as long as there are unmarked free_vars/pivot_vars

    # phase counters exist for both pivot and free variables
    pivot_var_phase = np.zeros(num_pivots, dtype=np.intp)
    free_var_phase = np.zeros(num_free_vars, dtype=np.intp)

    # use -n_rref[pivot_rows][free_vars] to compute pivots from the vector of free variables
    # pivot rows are always 0,...,num_pivots-1
    pivot_coeffs = -n_rref[0:num_pivots, free_vars]

    # pivot i depends on free variable j if pivot_coeffs[i, j]!=0
    # PAY ATTENTION: IT IS NOT pivot_coeffs[pivot_vars[i], free_vars[j]], BUT pivot_coeffs[i, j]!

    # for each pivot count on how many unmarked free variables they depend
    unmarked_depends = np.count_nonzero(pivot_coeffs, axis=1)

    # recall that the first n variables actually represent the graph eigenvalues
    # (and they are all pivot variables for certain),
    # so there is no need to check them for having permissible values (0,1 for graphs)
    # set their unmarked_depends to zero to have them excluded from further computations
    unmarked_depends[0:n] = 0

    # it's possible that some further pivots do not depend on any free variable
    # (e.g., if their value is fixed to zero),
    # hence unmarked_depends may contain further zero entries already at the start
    # all such pivots (eigenvalues + fixed to zero) can be immediately marked with an unreachable phase counter
    pivot_var_phase[np.where(unmarked_depends==0)] = -1

    # it's also possible that some free variables do not influence any pivot
    # hence one should actually count how many pivots have not been marked yet!
    num_pivots_unmarked = np.count_nonzero(pivot_var_phase!=-1)

    current_phase = 1
    while num_pivots_unmarked > 0:
        # select unmarked pivot with the smallest value of unmarked_depends (which is necessarily positive!)
        select_pivot = np.where(pivot_var_phase == 0, unmarked_depends, np.inf).argmin()

        # mark unmarked free_vars that this pivot depends on
        for j in range(num_free_vars):
            if free_var_phase[j] == 0:                             # unmarked free variable
                if pivot_coeffs[select_pivot, j] != 0:             # that this pivot depends on
                    free_var_phase[j] = current_phase              # mark such free variable!

                    # update unmarked_depends array
                    for i in range(num_pivots):
                        if pivot_var_phase[i] == 0:
                            if pivot_coeffs[i, j] != 0:
                                unmarked_depends[i] = unmarked_depends[i] - 1

        # all unmarked_depends that have become zero can be given the current phase counter
        # (this will at least include select_pivot!)
        for i in range(num_pivots):
            if pivot_var_phase[i] == 0 and unmarked_depends[i] == 0:
                pivot_var_phase[i] = current_phase
                num_pivots_unmarked = num_pivots_unmarked - 1

        # go to the next phase
        current_phase = current_phase + 1

    total_phases = current_phase

    # GREEDY FAIL-FAST ENUMERATION OF INDEP.VAR.CHOICES:
    # 2. FURTHER DATA STRUCTURE PREPARATION
    # - rearrange pivot and free variables according to their phase counters (with phase -1 appearing at the end)
    # - set info on the ranges of pivot/free variables with a given phase in the new ordering
    # - rearrange pivot_coeffs according to the new pivot/free ordering,
    #   as I want pivot[i] with phase k to be equal to
    #   pivot_coeffs[i, 0:free_vars_with_phases_up_to_k] * free_values[0:free_vars_with_phases_up_to_k]
    # - save the inverse orderings of pivot and free variables,
    #   so that I can later construct the correct adjacency matrix

    # reorder the variables according to their phases
    phased_pivot_vars = np.zeros(num_pivots, dtype=np.intp)
    phased_free_vars = np.zeros(num_free_vars, dtype=np.intp)

    phased_pivot_var_limit = np.zeros(total_phases, dtype=np.intp)
    phased_free_var_limit = np.zeros(total_phases, dtype=np.intp)

    # go through all identified phases>=1
    current_pivot = 0
    current_free = 0
    for p in range(1, total_phases):
        # pick up pivot variables from this phase
        for i in range(num_pivots):
            if pivot_var_phase[i] == p:
                phased_pivot_vars[current_pivot] = i
                current_pivot = current_pivot + 1

        # set up the limit of this phase
        phased_pivot_var_limit[p] = current_pivot

        # pick up free variables from this phase
        for j in range(num_free_vars):
            if free_var_phase[j] == p:
                phased_free_vars[current_free] = j
                current_free = current_free + 1

        # set up the limit of this phase
        phased_free_var_limit[p] = current_free

    # at the end, pick up the pivot variables with the unreachable phase -1
    for i in range(num_pivots):
        if pivot_var_phase[i] == -1:
            phased_pivot_vars[current_pivot] = i
            current_pivot = current_pivot + 1

    # note that the phases have values from 1 to total_phases-1 (inclusive)
    # var_limits are there so that they can be used in slicing, for example,
    # all free variables with phases up to k are
    #   phased_free_vars[0 : phased_free_var_limit[p]],
    # while all pivot variables with phase equal to k are
    #   phased_pivot_vars[phased_free_var_limit[p-1] : phased_free_var_limit[p]]

    # rearrange pivot_coeffs to respect the orders from phased_pivot_vars and phased_free_vars
    phased_pivot_coeffs = pivot_coeffs[phased_pivot_vars, :]
    phased_pivot_coeffs = phased_pivot_coeffs[:, phased_free_vars]

    # after this reordering,
    # the pivot i (in phased order, i.e., phased_pivot_vars[i] in the original order) with phase k>=1 will be equal to
    # phased_pivot_coeffs[i, 0:phased_free_var_limit[k]] * free_var_values[0:phased_free_var_limit[k]]


    # GREEDY FAIL-FAST ENUMERATION OF INDEP.VAR.CHOICES
    # 3. THE ACTUAL ENUMERATION WITH GENERATION PHASES USING ITERATIVE BACKTRACKING

    # initialization
    phase_len = np.zeros(total_phases, dtype=np.intp)
    max_phase_value = np.zeros(total_phases, dtype=np.intp)

    # permissible values may have two elements for simple graphs or three elements for signed graphs!
    base = len(permissible_values)          # this length is the basis of the number system in which we enumerate phases
    for p in range(1, total_phases):
        phase_len[p] = phased_free_var_limit[p] - phased_free_var_limit[p-1]
        max_phase_value[p] = base**phase_len[p] - 1

    phase_value = np.zeros(total_phases+1, dtype=np.intp)
    phased_free_var_value = np.zeros(num_free_vars, dtype=np.intp)

    # the initial node of the search tree
    p = 1
    phase_value[p] = 0
    go_back = False     # did we backtrack to this node?

    # testing measure - how many nodes did we visit in total?
    total_nodes = 1

    while True:
        if p==0:
            break       # backtracking has just been completed

        if p==total_phases:
            # complete solution found

            # trivial if all phase values are zeros, just skip
            # do not skip the trivial graph!
            if True: # np.count_nonzero(phase_value)!=0:
                # only free variables have been set in previous phases,
                # so compute the pivot variables completely now
                phased_pivot_var_value = np.matmul(phased_pivot_coeffs, phased_free_var_value)

                # reconstruct the full solution
                full_solution = np.zeros(n+n_over_2)

                # pivot variables
                for i in range(num_pivots):
                    full_solution[pivot_vars[phased_pivot_vars[i]]] = phased_pivot_var_value[i]

                for j in range(num_free_vars):
                    full_solution[free_vars[phased_free_vars[j]]] = phased_free_var_value[j]

                # reconstruct the eigenvalues which represent the first n variables
                eigs = np.zeros(n, dtype=np.intp)
                eigs[0:n] = full_solution[0:n] * norms[0:n]

                # reconstruct the adjacency matrix
                A = np.zeros((n, n), dtype=np.intp)
                entry_counter = n           # the first n original variables were eigenvalues
                for k in range(n):
                    for l in range(k + 1, n):
                        A[k, l] = full_solution[entry_counter]
                        A[l, k] = full_solution[entry_counter]
                        entry_counter += 1

                # Construct the graph using networkX
                g = nx.from_numpy_array(A)

                # Skipping disconnected graphs?
                if not skip_disconnected or nx.is_connected(g):

                    # Skipping regular graphs?
                    degrees = A.sum(axis=1)
                    if not skip_regular or not(np.allclose(degrees, degrees[0])):

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
                                           'adjacency': A,
                                           'eigenvalues': eigs,
                                           'eigenvectors': x})

            # backtrack to the previous node
            p = p-1
            go_back = True
            continue

        if go_back:
            # since you're backtracking, you have already checked that this node was acceptable
            # hence, move to the next node, if possible
            phase_value[p] = phase_value[p] + 1

            if phase_value[p] > max_phase_value[p]:
                # backtrack to the previous node
                p = p-1
                go_back = True
            else:
                # we are entering a new node, hence no backtracking
                go_back = False
                total_nodes = total_nodes + 1
        else:
            # we are entering a new node, so first check if it is acceptable
            # first convert phase_value into a string of digits with given base
            s = f'{np.base_repr(phase_value[p], base=base)}'.zfill(phase_len[p])

            # then use list of permissible value to populate phased_free_var_values
            for i in range(phase_len[p]):
                phased_free_var_value[phased_free_var_limit[p-1] + i] = permissible_values[int(s[i])]

            # phase_value translated in binary to free variables for this phase <- WORKS ONLY FOR SIMPLE GRAPHS!
            # s = np.binary_repr(phase_value[p], width=phase_len[p])
            # for i in range(phase_len[p]):
            #     phased_free_var_value[phased_free_var_limit[p-1] + i] = int(s[i])

            # compute pivots for this phase
            phased_pivot_var_value = np.matmul(phased_pivot_coeffs[phased_pivot_var_limit[p-1]:phased_pivot_var_limit[p], 0:phased_free_var_limit[p]],
                                               phased_free_var_value[0:phased_free_var_limit[p]])

            # any non-permissible phased_pivot_var_value?
            node_acceptable = (np.count_nonzero(np.isin(phased_pivot_var_value, permissible_values)) == len(phased_pivot_var_value))

            # where next, chief?
            if node_acceptable:
                # go forward to the next level
                p = p+1
                phase_value[p] = 0
                go_back = False
                total_nodes = total_nodes + 1
            else:
                # this node is not acceptable, so move to the next node, if possible
                phase_value[p] = phase_value[p] + 1

                if phase_value[p] > max_phase_value[p]:
                    # backtrack to the previous node
                    p = p - 1
                    go_back = True
                else:
                    # we are entering a new node, hence no backtracking
                    go_back = False
                    total_nodes = total_nodes + 1

    # Return the set of all identified graphs
    return graphs
