import numpy as np
from typing import Sequence, Tuple, Optional
import multiprocessing as mp
import matplotlib.pyplot as plt
import networkx as nx


from int_reconstruct import int_reconstruct
from irr_reconstruct import irr_reconstruct


def levelseq_to_adj(levels,
                    dtype=np.int8,
                    return_parents: bool = False
                   ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert a level sequence of a tree into its adjacency matrix (NumPy).

    Accepts either:
      - a sequence/array of integers, e.g. [0,1,2,2,1,2], or
      - a compact string of digits, e.g. "012212" (each character is a single decimal digit).

    Parameters
    ----------
    levels : sequence of int or str
        Level (distance-from-root) recorded in DFS first-visit order.
        If a str is provided, it must consist only of decimal digits '0'..'9'.
    dtype : numpy dtype, optional
        dtype for returned adjacency matrix (default np.int8).
    return_parents : bool, optional
        If True, also return the parent index array of length n where parent[0] == -1.

    Returns
    -------
    A : ndarray, shape (n, n)
        Symmetric adjacency matrix (0/1) of the tree.
    parents : ndarray, shape (n,) or None
        Parent index for each vertex (parent[0] == -1). Returned only if return_parents=True.

    Raises
    ------
    ValueError
        If the level sequence is invalid or if a string contains non-digit characters.
    """
    # Parse compact digit string if provided
    if isinstance(levels, (str, bytes)):
        s = levels.decode() if isinstance(levels, bytes) else levels
        if not s:
            raise ValueError("Empty string provided as level sequence.")
        if not s.isdigit():
            raise ValueError("String format must be compact digits only (e.g. '012212').")
        # convert each character to int (single-digit levels only)
        levels_arr = np.fromiter((ord(ch) - 48 for ch in s), dtype=int)  # ord('0')==48
    else:
        # Accept any sequence/array-like of integers
        levels_arr = np.asarray(levels, dtype=int).ravel()

    n = levels_arr.size
    if n == 0:
        A = np.zeros((0, 0), dtype=dtype)
        return (A, np.empty(0, dtype=int)) if return_parents else (A, None)

    if levels_arr[0] != 0:
        raise ValueError("level sequence must start with 0 (root)")

    # parent array; parent[0] = -1 (root)
    parents = -np.ones(n, dtype=int)

    # last_seen[l] = index of the last vertex (so far) encountered at level l
    last_seen = [-1]
    last_seen[0] = 0

    for i in range(1, n):
        l = int(levels_arr[i])
        if l < 0:
            raise ValueError(f"invalid level {l} at index {i}: level must be nonnegative")
        if l >= len(last_seen):
            last_seen.extend([-1] * (l + 1 - len(last_seen)))

        if l == 0:
            raise ValueError(f"invalid level 0 at index {i}: only root (index 0) may have level 0")

        parent_idx = last_seen[l - 1]
        if parent_idx == -1:
            raise ValueError(
                f"invalid level sequence at index {i}: no previous vertex with level {l-1} found"
            )

        parents[i] = parent_idx
        last_seen[l] = i

    # Build adjacency matrix
    A = np.zeros((n, n), dtype=dtype)
    for i in range(1, n):
        p = parents[i]
        A[i, p] = 1
        A[p, i] = 1

    if return_parents:
        return A, parents
    else:
        return A


small_int_trees = ['01111',
                   '012211',
                   '0121212',
                   '0111111111',
                   '01222222111111',
                   '01111111111111111',
                   '01212121212121212',
                   '01222222212121212',
                   '0122222121212121212',
                   '0123333233332333322121212',
                   '01111111111111111111111111',
                   '01222212222122221222212222',
                   '01222222222222111111111111',
                   '0121212121212121212121212121212',
                   '0123331232323123232312222122221',
                   '0123331233312333122221222212222',
                   '01222222222222212121212121212121212',
                   '0111111111111111111111111111111111111',
                   '0122222222222121212121212121212121212',
                   '0123232312323231232323123232312323231',
                   '0123331233312323231232323123232312222',
                   '012222222222222222222211111111111111111111',
                   '0122222222222222122221222212222122221222212222',
                   '0121212121212121212121212121212121212121212121212',
                   '01111111111111111111111111111111111111111111111111',
                   '01222212222122221222212222122221222212222122221111',
                   '0123333333332333333333233333333322222221212121212121212',
                   '01111222212222122221222212222122221222212222123331232323',
                   '01222222222222222222222121212121212121212121212121212121212',
                   '01122221222212222122221233312333123232323232323232323232323',
                   '01112222122221222212323231232323123232323232323232323232323',
                   '0122222222222222222221212121212121212121212121212121212121212',
                   '0122221222212222122221222212222122221222212222122221222212222',
                   '01222222222222222222222222222222111111111111111111111111111111',
                   '01222222222212222122221222212222122221222212222122221222212222',
                   '01111222212222122221222212222122221232323123232312323231232323',
                   '01112222122221222212222122221222212222123331233312323231232323',
                   '01122221222212222122221222212222122221222212333123331233312333',
                   '01111111111111111111111111111111111111111111111111111111111111111',
                   '01122221222212333123232312323231232323123232323232323232323232323',
                   '01222212222122221233312333123331232323123232323232323232323232323',
                   '0111222212222122221222212222122221222212333123331232323232323232323',
                   '0111122221222212222122221222212222123232312323231232323232323232323',
                   '01112222122221222212222122221233312323231232323123232312323231232323',
                   '01122221222212222122221222212222123331233312333123232312323231232323',
                   '01222212222122221222212222122221222212333123331233312333123331232323',
                   '01212121212121212121212121212121212121212121212121212121212121212121212',
                   '01222222222122222222212222222221222222222122222222212222222221222222222',
                   '01111111222222222122222222212323232323232323123232323232323231233333333',
                   '01123232312323231232323123232312323231232323123232323232323232323232323',
                   '01222212333123331232323123232312323231232323123232323232323232323232323',
                   '0111222212222122221222212222123331232323123232312323231232323232323232323',
                   '0112222122221222212222122221222212333123331233312323231232323232323232323',
                   '01112222122221222212323231232323123232312323231232323123232312323231232323',
                   '01122221222212222122221233312333123232312323231232323123232312323231232323',
                   '01222212222122221222212222123331233312333123331232323123232312323231232323',
                   '011111111111111111111111111222222222222222222122222222222222222222222222222222',
                   '0111222212222122221232323123232312323231232323123232312323231232323232323232323',
                   '0112222122221222212222123331233312323231232323123232312323231232323232323232323',
                   '0122221222212222122221222212333123331233312333123232312323231232323232323232323',
                   '01122221222212333123232312323231232323123232312323231232323123232312323231232323',
                   '01222212222122221233312333123331232323123232312323231232323123232312323231232323',
                   '012333333331233333333122222222212222222221222222222122222222212222222221222222222',
                   '0111111111111111111111111111111111111111111111111111111111111111111111111111111111',
                   '0122221222212222123331233312333123232312323231232323123232312323231232323232323232323',
                   '0112222122221233312323231232323123232312323231232323123232312323231232323232323232323',
                   '01222222222222222222222222222222222222222222111111111111111111111111111111111111111111',
                   '01234343423434342343434234343423434342343434234343423434342343434234343423434342343434',
                   '01222212333123331232323123232312323231232323123232312323231232323123232312323231232323',
                   '011111122222222212323232323232323123232323232323231232323232323232312333333331233333333',
                   '01222222222222222222222222222222212121212121212121212121212121212121212121212121212121212',
                   '01111111222212222122221222212222122221222212222122221222212222122221222212222122221232323',
                   '0122222222222222222222222222222121212121212121212121212121212121212121212121212121212121212',
                   '0111112222222221222222222122222222212333323333233332333323333123333333312333333331233333333',
                   '0112323231232323123232312323231232323123232312323231232323123232312323231232323232323232323',
                   '0122221233312333123232312323231232323123232312323231232323123232312323231232323232323232323',
                   '0122222222222222222222221222212222122221222212222122221222212222122221222212222122221222212222',
                   '01111112222122221222212222122221222212222122221222212222122221222212222122221233312323231232323',
                   '01111122221222212222122221222212222122221222212222122221222212222122221222212222123331233312333',
                   '0121212121212121212121212121212121212121212121212121212121212121212121212121212121212121212121212',
                   '01212121222222222122222222212222222221222222222122343434343434341223434343434343412234343434343434']


from int_eigenbasis import integer_orthogonal_eigenbasis


def tree_reader():
    for t in small_int_trees:
        A = levelseq_to_adj(t)
        n = A.shape[0]
        x = integer_orthogonal_eigenbasis(A)

        if x.dtype==object:
            print(f'eigenvector entries too big for level sequence {t}:\n{x}')

        norms = np.array([np.dot(x[0:n, i], x[0:n, i]) for i in range(n)], dtype=np.intp)
        orthogonal = x.T @ x
        if not np.allclose(orthogonal, np.diag(norms)):
            print(f'transposed eigenvectors:\n{x.T}')
            print(f'eigenvectors:\n{x}')
            print(f'eigenvector entries not orthogonal:\n{orthogonal}')
            print(f'eigenvector norms:\n{norms}')

        yield n, x, t


def process_each_tree(params):
    n, x, t = params
    print(f'processing tree with level sequence {t}')

    # graphs = int_reconstruct(n, x, [0,1], skip_disconnected=True, skip_regular=False)

    graphs = irr_reconstruct(n, x, [0,1], skip_disconnected=False, skip_regular=False)
    # for larger trees it easily goes into 2^more than 32 choices...

    if len(graphs) > 1:
        print(f'len(graphs)={len(graphs)} for level sequence {t}')
        for i, g in enumerate(graphs):
            # export adjacency matrix, eigenvalues and eigenvectors for each graph
            f_data = open(f'coeig_int_trees/t-{t}-entry-{i}.txt', 'w')
            print(f"adjacency matrix:\n{g['adjacency']}\n", file=f_data)
            print(f"eigenvalues: {g['eigenvalues']}\n", file=f_data)
            print(f"eigenvectors:\n{g['eigenvectors']}\n", file=f_data)
            f_data.close()

            # also save drawings made with networkX - both in kamada_kawai and spring layouts!
            fig = plt.figure()
            pos = nx.kamada_kawai_layout(g['graph'])
            nx.draw(g['graph'], pos=pos, ax=fig.add_subplot(111))
            fig.savefig(f'coeig_int_trees/t-{t}-entry-{i}-kamada-kawai.png')
            plt.close(fig)

            fig = plt.figure()
            pos = nx.spring_layout(g['graph'])
            nx.draw(g['graph'], pos=pos, ax=fig.add_subplot(111))
            fig.savefig(f'coeig_int_trees/t-{t}-entry-{i}-spring.png')
            plt.close(fig)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    trees = tree_reader()
    with mp.Pool(max(mp.cpu_count(),1)) as pool:
        list = pool.map(process_each_tree, trees)

    print('done')
