import re
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import multiprocessing as mp

from int_reconstruct import int_reconstruct
from irr_reconstruct import irr_reconstruct


def _find_graph_blocks(text: str) -> List[Tuple[int, str]]:
    """
    Find occurrences of Graph< N | ... > and return list of (N, body_text).
    body_text is the text between the '|' after N and the matching '>' (not including the final '>').
    """
    blocks = []
    i = 0
    while True:
        m = re.search(r'Graph\s*<\s*(\d+)\s*\|', text[i:], flags=re.IGNORECASE)
        if not m:
            break
        start = i + m.start()
        n = int(m.group(1))
        # find the position of the '|' we matched
        pipe_pos = i + m.end() - 1
        # now find matching '>' with nesting for <...>
        depth = 1
        j = pipe_pos + 1
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == '<':
                depth += 1
            elif ch == '>':
                depth -= 1
            j += 1
        body = text[pipe_pos+1:j-1].strip()  # between '|' and final '>'
        blocks.append((n, body))
        i = j
    return blocks

def _parse_edge_list(body: str, n: int, one_based=True, dtype=np.int8):
    """
    Parse body like {{a,b}, {c,d}, ...} and return n x n adjacency matrix.
    Uses provided n (from Graph< n | ... >).
    """
    # find all pairs {a,b} - allow spaces/newlines
    pairs = re.findall(r'\{\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\}', body)
    A = np.zeros((n, n), dtype=dtype)
    for a_str, b_str in pairs:
        a = int(a_str) - (1 if one_based else 0)
        b = int(b_str) - (1 if one_based else 0)
        if a < 0 or b < 0:
            raise ValueError("Negative/zero index encountered; check one_based flag.")
        # expand if necessary (rare because n is given)
        if a >= A.shape[0] or b >= A.shape[0]:
            newn = max(a, b) + 1
            B = np.zeros((newn, newn), dtype=dtype)
            B[:A.shape[0], :A.shape[0]] = A
            A = B
        A[a, b] = 1
        A[b, a] = 1
    return A

def _parse_adjacency_list(body: str, n: int, one_based=True, dtype=np.int8):
    """
    Parse body like [ {..}, {..}, ... ] where each inner {} is neighbor list for a vertex.
    Returns n x n adjacency matrix.
    """
    # find the first '[' and matching ']' in the body
    lb = body.find('[')
    rb = body.rfind(']')
    content = body
    if lb != -1 and rb != -1 and rb > lb:
        content = body[lb+1:rb]
    # scan content and extract top-level { ... } chunks in order
    neighbor_blocks = []
    depth = 0
    cur = []
    collecting = False
    for ch in content:
        if ch == '{':
            depth += 1
            if depth == 1:
                cur = []
                collecting = True
                token = ''
            else:
                token += ch
        elif ch == '}':
            if depth == 1 and collecting:
                token = token.strip()
                if token == '':
                    neighbor_blocks.append([])
                else:
                    # split tokens by comma and whitespace
                    nums = re.split(r'[,\s;]+', token)
                    nums = [int(x) for x in nums if x != '']
                    neighbor_blocks.append(nums)
                collecting = False
                token = ''
            else:
                token += ch
            depth = max(0, depth-1)
        else:
            if collecting:
                token += ch
            else:
                # outside inner braces: ignore
                pass

    # Build adjacency matrix; use n provided, but allow expansion if neighbor indices exceed n
    A = np.zeros((n, n), dtype=dtype)
    for i, nbrs in enumerate(neighbor_blocks):
        for v in nbrs:
            vi = int(v) - (1 if one_based else 0)
            if vi < 0:
                raise ValueError("Negative/zero index in adjacency list.")
            if vi >= A.shape[0] or i >= A.shape[0]:
                newn = max(vi, i) + 1
                B = np.zeros((newn, newn), dtype=dtype)
                B[:A.shape[0], :A.shape[0]] = A
                A = B
            A[i, vi] = 1
    # ensure undirected (symmetrize)
    if not np.array_equal(A, A.T):
        A = (A + A.T) > 0
        A = A.astype(dtype)
    return A

def read_magma_graphs_to_numpy_magma(filepath: str, *, one_based: bool = True, dtype=np.int8) -> List[np.ndarray]:
    """
    Read MAGMA file and return list of adjacency matrices for Graph< n | ... > occurrences.
    Supports edge-list style Graph< n | {{...}} > and adjacency-list style Graph< n | [ {...}, ... ] >.
    """
    text = open(filepath, 'r', encoding='utf-8').read()
    blocks = _find_graph_blocks(text)
    mats = []
    for n, body in blocks:
        # classification: if body contains "{{" soon after '|' treat as edge list
        trimmed = body.lstrip()
        if trimmed.startswith('{') and trimmed.lstrip().startswith('{'):
            # edge list representation e.g. {{a,b}, {c,d}, ...}
            A = _parse_edge_list(body, n, one_based=one_based, dtype=dtype)
            mats.append(A)
        elif '[' in body and ']' in body:
            # adjacency-list representation inside [ ... ]
            A = _parse_adjacency_list(body, n, one_based=one_based, dtype=dtype)
            mats.append(A)
        else:
            # fallback: if many {a,b} pairs present treat as edge list
            if re.search(r'\{\s*\d+\s*,\s*\d+\s*\}', body):
                A = _parse_edge_list(body, n, one_based=one_based, dtype=dtype)
                mats.append(A)
            else:
                # unknown block format: skip or warn
                # print(f"Warning: unrecognized Graph body; skipping. body start: {body[:80]!r}")
                continue
    return mats


from int_eigenbasis import integer_orthogonal_eigenbasis


def magma_graph_reader():
    mats1 = read_magma_graphs_to_numpy_magma("coeig_int_regular/QIG1.mgm")
    mats2 = read_magma_graphs_to_numpy_magma("coeig_int_regular/QIG2.mgm")
    mats3 = read_magma_graphs_to_numpy_magma("coeig_int_regular/QIG3.mgm")
    mats4 = read_magma_graphs_to_numpy_magma("coeig_int_regular/QIG4.mgm")

    counter = 0
    for A in mats1 + mats2 + mats3 + mats4:
        counter += 1

        # for int_reconstruct
        x = integer_orthogonal_eigenbasis(A)

        # for irr_reconstruct
        # _, x = np.linalg.eigh(A)

        if x.shape[0]>72:
            continue    # skip graphs that are too large

        yield x, counter


def process_each_graph(params):
    x, counter = params
    n = x.shape[0]
    print(f'processing graph {counter}...')

    # graphs = int_reconstruct(n, x, [0,1], skip_disconnected=True)

    graphs = irr_reconstruct(n, x, [0,1], skip_disconnected=True, skip_regular=False)
    # for larger graphs it may easily end up with 2^more than 35 choices...

    if len(graphs) > 1:
        print(f'len(graphs)={len(graphs)} for counter={counter}...')
        for i, g in enumerate(graphs):
            # export adjacency matrix, eigenvalues and eigenvectors for each graph
            f_data = open(f'coeig_int_regular/count-{counter}-entry-{i}.txt', 'w')
            print(f"adjacency matrix:\n{g['adjacency']}\n", file=f_data)
            print(f"eigenvalues: {g['eigenvalues']}\n", file=f_data)
            print(f"eigenvectors:\n{g['eigenvectors']}\n", file=f_data)
            f_data.close()

            # also save drawings made with networkX - both in kamada_kawai and spring layouts!
            fig = plt.figure()
            pos = nx.kamada_kawai_layout(g['graph'])
            nx.draw(g['graph'], pos=pos, ax=fig.add_subplot(111))
            fig.savefig(f'coeig_int_regular/count-{counter}-entry-{i}-kamada-kawai.png')
            plt.close(fig)

            fig = plt.figure()
            pos = nx.spring_layout(g['graph'])
            nx.draw(g['graph'], pos=pos, ax=fig.add_subplot(111))
            fig.savefig(f'coeig_int_regular/count-{counter}-entry-{i}-spring.png')
            plt.close(fig)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)

    graphs = magma_graph_reader()
    with mp.Pool(max(mp.cpu_count(),1)) as pool:
        list = pool.map(process_each_graph, graphs)

    print('done')
