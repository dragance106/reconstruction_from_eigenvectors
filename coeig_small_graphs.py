import numpy as np
import multiprocessing as mp
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import time


from irr_reconstruct import irr_reconstruct


# for multiprocessing to work, one must have a generator of new parameters
def graph_reader(n, permissible_values):

    # open the file with all simple graphs of given size
    with open('coeig_small_trees/trees' + str(n) + '.g6', 'r') as file:
        counter = 0

        # for each line in this file
        for line in file:

            # translate line with g6 code into graph
            s = line.strip()
            if not s or s.startswith('#'):
                continue

            counter += 1
            G = nx.from_graph6_bytes(s.encode('ascii'))

            # get its adjacency matrix and adjacency eigenvectors
            A = nx.to_numpy_array(G, dtype=int)
            _, x = np.linalg.eigh(A)

            # use this eigenvector matrix to reconstruct all co-eigenvector graphs
            yield n, x, permissible_values, counter


# Separately process each graph
def process_each_graph(params):
    n, x, permissible_values, counter = params

    if counter % 10000 == 0:
        print(f'processing tree {counter}...')

    graphs = irr_reconstruct(n, x, permissible_values,
                             skip_disconnected=True,
                             skip_regular=True)

    # if more than a single graph is found,
    # then write the necessary data externally
    if len(graphs) > 1:
        print(f'len(graphs)={len(graphs)} for graph {counter}')
        for i, g in enumerate(graphs):
            # export adjacency matrix, eigenvalues and eigenvectors for each graph
            f_data = open(f'coeig_small_trees/n-{n}-graph-{counter}-entry-{i}.txt', 'w')
            print(f"adjacency matrix:\n{g['adjacency']}\n", file=f_data)
            print(f"eigenvalues: {g['eigenvalues']}\n", file=f_data)
            print(f"eigenvectors:\n{g['eigenvectors']}\n", file=f_data)
            f_data.close()

            # also save drawings made with networkX - both in kamada_kawai and spring layouts!
            fig = plt.figure()
            pos = nx.kamada_kawai_layout(g['graph'])
            nx.draw(g['graph'], pos=pos, ax=fig.add_subplot(111))
            fig.savefig(f'coeig_small_trees/n-{n}-graph-{counter}-entry-{i}-kamada-kawai.png')
            plt.close(fig)

            fig = plt.figure()
            pos = nx.spring_layout(g['graph'])
            nx.draw(g['graph'], pos=pos, ax=fig.add_subplot(111))
            fig.savefig(f'coeig_small_trees/n-{n}-graph-{counter}-entry-{i}-spring.png')
            plt.close(fig)


# We'll stream batches from the generator and use imap_unordered on batches.
from itertools import islice


def batched(iterable, batch_size):
    """Lazily yield lists of up to batch_size items from iterable."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch


def process_batch(batch):
    """
    Process a list (batch) of graphs by calling process_each_graph on each.
    All side-effects (writing to disk) are done inside process_each_graph.
    We swallow exceptions per-item so one bad graph doesn't kill the whole run.
    """
    for g in batch:
        try:
            process_each_graph(g)
        except Exception as e:
            # simple logging to stderr; continue with other items
            import sys, traceback
            print(f'Error processing graph in batch: {e}', file=sys.stderr)
            traceback.print_exc()
    return None


if __name__=="__main__":
    tic = time.time()
    matplotlib.use("Agg")

    # number of vertices is one of: 8, 9, 10
    n = 15

    # permissible values for adjacency matrix entries are
    # either [0, 1] for simple graphs or [-1, 0, 1] for signed graphs
    permissible_values = [0, 1]

    # set up a generator that will read simple graphs
    graphs = graph_reader(n, permissible_values)

    # Process graphs in batches to avoid high task-submission overhead.
    num_workers = max(mp.cpu_count(), 1)

    # Try to estimate total number of tasks by counting lines in the g6 file used by graph_reader.
    # If that fails, fall back to a conservative default batch size.
    try:
        g6path = 'coeig_small_trees/trees' + str(n) + '.g6'
        with open(g6path, 'r') as _f:
            total_tasks = sum(1 for _ in _f)
    except Exception:
        total_tasks = None

    target_batches_per_worker = 200
    if total_tasks:
        batch_size = max(1, int(total_tasks / (num_workers * target_batches_per_worker)))
    else:
        batch_size = 10000

    print(f'Using {num_workers} workers, batch_size={batch_size} (estimated total_tasks={total_tasks})')

    # stream batches lazily into the pool using imap_unordered; do not collect results (they are written to disk)
    with mp.Pool(processes=num_workers) as pool:
        for _ in pool.imap_unordered(process_batch, batched(graphs, batch_size), chunksize=1):
            # no-op: results are handled inside process_each_graph; loop keeps the main process alive
            pass

    toc = time.time()
    print(f'time elapsed: {toc-tic} seconds')
