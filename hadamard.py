import numpy as np
import multiprocessing as mp
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import time

from int_reconstruct import int_reconstruct
from irr_reconstruct_old import irr_reconstruct


# decode Hadamard matrix equivalence class representative from a line
def convert_line(n, line):
    """
    According to Brendan McKay:
    In the following files there is one Hadamard matrix per line.
    The line consists of n hexadecimal numbers, one per row of the matrix.
    To obtain the actual rows, write the numbers in binary and CHANGE all zeros to -1.
    """
    row_strings = line.split()

    # eigenvector matrix, initialized as all-one
    x = np.ones((n, n), dtype=np.intp)

    for i in range(n):
        # convert row_string from hex to int and then
        # to binary of width n filled with 0s on the left
        row_binary = f'{int(row_strings[i], 16):0{n}b}'

        # only translate binary 0s to -1s in X
        for j in range(n):
            if row_binary[j] == '0':
                x[i, j] = -1

    return x


# for multiprocessing to work, one must have a generator of new parameters
def matrix_reader(n, permissible_values):

    # open the file with equivalent classes of Hadamard matrices
    with open('had_matrices/hadamard' + str(n) + '.txt', 'r') as file:
        counter = 0

        # for each line in this file
        for line in file:

            # translate line into matrix with equivalence class representative
            # and return it as the next eigenvector matrix to process
            x = convert_line(n, line)
            counter += 1

            for column in range(n):
                yield n, x, permissible_values, counter, column


# Separately process each Hadamard matrix equivalence class representative
def process_all_choices(params):
    n, x, permissible_values, counter, column = params
    print(f'processing matrix {counter}, column {column}...')

    # make the specified column all-one by negating corresponding rows
    x_copy = x.copy()
    c = x_copy[:, column]
    x_copy *= c[:, None]

    graphs = irr_reconstruct(n, x_copy, permissible_values)

    return graphs


if __name__=="__main__":
    tic = time.time()
    np.set_printoptions(threshold=np.inf)

    # number of vertices is one of: 4, 8, 12, 16, 20, 24, 28, 32
    n = 16

    # permissible values for adjacency matrix entries are
    # either [0, 1] for simple graphs or [-1, 0, 1] for signed graphs
    permissible_values = [0, 1]

    # set up a generator that will read Hadamard matrix equivalence class representatives
    matrices = matrix_reader(n, permissible_values)

    # process each matrix with the next free processor core
    with mp.Pool(max(mp.cpu_count(), 1)) as pool:

        # collect all graphs found for each eigenvector matrix
        list_list_graphs = pool.map(process_all_choices, matrices)

    # list_list_graphs is a list of lists in which every element is a dictionary
    # with fields 'graph' (networkx graph), 'eigenvectors' (Hadamard matrix), 'eigenvalues' and 'adjacency' (matrix)

    # get a final list of unique graphs
    final_graphs = []
    for l in list_list_graphs:
        for g in l:
            same_old = False
            for h in final_graphs:
                if nx.vf2pp_is_isomorphic(g['graph'], h['graph']):
                    same_old = True
                    break

            if not same_old:
                final_graphs.append(g)

    # report the final graphs
    toc = time.time()
    print(f'time elapsed: {toc-tic} seconds')

    f_all_data = open('had_graphs/hadamard' + str(n) + '_all_data.txt', 'w')
    f_graph6_codes = open('had_graphs/hadamard' + str(n) + '.g6', 'w')

    matplotlib.use("Agg")
    counter = 1

    # how to use spectral layout in nx.draw?

    for g in final_graphs:
        # export graph6 code, adjacency matrix, eigenvalues, and eigenvectors
        f_all_data.write(nx.to_graph6_bytes(g['graph'], header=False).decode('utf-8'))
        print(f"adjacency matrix:\n{g['adjacency']}", file=f_all_data)
        print(f"eigenvalues: {g['eigenvalues']}", file=f_all_data)
        print(f"eigenvectors:\n{g['eigenvectors']}", file=f_all_data)

        # separately export graph6 codes only - use write since to_graph6_bytes already add \n at the end...
        f_graph6_codes.write(nx.to_graph6_bytes(g['graph'], header=False).decode('utf-8'))

        # also save drawings made with networkX - both in kamada_kawai and spring layouts!
        fig = plt.figure()
        pos = nx.kamada_kawai_layout(g['graph'])
        nx.draw(g['graph'], pos=pos, ax=fig.add_subplot(111))
        fig.savefig('had_graphs/hadamard' + str(n) + '-' + str(counter) + '-kamada-kawai.png')
        plt.close(fig)

        fig = plt.figure()
        pos = nx.spring_layout(g['graph'])
        nx.draw(g['graph'], pos=pos, ax=fig.add_subplot(111))
        fig.savefig('had_graphs/hadamard' + str(n) + '-' + str(counter) + '-spring.png')
        plt.close(fig)

        counter += 1

    f_all_data.close()
    f_graph6_codes.close()
