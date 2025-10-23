import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from irr_reconstruct import irr_reconstruct

for n in range(4, 21):
    A = np.zeros((n,n),dtype=int)
    for i in range(n-1):
        A[i,i+1] = 1
        A[i+1,i] = 1

    _, x = np.linalg.eigh(A)

    graphs = irr_reconstruct(n, x, [0, 1],
                             skip_disconnected=True,
                             skip_regular=False)

    if len(graphs) > 1:
        print(f'len(graphs)={len(graphs)} for order {n}')
        for i, g in enumerate(graphs):
            # export adjacency matrix, eigenvalues and eigenvectors for each graph
            f_data = open(f'coeig_small_paths/n-{n}-entry-{i}.txt', 'w')
            print(f"adjacency matrix:\n{g['adjacency']}\n", file=f_data)
            print(f"eigenvalues: {g['eigenvalues']}\n", file=f_data)
            print(f"eigenvectors:\n{g['eigenvectors']}\n", file=f_data)
            f_data.close()

            # also save drawings made with networkX - both in kamada_kawai and spring layouts!
            fig = plt.figure()
            pos = nx.kamada_kawai_layout(g['graph'])
            nx.draw(g['graph'], pos=pos, ax=fig.add_subplot(111))
            fig.savefig(f'coeig_small_paths/n-{n}-entry-{i}-kamada-kawai.png')
            plt.close(fig)

            fig = plt.figure()
            pos = nx.spring_layout(g['graph'])
            nx.draw(g['graph'], pos=pos, ax=fig.add_subplot(111))
            fig.savefig(f'coeig_small_paths/n-{n}-entry-{i}-spring.png')
            plt.close(fig)
