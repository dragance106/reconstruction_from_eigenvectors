The Python code contained here implements two algorithms for reconstruction of a graph from its set of orthogonal eigenvectors. 
Both algorithms are based on spectral decomposition which they transform into linear system with adjacency matrix entries with unknown variables.
In case of integral graphs, for which there exist sets of integer eigenvectors, the algorithm in int_reconstruct.py uses RREF
to perform exact reconstruction by decomposing pivots and free variables into phases that enable fail-fast validity checking
as soon as the values of enough free variables are set to enable computations of corresponding pivots.
In case of graphs with irrational eigenvalues, for which eigenbasis necessarily contains irrational entries,
the algorithm in irr_reconstruct.py selects a numerically stable invertible submatrix made from product of eigenvector entry pairs,
while trying to maximize the number of fixed zero entries, and then perform an exhaustive enumeration of all graphs.

The other files exemplify the use of these algorithms:
- hadamard.py enumerates sets of graphs with eigenbasis given by Hadamard matrices
- coeig-paths.py constructs graphs having the same eigenbasis as paths
- coeig_int_regular.py constructs graphs having the same eigenbasis as set of quartic integral Cayley graphs identified by Marsha Minchenko and Ian Wanless
- coeig_small_graphs.py constructs graphs having the same eigenbasis as given sets of small graphs/trees, while
- coeig_int_trees.py tries to construct graphs having the same eigenbasis as small integral trees identified by Andries Brouwer. 
Interestingly, it appears that no other graph shares the one particular integer orthogonal eigenbasis found by the code,
which still does not mean that these graphs are co-eigenvector unique...

Details of the code and use cases are described in a forthcoming paper "Computational reconstruction of graphs from their orthogonal eigenvectors" 
by Mohammad Ghebleh, Salem Al-Yakoob, Ali Kanso and Dragan Stevanović.
