import networkx
import networkx as nx
import numpy
import numpy as np
import scipy as sp
import ConnectionGraphX as cgx

import ConnectionNetworkX

# Initialize a graph
g = nx.cycle_graph(3)
DIM_CONNECTION = 2

# Test rotation
r = numpy.array([[0, -1], [1, 0]])

# Use CNX to add some rotation
c = ConnectionNetworkX.ConnectionNetworkX(nx.adjacency_matrix(g), DIM_CONNECTION)
c.updateEdgeSignature((0,1), r)
c.updateEdgeSignature((1, 2), r)

# Look at some info.
print(c.connectionLaplacianMatrix.toarray())
vals, _ = sp.sparse.linage.eigs(c.connectionLaplacianMatrix, k=4, which='LM')
print(vals)
# Very weird. These are the "wrong" eigenvalues.

# To see this: construct an identical connection graph using CGX
a = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
h = cgx.ConnectionGraphX(np.array(a), DIM_CONNECTION)
h.setEdgeConnection((0,1), r)
h.setEdgeConnection((1, 2), r)

# Verify that, as numpy arrays, the CLMs are identical
print(h.connectionLaplacianMatrix - c.connectionLaplacianMatrix.toarray())

# But weird, the eigenvalues are completetely different. These are the correct ones.
print(h.connectionLaplacianMatrixEigenvalues)