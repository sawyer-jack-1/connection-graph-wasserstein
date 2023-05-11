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
r = numpy.array([[-1, 0], [0, -1]])

# Use CNX to add some rotation
c = ConnectionNetworkX.ConnectionNetworkX(nx.adjacency_matrix(g), DIM_CONNECTION)
c.updateEdgeSignature((0, 1), r)
c.updateEdgeSignature((1, 2), r)

A = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
mean = A.mean(axis=1)

print(mean)
print(A.shape)

A = A - mean[:, np.newaxis]

print(A)

c.removeEdge((0, 1))
# print(c.connectionLaplacianMatrix)
