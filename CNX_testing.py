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
W = 3
H = 3
c = ConnectionNetworkX.cnxFromPixelGrid(W, H, DIM_CONNECTION)

print(c.gridEmbedding)