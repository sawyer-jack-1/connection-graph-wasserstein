import networkx as nx
import scipy
import numpy

class ConnectionNetworkX(nx.DiGraph):

    def __init__(self, a):
        super().__init__(a)
