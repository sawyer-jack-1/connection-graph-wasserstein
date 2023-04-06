import networkx as nx
import scipy
import numpy

class ConnectionNetworkX(nx.Graph):

    def __init__(self, a, dim):
        super().__init__(a)
        self.dimConnection = dim
        self.nNodes = self.number_of_nodes()
        self.nEdges = self.number_of_edges()
        self.connectionIncidenceMatrix = scipy.sparse.lil_matrix((self.nNodes * self.dimConnection, self.nEdges *
                                                                  self.dimConnection), dtype=float)
        self.connectionLaplacianMatrix = scipy.sparse.lil_matrix((self.nNodes * self.dimConnection, self.nNodes *
                                                                  self.dimConnection), dtype=float)

        self.initializeConenctionLaplacian()
    def initializeConenctionLaplacian(self):

        for edgeIndex, e in zip(range(self.nEdges), self.edges()):
            fromNode = e[0]
            toNode = e[1]

            colIndexRange = range(edgeIndex * self.dimConnection, (edgeIndex + 1) * self.dimConnection)
            rowIndexRangeFromNode = range(fromNode * self.dimConnection, (fromNode + 1) * self.dimConnection)
            rowIndexRangeToNode = range(toNode * self.dimConnection, (toNode + 1) * self.dimConnection)

            self.connectionIncidenceMatrix[rowIndexRangeFromNode, colIndexRange] = 1
            self.connectionIncidenceMatrix[rowIndexRangeToNode, colIndexRange] = -1

        self.connectionLaplacianMatrix = self.connectionIncidenceMatrix * self.connectionIncidenceMatrix.transpose()

    def updateEdgeSignature(self, edge, O):

        toNode = edge[1]
        edgeIndex = list(self.edges()).index(edge)


        for i in range(self.dimConnection):
            for j in range(self.dimConnection):
                self.connectionIncidenceMatrix[toNode * self.dimConnection + i, edgeIndex * self.dimConnection + j] = (-1) * O[j][i]

        self.connectionLaplacianMatrix = self.connectionIncidenceMatrix * self.connectionIncidenceMatrix.transpose()
