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

        # Force into lil_matrix since scipy wants to convert to csr after matmul.
        self.connectionLaplacianMatrix = scipy.sparse.lil_matrix(
            self.connectionIncidenceMatrix * self.connectionIncidenceMatrix.transpose())

    def updateEdgeSignature(self, edge, rotation):
        fromNode = edge[0]
        toNode = edge[1]
        edgeIndex = list(self.edges()).index(edge)
        d = self.dimConnection
        O_sparse = scipy.sparse.lil_matrix(rotation)

        self.connectionIncidenceMatrix[(toNode * d):((toNode + 1) * d), (edgeIndex * d):((edgeIndex + 1) * d)] = (
                                                                                                                     -1) * O_sparse
        self.connectionLaplacianMatrix[(fromNode * d):((fromNode + 1) * d), (toNode * d):((toNode + 1) * d)] = (
                                                                                                                   -1) * O_sparse
        self.connectionLaplacianMatrix[(toNode * d):((toNode + 1) * d), (fromNode * d):((fromNode + 1) * d)] = (
                                                                                                                   -1) * O_sparse.T

    def removeEdge(self, edge):

        fromNode = edge[0]
        toNode = edge[1]
        edgeIndex = list(self.edges()).index(edge)
        d = self.dimConnection
        z = scipy.sparse.lil_matrix((d, d))

        degreeFrom = self.connectionLaplacianMatrix[fromNode * d, fromNode * d]
        degreeTo = self.connectionLaplacianMatrix[toNode * d, toNode * d]

        self.remove_edge(fromNode, toNode)
        self.nEdges = self.number_of_edges()

        self.connectionIncidenceMatrix[(fromNode * d):((fromNode + 1) * d), (edgeIndex * d):((edgeIndex + 1)*d)] = z
        self.connectionIncidenceMatrix[(toNode * d):((toNode + 1)*d), (edgeIndex * d):((edgeIndex + 1)*d)] = z
        self.connectionLaplacianMatrix[(fromNode * d):((fromNode + 1) * d), (fromNode * d):((fromNode + 1) * d)] = (degreeFrom - 1) * scipy.sparse.lil_matrix(numpy.eye(d))
        self.connectionLaplacianMatrix[(toNode * d):((toNode + 1) * d), (toNode * d):((toNode + 1) * d)] = (degreeTo - 1) * scipy.sparse.lil_matrix(numpy.eye(d))
        self.connectionLaplacianMatrix[(toNode * d):((toNode + 1)*d), (fromNode * d):((fromNode + 1)*d)] = z
        self.connectionLaplacianMatrix[(fromNode * d):((fromNode + 1)*d), (toNode * d):((toNode + 1)*d)] = z

    def printConnectionLaplacianEigenvalues(self, n=10, w="SM", showBalanced=True):
        vals = scipy.sparse.linalg.eigsh(self.connectionLaplacianMatrix, which=w, k=n, return_eigenvectors=False)
        tolerance = 1e-8

        print(vals)

        if showBalanced:
            if abs(vals[n-1]) < tolerance:
                print("MOST LIKELY CONSISTENT: |lambda_min| < 1e-8. ")
            else:
                print("MOST LIKELY INCONSISTENT: |lambda_min| >= 1e-8. ")


