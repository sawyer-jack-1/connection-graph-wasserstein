from GraphX import GraphX
import numpy as np

class ConnectionGraphX(GraphX):

    def __init__(self, adj, dim):
        super().__init__(adj)
        self.__dimConnection = dim

        connectionDegreeMatrix_ = np.zeros((self.nNodes*dim, self.nNodes*dim))
        for i in range(self.nNodes):
            connectionDegreeMatrix_[(i*dim):((i+1)*dim), (i*dim):((i+1)*dim)] = self.degreeVector[i] * np.eye(dim)
        self.__connectionDegreeMatrix = connectionDegreeMatrix_

        # Initializing the rest to optimizing the addition of actual rotations later
        connection_ = np.zeros((self.nNodes, self.nNodes, dim, dim))
        connectionIncidenceMatrix_ = np.zeros((self.nNodes*dim, self.nEdges*dim))
        connectionAdjacencyMatrix_ = np.zeros((self.nNodes*dim, self.nNodes*dim))
        currentEdge = 0
        for j in range(self.nNodes):
            connection_[j, j, :, :] = np.eye(dim)
            for i in range(j):
                if self.adjacencyMatrix[i][j] > 0:
                    connectionIncidenceMatrix_[(i * dim):((i+1) * dim), (currentEdge * dim):((currentEdge+1) * dim)] = self.adjacencyMatrix[i][j] * np.eye(dim)
                    connectionIncidenceMatrix_[(j * dim):((j+1) * dim), (currentEdge * dim):((currentEdge+1) * dim)] = (-1) * self.adjacencyMatrix[i][j] * np.eye(dim)
                    currentEdge += 1

                    connectionAdjacencyMatrix_[(i*dim):((i+1)*dim), (j*dim):((j+1)*dim)] = np.eye(dim)
                    connectionAdjacencyMatrix_[(j*dim):((j+1)*dim), (i*dim):((i+1)*dim)] = np.eye(dim)

                    connection_[i, j, :, :] = np.eye(dim)
                    connection_[j, i, :, :] = np.eye(dim)

        self.__connectionIncidenceMatrix = connectionIncidenceMatrix_
        self.__connectionAdjacencyMatrix = connectionAdjacencyMatrix_
        self.__connectionLaplacianMatrix = np.subtract(self.__connectionDegreeMatrix, self.__connectionAdjacencyMatrix)
        self.__connection = connection_

        connectionLaplacianMatrixEigenvalues, connectionLaplacianMatrixEigenvectors = np.linalg.eig(self.__connectionLaplacianMatrix)
        idx0 = connectionLaplacianMatrixEigenvalues.argsort()[::1]
        connectionLaplacianMatrixEigenvalues = connectionLaplacianMatrixEigenvalues[idx0]
        connectionLaplacianMatrixEigenvectors = connectionLaplacianMatrixEigenvectors[:, idx0]
        self.__connectionLaplacianMatrixEigenvalues = np.around(connectionLaplacianMatrixEigenvalues, decimals=5)  # LM  eigenvalues, ordered increasing
        self.__connectionLaplacianMatrixEigenvectors = np.around(connectionLaplacianMatrixEigenvectors, decimals=5)  # (Corresponding) LM eigenvectors

    def setEdgeConnection(self, nodeTuple, connectionMatrix):
        # Convention: connectionMatrix is O(d)

        fromNode = nodeTuple[0]
        toNode = nodeTuple[1]
        dim = self.__dimConnection

        self.__connection[fromNode, toNode, :, :] = connectionMatrix
        self.__connection[toNode, fromNode, :, :] = connectionMatrix.T

        self.__connectionAdjacencyMatrix[(fromNode*dim):((fromNode+1)*dim), (toNode*dim):((toNode+1)*dim)] = connectionMatrix
        self.__connectionAdjacencyMatrix[(toNode*dim):((toNode+1)*dim), (fromNode*dim):((fromNode+1)*dim)] = connectionMatrix.T
        self.__connectionLaplacianMatrix = np.subtract(self.__connectionDegreeMatrix, self.__connectionAdjacencyMatrix)

    @property
    def dimConnection(self):
        return self.__dimConnection

    @property
    def connection(self):
        return self.__connection

    @property
    def connectionIncidenceMatrix(self):
        return self.__connectionIncidenceMatrix

    @property
    def connectionAdjacencyMatrix(self):
        return self.__connectionAdjacencyMatrix

    @property
    def connectionDegreeMatrix(self):
        return self.__connectionDegreeMatrix

    @property
    def connectionLaplacianMatrix(self):
        return self.__connectionLaplacianMatrix

    @property
    def connectionLaplacianMatrixEigenvalues(self):
        return self.__connectionLaplacianMatrixEigenvalues

    @property
    def connectionLaplacianMatrixEigenvectors(self):
        return self.__connectionLaplacianMatrixEigenvectors
