import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# import mpl_toolkits
# from mpl_toolkits.mplot3d import Axes3D

class GraphX(object):
    # initialization arg: adjacency matrix a (format np.array)
    # Throwing in a __new__ just to do some non-comprehensive idiot-proofing
    #def __new__(cls, adj, *args, **kwargs):
    #    try:
    #        if (adj.shape[0] != adj.shape[1]) or ((len(adj.shape) >= 3) or not np.allclose(adj, adj.T)):
    #            raise ValueError
    #    except ValueError:
    #        print("[Err:GraphX:invalidAdjacencyDimensions] Adjacency matrices must be square and symmetric.")
    #        return super(GraphX, cls).__new__(cls)
    #    return object.__new__(cls)

    def __init__(self, adj, *args, **kwargs):
        self.__adjacencyMatrix = adj  # adjacency matrix, used in init
        self.__nNodes = adj.shape[0]  # number of nodes
        self.__degreeVector = adj.sum(1)  # vector of node degrees
        d = np.zeros((self.__nNodes, self.__nNodes))
        normalizedD = np.zeros((self.__nNodes, self.__nNodes))

        e = 0  # Quick loop to calculate some stuff, incl. nEdges
        for i in range(self.__nNodes):
            d[i][i] = self.__degreeVector[i]
            if self.__degreeVector[i] != 0:
                normalizedD[i][i] = self.__degreeVector[i] ** (-0.5)
            for j in range(i):
                e += adj[i][j]
        self.__nEdges = e  # Number of edges in the graph

        incidenceMatrix = np.zeros((self.__nNodes, self.__nEdges))
        currentEdge = 0
        for j in range(self.__nNodes):
            for i in range(j):
                if self.__adjacencyMatrix[i][j] > 0:
                    incidenceMatrix[i][currentEdge] = self.__adjacencyMatrix[i][j]
                    incidenceMatrix[j][currentEdge] = (-1) * self.__adjacencyMatrix[i][j]
                    currentEdge += 1
        self.__incidenceMatrix = incidenceMatrix  # Graph incidence matrix

        self.__degreeMatrix = d  # diagonal matrix of degrees
        self.__laplacianMatrix = np.subtract(self.__degreeMatrix,
                                             self.__adjacencyMatrix)  # (combinatorial) Laplacian Matrix (LM)
        self.__normalizedLaplacianMatrix = np.matmul(np.matmul(normalizedD, self.__laplacianMatrix),
                                                     normalizedD)  # (normalized) LM
        self.__randomWalkLaplacianMatrix = np.subtract(np.eye(self.__nNodes),
                                                       np.matmul(np.matmul(normalizedD, normalizedD),
                                                                 self.__adjacencyMatrix))  # (random walk) LM

        eigenvalues, eigenvectors = np.linalg.eig(self.__laplacianMatrix)
        idx0 = eigenvalues.argsort()[::1]
        eigenvalues = eigenvalues[idx0]
        eigenvectors = eigenvectors[:, idx0]
        self.__laplacianMatrixEigenvalues = np.around(eigenvalues, decimals=5)  # LM  eigenvalues, ordered increasing
        self.__laplacianMatrixEigenvectors = np.around(eigenvectors, decimals=5)  # (Corresponding) LM eigenvectors

        normalizedEigenvalues, normalizedEigenvectors = np.linalg.eig(self.__normalizedLaplacianMatrix)
        idx1 = normalizedEigenvalues.argsort()[::1]
        normalizedEigenvalues = normalizedEigenvalues[idx1]
        normalizedEigenvectors = normalizedEigenvectors[:, idx1]
        self.__normalizedLaplacianMatrixEigenvalues = np.around(normalizedEigenvalues,
                                                                decimals=5)  # Normalized LM eigenvalues, ordered increasing
        self.__normalizedLaplacianMatrixEigenvectors = np.around(normalizedEigenvectors,
                                                                 decimals=5)  # Corresponding LM eigenvectors

        self.__isConnected = self.__laplacianMatrixEigenvalues[
                                 1] > 1e-5  # (Boolean) if the graph is connected, "best guess"

    def draw(self):

        xData = self.__normalizedLaplacianMatrixEigenvectors[:, 1].T
        yData = self.__normalizedLaplacianMatrixEigenvectors[:, 2].T

        leftLimit = np.amin(self.__normalizedLaplacianMatrixEigenvectors[:, 1].T)
        rightLimit = np.amax(self.__normalizedLaplacianMatrixEigenvectors[:, 1].T)
        topLimit = np.amin(self.__normalizedLaplacianMatrixEigenvectors[:, 2].T)
        bottomLimit = np.amax(self.__normalizedLaplacianMatrixEigenvectors[:, 2].T)

        fig, ax = plt.subplots(figsize=(3, 3), facecolor='lightskyblue', layout='constrained')
        ax.set_xlim(left=leftLimit, right=rightLimit)
        ax.set_ylim(bottom=bottomLimit, top=topLimit)
        ax.set_title("Spectral Embedding of G")

        for i in range(self.__nNodes):

            center = xData[i], yData[i]
            nodeLabel = matplotlib.patches.Circle(center, radius=5e-2)
            ax.text(xData[i], yData[i], str(i+1))
            ax.add_patch(nodeLabel)

            for j in range(i):
                if self.__adjacencyMatrix[i][j] > 0:
                    edge = matplotlib.lines.Line2D([xData[i], xData[j]], [yData[i], yData[j]], figure=fig)
                    ax.add_line(edge)

        plt.show()

    @property
    def adjacencyMatrix(self):
        return self.__adjacencyMatrix

    @property
    def nNodes(self):
        return self.__nNodes

    @property
    def nEdges(self):
        return self.__nEdges

    @property
    def isConnected(self):
        return self.__isConnected

    @property
    def degreeVector(self):
        return self.__degreeVector

    @property
    def degreeMatrix(self):
        return self.__degreeMatrix

    @property
    def laplacianMatrix(self):
        return self.__laplacianMatrix

    @property
    def incidenceMatrix(self):
        return self.__incidenceMatrix

    @property
    def normalizedLaplacianMatrix(self):
        return self.__normalizedLaplacianMatrix

    @property
    def randomWalkLaplacianMatrix(self):
        return self.__randomWalkLaplacianMatrix

    @property
    def laplacianMatrixEigenvalues(self):
        return self.__laplacianMatrixEigenvalues

    @property
    def laplacianMatrixEigenvectors(self):
        return self.__laplacianMatrixEigenvectors

    @property
    def normalizedLaplacianMatrixEigenvalues(self):
        return self.__normalizedLaplacianMatrixEigenvalues

    @property
    def normalizedLaplacianMatrixEigenvectors(self):
        return self.__normalizedLaplacianMatrixEigenvectors
