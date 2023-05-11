import networkx as nx
import scipy
import numpy as np
import random
from tqdm import tqdm

import puppets_data
from pyLDLE2 import datasets
from pyLDLE2 import buml_
from pyLDLE2 import util_


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

        self.imageData = None
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

        # self.remove_edge(fromNode, toNode)
        # self.nEdges = self.number_of_edges()

        self.connectionIncidenceMatrix[(fromNode * d):((fromNode + 1) * d), (edgeIndex * d):((edgeIndex + 1) * d)] = z
        self.connectionIncidenceMatrix[(toNode * d):((toNode + 1) * d), (edgeIndex * d):((edgeIndex + 1) * d)] = z
        self.connectionLaplacianMatrix[(fromNode * d):((fromNode + 1) * d), (fromNode * d):((fromNode + 1) * d)] = (
                                                                                                                               degreeFrom - 1) * scipy.sparse.lil_matrix(
            np.eye(d))
        self.connectionLaplacianMatrix[(toNode * d):((toNode + 1) * d), (toNode * d):((toNode + 1) * d)] = (
                                                                                                                       degreeTo - 1) * scipy.sparse.lil_matrix(
            np.eye(d))
        self.connectionLaplacianMatrix[(toNode * d):((toNode + 1) * d), (fromNode * d):((fromNode + 1) * d)] = z
        self.connectionLaplacianMatrix[(fromNode * d):((fromNode + 1) * d), (toNode * d):((toNode + 1) * d)] = z

    def printConnectionLaplacianEigenvalues(self, n=10, w="SM", showBalanced=True):
        vals = scipy.sparse.linalg.eigsh(self.connectionLaplacianMatrix, which=w, k=n, return_eigenvectors=False)
        tolerance = 1e-8

        print(vals)

        if showBalanced:
            if abs(vals[n - 1]) < tolerance:
                print("MOST LIKELY CONSISTENT: |lambda_min| < 1e-8. ")
            else:
                print("MOST LIKELY INCONSISTENT: |lambda_min| >= 1e-8. ")


def cnxFromImageDirectory(filePath, intrinsicDimension, k=None, nImages=None, save_dir_root=None):
    Y, labelsMat, _ = puppets_data.puppets_data(filePath)

    if nImages is None:
        X = Y
        nImages = X.shape[0]
    else:
        X = np.array(Y.copy())
        totalImages = X.shape[0]
        sampleIndices = random.sample(range(totalImages), nImages)
        X = [X[i] for i in sampleIndices]
        X = np.array(X)

    if k is None:
        k = X.shape[0] // 50

    buml_obj = buml_.BUML(local_opts={'algo': 'LPCA', 'k': k},
                          intermed_opts={'eta_max': 1},
                          vis_opts={'c': labelsMat[:, 0], 'save_dir': save_dir_root},
                          verbose=True, debug=True, exit_at='local_views')

    buml_obj.fit(X=X)

    cnxAdjacency = buml_obj.LocalViews.U + buml_obj.LocalViews.U.T - scipy.sparse.identity(nImages,
                                                                                                   format='lil')
    cnx = ConnectionNetworkX(cnxAdjacency, intrinsicDimension)

    cnx.imageData = X

    nRemoteEdges = 0
    totalEdgesBeforeRemoval = cnx.nEdges
    for i in tqdm(range(nImages)):
        n_i = nx.neighbors(cnx, i)
        for j in [j for j in n_i if j > i]:

            n_ij = buml_obj.LocalViews.U[i,:].multiply(buml_obj.LocalViews.U[j,:]).nonzero()[1]

            if len(n_ij) >= intrinsicDimension:
                X_Uij_i = buml_obj.LocalViews.local_param_post.eval_({'view_index': i, 'data_mask': n_ij})
                X_Uij_j = buml_obj.LocalViews.local_param_post.eval_({'view_index': j, 'data_mask': n_ij})

                Tij, _ = scipy.linalg.orthogonal_procrustes(X_Uij_i, X_Uij_j)

                cnx.updateEdgeSignature((i,j), Tij)

            else:
                cnx.removeEdge((i,j))
                nRemoteEdges += 1

    print('Proportion of edges which were removed due to remoteness: ', nRemoteEdges / totalEdgesBeforeRemoval)
    cnx.printConnectionLaplacianEigenvalues()

    return cnx

def cnxFromPixelGrid(width, height, intrinsicDimension):

    g = nx.grid_2d_graph(width, height)

    cnx = ConnectionNetworkX(nx.adjacency_matrix(g), intrinsicDimension)

    return cnx