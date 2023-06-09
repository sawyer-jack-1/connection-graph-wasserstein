import math

import networkx as nx
import scipy
import numpy as np
import random
from tqdm import tqdm
import torch

from scipy.linalg import svdvals

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
        self.width = None
        self.height = None

        self.gridEmbedding = None # To be implemented when initializing as a grid graph.

    def initializeConenctionLaplacian(self):
        for edgeIndex, e in zip(range(self.nEdges), self.edges()):

            fromNode = e[0]
            toNode = e[1]
            w = nx.adjacency_matrix(self)[fromNode, toNode]

            colIndexRange = range(edgeIndex * self.dimConnection, (edgeIndex + 1) * self.dimConnection)
            rowIndexRangeFromNode = range(fromNode * self.dimConnection, (fromNode + 1) * self.dimConnection)
            rowIndexRangeToNode = range(toNode * self.dimConnection, (toNode + 1) * self.dimConnection)

            self.connectionIncidenceMatrix[rowIndexRangeFromNode, colIndexRange] = math.sqrt(w)
            self.connectionIncidenceMatrix[rowIndexRangeToNode, colIndexRange] = -math.sqrt(w)

        # Force into lil_matrix since scipy wants to convert to csr after matmul.
        self.connectionLaplacianMatrix = scipy.sparse.lil_matrix(
            self.connectionIncidenceMatrix * self.connectionIncidenceMatrix.transpose())

    def updateEdgeSignature(self, edge, rotation):
        fromNode = edge[0]
        toNode = edge[1]
        edgeIndex = list(self.edges()).index(edge)
        d = self.dimConnection
        O_sparse = scipy.sparse.lil_matrix(rotation)
        w = nx.adjacency_matrix(self)[fromNode, toNode]

        self.connectionIncidenceMatrix[(toNode * d):((toNode + 1) * d), (edgeIndex * d):((edgeIndex + 1) * d)] = (
                                                                                                                     -math.sqrt(w)) * O_sparse
        self.connectionLaplacianMatrix[(fromNode * d):((fromNode + 1) * d), (toNode * d):((toNode + 1) * d)] = (
                                                                                                                   -w) * O_sparse
        self.connectionLaplacianMatrix[(toNode * d):((toNode + 1) * d), (fromNode * d):((fromNode + 1) * d)] = (
                                                                                                                   -w) * O_sparse.T

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


def cnxFromImageDirectory(filePath, intrinsicDimension, k=None, nImages=None, save_dir_root=None, tol=1e-3):
    Y, labelsMat, _ = puppets_data.puppets_data(filePath)

    np.random.seed(42)
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
    
    cnxAdjacency = buml_obj.LocalViews.U + buml_obj.LocalViews.U.T -\
                    scipy.sparse.identity(nImages, format='lil')
    cnx = ConnectionNetworkX(cnxAdjacency, intrinsicDimension)
    cnx.imageData = X
    
    nRemoteEdges = 0
    totalEdgesBeforeRemoval = cnx.nEdges
    for i in tqdm(range(nImages)):
        n_i = nx.neighbors(cnx, i)
        for j in [j for j in n_i if j > i]:

            n_ij = buml_obj.LocalViews.U[i,:].multiply(buml_obj.LocalViews.U[j,:]).nonzero()[1]

            X_Uij_i = buml_obj.LocalViews.local_param_post.eval_({'view_index': i, 'data_mask': n_ij})
            X_Uij_j = buml_obj.LocalViews.local_param_post.eval_({'view_index': j, 'data_mask': n_ij})

            X_Uij_i = X_Uij_i - X_Uij_i.mean(axis=0)[np.newaxis, :]
            X_Uij_j = X_Uij_j - X_Uij_j.mean(axis=0)[np.newaxis, :]
            
            svdvals_ = svdvals(np.dot(X_Uij_i.T,X_Uij_j))
            if svdvals_[-1] > tol:
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

    g.add_edges_from([
                         ((x, y), (x+1, y+1))
                         for x in range(width - 1)
                         for y in range(height - 1)
                     ] + [
                         ((x+1, y), (x, y+1))
                         for x in range(width - 1)
                         for y in range(height - 1)
                     ], weight=math.sqrt(2))

    cnx = ConnectionNetworkX(nx.adjacency_matrix(g), intrinsicDimension)

    ge = {}

    if ((height > 1) & (width > 1)):
        for node in cnx.nodes:
            x = (node % width) / (width - 1)
            y = 1 - (node // height) / (height - 1)
            ge[node] = (x, y)

        cnx.gridEmbedding = ge

    cnx.width = width
    cnx.height = height
    return cnx

def loss_fn(phi, B, w, c):
    loss0 = -torch.sum(phi*c)

    loss1 = torch.matmul(B, phi).reshape((w.shape[0],-1))
    loss1 = torch.linalg.norm(loss1, dim=1)
    loss1 = loss1 - w
    loss1 = torch.nn.ReLU()(loss1)
    loss1 = torch.sum(loss1**2)

    return loss0, loss1

def active_edges(phi, B, w, c):
    loss = torch.matmul(B, phi).reshape((w.shape[0],-1))
    loss = torch.linalg.norm(loss, dim=1)
    loss = loss - w
    return loss

def optimal_J(phi, B, w, alpha, d):
    Bphi = torch.matmul(B, phi).reshape((w.shape[0],-1))
    loss = torch.linalg.norm(Bphi, dim=1)
    loss = (loss - w)/alpha
    J2 = torch.nn.ReLU()(loss)
    temp = w + alpha*J2
    J = Bphi * torch.unsqueeze(J2/temp, 1)
    return -J

def optimize(B, w, c, alpha, learning_rate, n_epochs, phi0 = None, print_freq=10):
    if phi0 is None:
        phi = torch.randn(B.shape[1], 1, requires_grad=True)
    else:
        phi = torch.tensor(phi0, requires_grad=True)
    optimizer = torch.optim.Adam([phi], lr=learning_rate)
    for epoch in range(n_epochs):
        # Compute loss
        loss0, loss1 = loss_fn(phi, B, w, c)
        loss = loss0 + (0.5/alpha)*loss1

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % print_freq == 0:
            print(f"epoch: {epoch}, loss: {loss:>7f}, loss0: {loss0:>7f}, loss1: {loss1:>7f}")
    return phi