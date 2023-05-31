import math
import pdb

import networkx as nx
import scipy
import numpy as np
import random
from tqdm import tqdm
import torch

from scipy.linalg import svdvals
from scipy.sparse import kron, csr_matrix, lil_matrix, identity

import puppets_data
from pyLDLE2 import datasets
from pyLDLE2 import buml_
from pyLDLE2 import util_


class ConnectionNetworkX(nx.Graph):

    def __init__(self, adj_mat, d):
        super().__init__(adj_mat)
        self.d = d
        self.initializeConenctionLaplacian()

        self.imageData = None
        self.width = None
        self.height = None

        self.gridEmbedding = None # To be implemented when initializing as a grid graph.
        
    def initializeConenctionLaplacian(self):
        directed_self = nx.DiGraph(self.edges())
        #W = kron(nx.adjacency_matrix(directed_self), np.ones((self.d, self.d)))
        self.B = kron(nx.incidence_matrix(directed_self, oriented=True), np.eye(self.d))
        self.CL = self.B.dot(self.B.T)
        self.B = self.B.tolil()
        self.CL = self.CL.tolil()

    def setBlockOfB(self, u, i, z):
        d = self.d
        self.B[(u * d):((u + 1) * d), (i * d):((i + 1) * d)] = z
        
    def setBlockOfCL(self, u, v, z):
        d = self.d
        self.CL[(u * d):((u + 1) * d), (v * d):((v + 1) * d)] = z
        
    def updateEdgeSignature(self, edge, rotation, w=None):
        u = edge[0]
        v = edge[1]
        i = list(self.edges()).index(edge)
        O = lil_matrix(rotation)
        if w is None:
            w = self[u][v]['weight']

        #print(u, v, i, self.B.shape, O.shape)
        self.setBlockOfB(v, i, -math.sqrt(w) * O)
        self.setBlockOfCL(u, v, -w * O)
        self.setBlockOfCL(v, u, -w * O.T)
        
    def removeEdge(self, edge):
        u = edge[0]
        v = edge[1]
        i = list(self.edges()).index(edge)
        d = self.d
        z1 = lil_matrix((d, d))
        z2 = lil_matrix(np.eye(d))

        d_out = self.CL[u * d, u * d]
        d_in = self.CL[v * d, v * d]

        # self.remove_edge(fromNode, toNode)
        self.setBlockOfB(u, i, z1)
        self.setBlockOfB(v, i, z1)
        self.setBlockOfCL(u, u, (d_out - 1) * z2)
        self.setBlockOfCL(v, v, (d_in - 1) * z2)
        self.setBlockOfCL(u, v, z1)
        self.setBlockOfCL(v, u, z1)

    def printConnectionLaplacianEigenvalues(self, k=10, which="LM", sigma=-1e-3, showBalanced=True):
        vals = scipy.sparse.linalg.eigsh(self.CL, which=which, sigma=sigma, k=k, return_eigenvectors=False)
        tolerance = 1e-8
        print(vals)
        if showBalanced:
            if abs(vals[k - 1]) < tolerance:
                print("MOST LIKELY CONSISTENT: |lambda_min| < 1e-8. ")
            else:
                print("MOST LIKELY INCONSISTENT: |lambda_min| >= 1e-8. ")


def cnxFromData(X, k, d,  tol=1e-3):
    buml_obj = buml_.BUML(d = d, local_opts={'algo': 'LPCA', 'k': k},
                          intermed_opts={'eta_max': 1},
                          vis_opts={'c': X[:, 0], 'save_dir': None},
                          verbose=True, debug=True, exit_at='local_views')
    buml_obj.fit(X=X)
    U = buml_obj.LocalViews.U
    n = X.shape[0]
    I_n = identity(n, format='lil')
    G = nx.Graph((U + U.T) - I_n)
    
    sigma = {}
    nRemoteEdges = 0
    totalEdgesBeforeRemoval = len(G.edges)
    for i in tqdm(range(n)):
        n_i = nx.neighbors(G, i)
        for j in [j for j in n_i if j > i]:

            n_ij = buml_obj.LocalViews.U[i,:].multiply(buml_obj.LocalViews.U[j,:]).nonzero()[1]

            X_Uij_i = buml_obj.LocalViews.local_param_post.eval_({'view_index': i, 'data_mask': n_ij})
            X_Uij_j = buml_obj.LocalViews.local_param_post.eval_({'view_index': j, 'data_mask': n_ij})

            X_Uij_i = X_Uij_i - X_Uij_i.mean(axis=0)[np.newaxis, :]
            X_Uij_j = X_Uij_j - X_Uij_j.mean(axis=0)[np.newaxis, :]
            
            svdvals_ = svdvals(np.dot(X_Uij_i.T,X_Uij_j))
            if svdvals_[-1] > tol:
                Tij, _ = scipy.linalg.orthogonal_procrustes(X_Uij_i, X_Uij_j)
                sigma[(i,j)] = Tij
            else:
                G.remove_edge(i, j)
                nRemoteEdges += 1
    
    
    print('Proportion of edges which were removed due to remoteness: ', nRemoteEdges / totalEdgesBeforeRemoval)
    cnx = ConnectionNetworkX(nx.adjacency_matrix(G), d)
    
    assert cnx.number_of_nodes() == n, 'a node is missing'

    for i in tqdm(range(n)):
        n_i = nx.neighbors(cnx, i)
        for j in [j for j in n_i if j > i]:
            cnx.updateEdgeSignature((i,j), sigma[(i,j)])
    
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