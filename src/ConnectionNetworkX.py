import math
import pdb

import networkx as nx
import scipy
import numpy as np
import random
from tqdm import tqdm
import torch

from scipy.linalg import svdvals, svd, pinv
from scipy.sparse import kron, csr_matrix, lil_matrix, identity, triu

from pyLDLE2 import datasets
from pyLDLE2 import buml_
from pyLDLE2 import util_

from sklearn.neighbors import NearestNeighbors


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
        adj_mat = triu(nx.adjacency_matrix(self))
        directed_self = nx.DiGraph(adj_mat)
        #W = kron(nx.adjacency_matrix(directed_self), np.ones((self.d, self.d)))
        B_ = -nx.incidence_matrix(directed_self, oriented=True, weight='weight')
        B_.data = np.sign(B_.data)*np.sqrt(np.abs(B_.data))
        self.B = kron(B_, np.eye(self.d))
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
        
    def removeEdge(self, edge, w=None):
        u = edge[0]
        v = edge[1]
        i = list(self.edges()).index(edge)
        d = self.d
        z1 = lil_matrix((d, d))
        z2 = lil_matrix(np.eye(d))
        
        if w is None:
            w = self[u][v]['weight']

        d_out = self.CL[u * d, u * d]
        d_in = self.CL[v * d, v * d]

        # self.remove_edge(fromNode, toNode)
        self.setBlockOfB(u, i, z1)
        self.setBlockOfB(v, i, z1)
        self.setBlockOfCL(u, u, (d_out - w) * z2)
        self.setBlockOfCL(v, v, (d_in - w) * z2)
        self.setBlockOfCL(u, v, z1)
        self.setBlockOfCL(v, u, z1)

    def printConnectionLaplacianEigenvalues(self, k=None, which="LM", sigma=-1e-3, showBalanced=True):
        if k is None:
            k = self.d+1
        vals = scipy.sparse.linalg.eigsh(self.CL, which=which, sigma=sigma,
                                         k=k, return_eigenvectors=False)
        tolerance = 1e-8
        print(vals)
        if showBalanced:
            if abs(vals[self.d-1]) < tolerance:
                print("MOST LIKELY CONSISTENT: |lambda_min| < 1e-8. ")
            else:
                print("MOST LIKELY INCONSISTENT: |lambda_min| >= 1e-8. ")


def epanechnikov_kernel(dist, eps):
    return np.sqrt((1-dist**2/eps)*np.clip(dist/np.sqrt(eps), 0, 1))

def gaussian_kernel(W, eps):
    W.data = np.exp(-W.data**2/eps)*np.clip(W.data/np.sqrt(eps), 0, 1)
    W.eliminate_zeros()
    return W

# eps_pca <= eps
def cnxFromData(X, eps_pca, eps, d,  tol=1e-6, kernel='gaussian', triv_sigma=False):
    # Form neighborhoods
    neigh = NearestNeighbors(radius=eps)
    neigh.fit(X)
    neigh_inds_pca = neigh.radius_neighbors(radius=eps_pca, return_distance=False)
    n, p = X.shape
    O = np.zeros((n, p, d))
    for i in range(n):
        n_i = neigh_inds_pca[i] # N_i = |n_i|
        X_i = (X[n_i,:] - X[i,:][None,:]).T # p x N_i
        X_i_norm = np.linalg.norm(X_i, axis=0) # N_i dimensional
        D_i = epanechnikov_kernel(X_i_norm, eps_pca) #N_i dimensional
        B_i = X_i * D_i[None,:]
        U_i, Sigma_i, V_iT = svd(B_i)
        O_i = U_i[:,:d]
        O[i,:,:] = O_i
    
    neigh_graph = neigh.radius_neighbors_graph(mode='distance')
    if kernel == 'gaussian':
        G = nx.Graph(gaussian_kernel(neigh_graph, eps))
    else:
        G = nx.Graph(neigh_graph)
    
    sigma = {}
    nRemoteEdges = 0
    I_d = np.eye(d)
    totalEdgesBeforeRemoval = len(G.edges)
    print('Total edges before removal:', totalEdgesBeforeRemoval)
    for i in tqdm(range(n)):
        n_i = nx.neighbors(G, i)
        O_i = O[i,:,:]
        O_iO_iT = O_i.dot(O_i.T)
        for j in [j for j in n_i if j > i]:
            O_j = O[j,:,:]
            O_jO_jT = O_j.dot(O_j.T)
            grassmannian = np.linalg.norm(O_iO_iT-O_jO_jT, ord=2)
            if grassmannian < tol:
                if triv_sigma:
                    sigma[(i,j)] = I_d
                else:
                    O_iTO_j = O_i.T.dot(O_j)
                    U, Sigma, VT = svd(O_iTO_j)
                    sigma_ij = U.dot(VT)
                    sigma[(i,j)] = sigma_ij
            else:
                G.remove_edge(i, j)
                nRemoteEdges += 1
    
    print('Proportion of edges which were removed due to remoteness: ', nRemoteEdges / totalEdgesBeforeRemoval)
    cnx = ConnectionNetworkX(nx.adjacency_matrix(G), d)
    #cnx.printConnectionLaplacianEigenvalues()
    assert cnx.number_of_nodes() == n, 'a node is missing'

    edge_attribs = {}
    for i in tqdm(range(n)):
        n_i = nx.neighbors(cnx, i)
        for j in [j for j in n_i if j > i]:
            sigma_ij = sigma[(i,j)]
            cnx.updateEdgeSignature((i,j), sigma_ij)
            if d == 2:
                edge_attribs[(i,j)] = {'theta': np.arctan2(sigma_ij[1,0], sigma_ij[0,0]) + np.pi,
                                       'reflection': np.linalg.det(sigma_ij)}
    
    if d == 2:
        nx.set_edge_attributes(cnx, edge_attribs)
            
    #cnx.printConnectionLaplacianEigenvalues()
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

def activenes_of_edges(phi, B, w, c):
    loss = B.dot(phi).reshape((w.shape[0],-1))
    loss = np.linalg.norm(loss, axis=1)
    loss = loss - w
    return loss

def optimal_primal(phi, B, w, c, alpha, d):
    Bphi = B.dot(phi).reshape((w.shape[0],-1))
    loss = np.linalg.norm(Bphi, axis=1)
    loss = (loss - w)/alpha
    J2 = loss * (loss > 0)
    temp = w + alpha*J2
    J = Bphi * (J2/temp)[:,None]
    J = -J
    err = B.T.dot(J.flatten()[:,None]) - c
    print('mean abs err of sum_{v in V} ||(B^TJ-c)(v)||_1', np.mean(np.abs(err)))
    return J

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

def optimize_(B, W, c, learning_rate=0.1, alpha=1, n_epochs=10000, phi0=None):
    B_torch = torch.tensor(B.toarray().astype('float32'))
    w_torch = torch.tensor(W.astype('float32'))
    c_torch = torch.tensor(c.astype('float32'))
    
    if phi0 == 'least_squares':
        phi0 = pinv(B.T.dot(B).toarray()).dot(c).astype('float32')
    
    if phi0 is not None:
        print('Initial loss:', loss_fn(torch.tensor(phi0), B_torch, w_torch, c_torch))
    
    phi = optimize(B_torch, w_torch, c_torch, alpha, learning_rate, n_epochs, phi0 = phi0)
    return phi.detach().numpy(), phi0