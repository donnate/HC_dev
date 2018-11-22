import numpy as np
import scipy as sc
from scipy.spatial.distance import cdist

def diffusion_map(A, K_n):
    thr_inv = np.vectorize(lambda x:  x if x > 1e-9 else 1.0)
        
    Dinv = sc.sparse.diags(1.0/thr_inv(np.array(A.sum(1))).flatten(),0)
    n = A.shape[0]
    K = sc.sparse.csc_matrix((n, n))
    for k in range(K_n):
        if k == 0:
            Pk = Dinv.dot(A)
        else:
            Pk = Pk.dot(Dinv.dot(A))
    K = 0.5 * (Pk + Pk.T)
    DK = sc.sparse.csr_matrix(np.diag(K.diagonal()**(-0.5)))
    K = DK.dot(K.dot(DK))
    return K


def efficient_rank(pi_prev):
    _,lam,_=np.linalg.svd(pi_prev)
    S = lam.sum()
    return np.exp(np.sum(-np.log(lam/S)*lam/S))
    
    
def modularity(A, labels):
    A2 = copy.deepcopy(A)
    np.fill_diagonal(A2,0)
    m = float(A2.sum())
    Q = 1.0
    k = A2.sum(1)
    B = np.zeros(A2.shape)
    for c in np.unique(labels):
        index_c  = np.where(labels == c)
        for cc in index_c[0]:
            B[cc, (labels == c)]+=1
    np.fill_diagonal(B,0)
    #Q += - float(k.T.dot(B.dot(k))/(2.0*m)**2)
    Q = 1.0/(2*m) *np.multiply( A2-k.reshape([-1,1]).dot(k.reshape([1,-1]))/(2*m), B).sum()
    return Q


def elbow_method(X, labels):
    D = 0
    for c in np.unique(labels):
        index_c  = np.where(labels == c)[0]
        mu_c = X[index_c,:].mean(0)
        D += (cdist(X[index_c,:], mu_c) **2).sum()
    return D


def elbow_variance_method(X, labels):
    D = 0
    for c in np.unique(labels):
        index_c  = np.where(labels == c)[0]
        index_c_perp = np.setdiff1d(range(len(labels)), index_c)
        D += (cdist(X[index_c,:], X[index_c_perp,:])**2).sum()

    return D/ (cdist(X, X)**2).sum()