import numpy as np
import scipy as sc


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