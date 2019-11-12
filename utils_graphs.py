import numpy as np
import scipy as sc

def create_similarity_matrix(W, type_lap, alpha):
    n_nodes, _ = W.shape
    sqrtv = np.vectorize(lambda x: 1.0/np.sqrt(x) if x > 1e-6 else 0.0)
    if type_lap == "normalized_2_hops":
        K = W.T.dot(W)
        Deg = np.diagflat(sqrtv(K.diagonal()))
        K = sc.sparse.csc_matrix(Deg.dot(K.dot(Deg)) + alpha * np.diag(np.ones(n_nodes)))
    elif type_lap == "normalized_laplacian":
        Deg =  np.diagflat(sqrtv(K.sum(1)))
        K = - W + sc.sparse.csc_matrix(alpha * np.diag(np.ones(n_nodes)) + np.diag(Deg))
    else:
	K = W + alpha * sc.sparse.csc_matrix(np.diag(np.ones(n_nodes)))
    return K
