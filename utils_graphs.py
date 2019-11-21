import numpy as np
import scipy as sc

def create_similarity_matrix(W, type_lap, alpha):
    n_nodes, _ = W.shape
    sqrtv = np.vectorize(lambda x: 1.0/np.sqrt(x) if x > 1e-6 else 0.0)
    if type_lap == "normalized_2_hops":
        K = W.T.dot(W)
        Deg = np.diagflat(sqrtv(K.diagonal()))
        K = Deg.dot(K.dot(Deg)) + alpha * np.diag(np.ones(n_nodes))
    elif type_lap == "normalized_laplacian":
        Deg =  np.diagflat(sqrtv(W.sum(1)))
        K = np.diag((alpha + 1.0) * np.ones(n_nodes)) + np.diag(Deg).dot(W.dot(np.diag(Deg)))
    elif type_lap == "laplacian":
        Deg = np.sum(W, 1)
        K = np.diag(alpha * np.ones(n_nodes) + Deg) + W
    elif type_lap == "regularized_self":
        K = W + alpha * np.diag(np.ones(n_nodes))
    else: 
        print("type of similarity not recognized")
        return np.nan
    K = sc.sparse.csc_matrix(K)
    return K
