import numpy as np
import scipy as sc
import sklearn as sk
from sklearn.clusters import AgglomerativeClustering


def compute_effective_rank_matrix(pi):
    U, Lambda, _ = np.linalg.svd(pi)
    Lambda /= np.sum(np.abs(Lambda))
    return np.exp(-1.0 * np.sum(Lambda * np.log(Lambda)))

def compute_distance_to_extremities(pi):
	n, _ = pi.shape
    return np.linalg.norm(pi-np.eye(n)), np.linalg.norm(pi-1.0/n *np.ones((n,n)))


def compute_difference_wrt_hc(K):
    ### Look at what happens along the path.
    print "not implemented yet"
    return np.nan