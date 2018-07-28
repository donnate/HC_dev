import numpy as np
import scipy as sc
from scipy.spatial.distance import cdist


METRIC = 'euclidean'
EPS = 1e-2

def post_process(pi, metric=METRIC, eps=EPS):
    ''' Once we have a path, we can process it to get new clusters
    INPUT
    ---------------------------------------------------------------
    pi              :   dictionary (each key corresponding to a value of 
                        lambda) of the different vector assignments
    metric          :   type of metric to use to compare the different
                        centroids (default: 'euclidean', can use any type
                        of metric accepted by cdist)
    eps             :   the tolerance level for declaring that two centroids
                        are fused( correspond to the same cluster)

    OUTPUT
    ---------------------------------------------------------------
    clusters        :   dictionary (each key corresponding to a value of 
                        lambda) of the cluster assignment at each level
    cluster_distance:   dictionary (each key corresponding to a value of 
                        lambda) of the distances between clusters
                        at each level
    '''
    lambdas = pi.keys()
    n_nodes, _ = pi[lambdas[0]].shape #gets the number of nodes
    clusters = {}
    cluster_distance = {}
    # Version 1
#    for lambd in lambdas:
#         clusters[k] = range(n_nodes)
#         dist = cdist(pi[lambd].T, pi[lambd].T,
#                      metric=metric)
#         # Assign sequentially the nodes to their corresponding clusters
#         for i in range(1, n_nodes):
#             for j in range(i):
#                 if dist[i,j]<eps:
#                     cluster[lambd][i] = cluster[lambd][j]
#         n_clust = len(np.unique(cluster[lambd]))
#         cluster_distances[lambd] = np.zeros((n_clust, n_clust))
#         list_clusters = np.unique(cluster[lambd])
#         for i in range(1, n_clust):
#             c_i = list_clusters[i]
#             index_i = np.where(cluster[lambd] == c_i)[0]
#             for j in range(i):
#                 c_j = list_clusters[j]
#                 index_j = np.where(cluster[k] == c_j)[0]
#                 cluster_distance[lambd][i,j] = np.mean(dist[c_i,:][:,c_j])
    # Version 2- TO DO: but each cluster should 
    for lambd in lambdas:
        clusters[k] = pi.lambd
        dist = cdist(pi[lambd].T, pi[lambd].T,
                     metric=metric)
        # Assign sequentially the nodes to their corresponding clusters
        for i in range(1, n_nodes):
            for j in range(i):
                if dist[i,j]<eps:
                    cluster[lambd][i] = cluster[lambd][j]
        n_clust = len(np.unique(cluster[lambd]))
        cluster_distances[lambd] = np.zeros((n_clust, n_clust))
        list_clusters = np.unique(cluster[lambd])
        for i in range(1, n_clust):
            c_i = list_clusters[i]
            index_i = np.where(cluster[lambd] == c_i)[0]
            for j in range(i):
                c_j = list_clusters[j]
                index_j = np.where(cluster[k] == c_j)[0]
                cluster_distance[lambd][i,j] = np.mean(dist[c_i,:][:,c_j])

    return clusters, cluster_distance
