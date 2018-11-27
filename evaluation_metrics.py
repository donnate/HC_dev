import numpy as np
import pandas as pd
import scipy as sc
import sklearn as sk
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import calinski_harabaz_score, homogeneity_completeness_v_measure,\
                            fowlkes_mallows_score, silhouette_score


def compute_RSM(D, Z, type_hc="tree", lambdas = None):
    N_NODES = D.shape[0]
    sim = np.zeros((N_NODES, N_NODES))
    if type_hc == 'tree':
        assignements = {i: [i] for i in range(N_NODES)}
        for i in range(Z_ward.shape[0]):
            a =  Z[i, 0]
            b =  Z[i, 1]
            assignements[N_NODES + i] = list(assignements[a]) + list(assignements[b])
            for aa in list(assignements[a]) + list(assignements[b]):
                for bb in list(assignements[a]) + list(assignements[b]):
                    sim[int(aa), int(bb)] = sim[int(aa), int(bb)] + 1
        sim = sim / np.diag(sim).max()
        return sim
    else:
    	lambdas = np.sort(lambdas)
    	increment_lambda = np.diff(lambdas)
    	DELTA_LAMBDA = lambdas[-1] - lambdas[0]
    	if lambdas is None:
    	    print('Error: must provide the lambda path')
    	    return None
    	dist_old = np.eye(N_NODES) 
    	for i, l in enumerate(np.sort(lambdas)):
    		dist = cdist(Z[l], Z[l], 'cosine')  ### distance matrix between the different elements
    		if i > 0:
    			increment_distance = dist - dist_old
    			sim += increment_lambda[i-1]/DELTA_LAMBDA * increment_distance
    		
	sim = pd.DataFrame(sim, columns = D.columns, index=D.index)
    return sim

	
def classification_performance(X, labels, train_labels):
	res = [accuracy_score(labels, train_labels), 
	       f1_score(labels, train_labels, average='macro'), 
	       matthews_corrcoef(labels, train_labels)
           ]
    return pd.DataFrame(res, columns = ['Accuracy', 'F1', 'matthews_corrcoef'])

def clustering_performance(X, labels, train_labels):
	res = [calinski_harabaz_score(X, labels, train_labels)]+
		  [e for e in homogeneity_completeness_v_measure(labels, train_labels)]]+
		  [fowlkes_mallows_score(labels, train_labels),
		   silhouette_score(X, labels)]
    return pd.DataFrame(res, columns = ['CH', 'Homogeneity', 'Completeness', 'V_meas',
                                        'FM', 'Silhouette'])           

