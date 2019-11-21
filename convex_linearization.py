"""
Code for the dual FISTA algorithm for hierarchical
convex clustering
"""
import copy
import math
import numpy as np
import scipy as sc
import time

from projections import *
from utils import *
from utils_graphs import *

TOL_PROJ = 1e-4
TOL_INC = 5
TOL = 1e-2
MAX_ITER_PROJ = 1e4
MAXITER_FISTA = 200
ALPHA = 0.5


def hcc_FISTA_linearized(K, pi_prev, lambd, alpha=ALPHA, maxiterFISTA=MAXITER_FISTA,
                      eta=1.0, tol=TOL, verbose=True, tol_projection=TOL_PROJ,
                      max_iter_projection=MAX_ITER_PROJ,
                      logger = None):
    ''' Hierarchical clustering algorithm based on linearization (dual)
    Input: similarity matrix K assumed to be from a Mercer kernel (or at least PSD)
    Output: the regularized soft membership assignment matrix
    --------------------------------------------------------------------------
    
    INPUT:
    -----------------------------------------------------------
    K            :      the similarity matrix
    pi_prev      :      initialization value for pi (warm start)
    lambd        :      the level of regularization desired 
    alpha        :      parameter for the total variation mixed penalty
    maxiterFISTA :      max number of iterations for FISTA (updates in X)
    tol          :      tolerance level for the stopping criterion,
                        as a fraction of the number of nodes
    sparse       :      boolean: should the algorithm assume sparisty of K?
    verbose      :      boolean: allow printing of various statistics
                        and intermediary parameters
    OUTPUT:
    -----------------------------------------------------------
    X           :      the corresponding soft clustering assignment matrix
    t           :      the time that the procedure took
    delta_x     :      list of updates in x (for each iteration)
    
    --------------------------------------------------------------------------
    '''
    # Initialization
    x_k, x_km1, y_k = pi_prev, pi_prev, pi_prev
    n_nodes, _ = pi_prev.shape
    K_tilde = K
    np.set_diagonal(K_tilde, 0)
    delta_k = delta_k[:, mask]
    delta_k = delta_k.todense()
    
    t_k, it = 1, 1
    converged = False
    tic0 = time.time()
    delta_x =[]
    dual = []
    it = 0
    eps_reg =1e-5

    inc = 0
    vpos = np.vectorize(lambda x: x if x>0 else 0)
    while not converged:
        belly = (alpha * r + (1-alpha) * s).dot(delta_k.T)
        proj = project_DS2(vpos(B-lambd * belly),  max_it=max_iter_projection, eps = tol_projection)
        #print("update.max(())", L_x.max()

        if verbose: print("max q, p",q.max(), q.min(), p.max(), p.min())

        #### Check convergence
        delta_x.append(np.linalg.norm( x_k - x_km1, 'fro')/np.linalg.norm(x_km1,'fro'))
        converged =  (delta_x[-1] < tol and it >= 2)\
                     or (it > maxiterFISTA)

        t_k = t_kp1
        x_km1 = x_k
        p_old, q_old = copy.deepcopy(p), copy.deepcopy(q)
        it += 1

        if verbose:
             try: eff_rank = efficient_rank(x_k)
             except: eff_rank = np.nan
             if logger is not None: 
                    logger.info('inner loop %i: efficient rank x_k: %f, delta_x: %f'%(it, eff_rank, delta_x[-1])
                               )
             else: 
                print('inner loop %i: efficient rank x_k: %f, delta_x: %f'%(it, eff_rank, delta_x[-1]))

    toc0 = time.time()
    if verbose: print("time:",time.time() - tic0)
    belly = (alpha * p+ (1.0 - alpha) * q).dot(delta_k.T)
    x_k = project_DS2(B - lambd * belly, max_it=max_iter_projection, eps = tol_projection)
    val = np.trace(x_k.T.dot(K.todense().dot(x_k)) 
                   - 2*(K.todense() -lambd *(delta_k.dot(alpha * p.T + (1.0 - alpha) * q.T)).dot(x_k)))
    if logger is not None:
        logger.info('--------------------------------------------------------')
    return x_k, toc0-tic0, delta_x, delta_p, delta_q, dual, val
