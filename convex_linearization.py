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

TOL_PROJ = 1e-4
TOL_INC = 5
TOL = 1e-2
MAX_ITER_PROJ = 1e4
MAXITER_FISTA = 200
ALPHA = 0.5


def hcc_FISTA_denoise(K, B, pi_prev, lambd, alpha=ALPHA, maxiterFISTA=MAXITER_FISTA,
                      eta=1.0, tol=TOL, verbose=True, tol_projection=TOL_PROJ,
                      max_iter_projection=MAX_ITER_PROJ,
                      logger = None):
    ''' Hierarchical clustering algorithm based on FISTA (dual)
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
    delta_p     :      list of updates in p (for each iteration)
    delta_q     :      list of updates in q (for each iteration)
    dual        :      dual updates
    
    --------------------------------------------------------------------------
    '''
    # Initialization
    x_k, x_km1, y_k = pi_prev, pi_prev, pi_prev
    n_nodes, _ = pi_prev.shape
    if type(K) == np.matrixlib.defmatrix.matrix:
        mask = np.array(K).flatten().reshape([-1,]).nonzero()[0]
    else:
        indices = K.nonzero()
        mask = []
        for i in range(K.nnz):
            mask += [indices[0][i]*n_nodes+indices[1][i]]
        mask = np.array(mask)


    I = sc.sparse.eye(n_nodes)
    delta_k=sc.sparse.lil_matrix((n_nodes, n_nodes**2))
    for ii in range(n_nodes):
        ind =K[ii, :].nonzero()[1]
        delta_k[ii, ii * n_nodes + ind] = K[ii, ind]
        delta_k[ii, ii + ind * n_nodes] = -K[ii, ind]
        delta_k[ii, ii + ii * n_nodes] = 0.0
    delta_k = delta_k[:, mask]
    delta_k = delta_k.todense()

    lmax = np.linalg.norm(delta_k,'fro')**2
    gamma = 16 * max([alpha**2,(1-alpha)**2])*lmax * lambd**2
    if verbose: print("lmax",lmax, "gamma", gamma)
    I = sc.sparse.eye(n_nodes)
    update = (delta_k.T.dot(x_k.T)).T
    #print("update.max(())", update.max())

    q = np.zeros((n_nodes, len(mask)))
    p = project_unit_ball(update,
                         is_sparse=False)


    q[np.where(update != 0)]= project_unit_cube(update[np.where(update != 0)],
                                                is_sparse=False)



    index_rev = [jj*n_nodes + ii  for ii in range(n_nodes) for jj in range(n_nodes)]
    #print("init p", p.max(), q.max(), q.min())
    t_k, it = 1, 1
    converged = False
    
    t_k, it = 1, 1
    converged = False

    tic0 = time.time()
    delta_x = []
    delta_p = []
    delta_q = []
    dual = []
    it = 0
    eps_reg =1e-5

    p_old = copy.deepcopy(p)
    q_old = copy.deepcopy(q)
    r = copy.deepcopy(p)
    s = copy.deepcopy(q)
    inc = 0
    vpos = np.vectorize(lambda x: x if x>0 else 0)
    while not converged:
        belly = (alpha * r + (1-alpha) * s).dot(delta_k.T)
        proj = project_DS2(vpos(B-lambd * belly),  max_it=max_iter_projection, eps = tol_projection)
        x_k = proj
        L_x = proj.dot(delta_k)
        #print("update.max(())", L_x.max())

        update_p = p + 2.0 * alpha *lambd / gamma * L_x
        p = project_unit_ball(update_p,
                              is_sparse=False)

        update_q = q+ 2.0 * (1-alpha) / gamma * L_x
        inv_update_q = copy.deepcopy(update_q)
        inv_update_q[np.where(update_q!=0)] = 1.0/np.abs(update_q[np.where(inv_update_q!=0)])
        q =   np.multiply(inv_update_q, update_q)    

        if verbose: print("max q, p",q.max(), q.min(), p.max(), p.min())

        #### Check convergence
        delta_x.append(np.linalg.norm( x_k - x_km1, 'fro')/np.linalg.norm(x_km1,'fro'))
        delta_p.append(np.linalg.norm(p - p_old, 'fro'))
        delta_q.append(np.linalg.norm(q - q_old, 'fro'))
        
        if delta_x[-1] < TOL:
                inc += 1
        else:
                inc = 0

        converged = (math.sqrt((alpha**2 * np.linalg.norm(p-p_old,'fro')**2 
                                + (1 - alpha)**2 * np.linalg.norm(q-q_old,'fro')**2))
                     / np.max([0, math.sqrt((alpha**2 * np.linalg.norm(p_old,'fro')**2 
                                 + (1 - alpha)**2 * np.linalg.norm(q_old,'fro')**2))]) 
                     < tol)\
                     or (delta_x[-1] < tol and it >= 2)\
                     or (it > maxiterFISTA)
        if verbose: 
            if logger is not None:  logger.info("norm dual= %f"%(math.sqrt((alpha**2 * np.linalg.norm(p-p_old,'fro')**2 
                                + (1 - alpha)**2 * np.linalg.norm(q - q_old, 'fro')**2))
                     / np.max([0, math.sqrt((alpha**2 * np.linalg.norm(p_old, 'fro')**2 
                                 + (1 - alpha)**2 * np.linalg.norm(q_old, 'fro')**2))])))
            else: print("norm", math.sqrt((alpha**2 * np.linalg.norm(p - p_old, 'fro')**2 
                                + (1 - alpha)**2 * np.linalg.norm(q - q_old, 'fro')**2))
                     / np.max([0, math.sqrt((alpha**2 * np.linalg.norm(p_old, 'fro')**2 
                                 + (1 - alpha)**2 * np.linalg.norm(q_old, 'fro')**2))]))

        t_kp1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k**2))

        r = p + (t_k - 1.0) / t_kp1 * (p - p_old)
        s = q + (t_k - 1.0) / t_kp1 * (q - q_old)
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
