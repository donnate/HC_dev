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

def hcc_FISTA_denoise(K, B, pi_prev, lambd, alpha=0.5, maxiterFISTA=100, eta=0.1, tol=1e-2,
                      verbose=True, tol_projection=1e-4, max_iter_projection=100000,
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
    print(K, type(K))
    if type(K) == np.matrixlib.defmatrix.matrix:
        mask = np.array(K).flatten().reshape([-1,]).nonzero()[0]
    else:
        indices = K.nonzero()
        mask = []
        for i in range(K.nnz):
            mask += [indices[0][i]*n_nodes+indices[1][i]]
        mask = np.array(mask)
     
    
    #lmax = K_tilde.max() * sc.sparse.linalg.norm(K_tilde,'fro')
    
    

    I = sc.sparse.eye(n_nodes)
    delta_k=sc.sparse.lil_matrix((n_nodes, n_nodes**2))
    for ii in range(n_nodes):
        ind =K[ii, :].nonzero()[1]
        #print ind
        delta_k[ii, ii * n_nodes + ind] = K[ii, ind]
        delta_k[ii, ii + ind * n_nodes] = -K[ii, ind]
        delta_k[ii, ii + ii * n_nodes] = 0.0
    delta_k = delta_k.tocsr()


    lmax = sc.sparse.linalg.norm(delta_k,'fro')**2
    gamma = 4 * max([alpha**2,(1-alpha)**2])*lmax * lambd**2
    if verbose: print("lmax",lmax)
    I = sc.sparse.eye(n_nodes)
    update = (delta_k[:, mask].T.dot(x_k.T)).T
    p=sc.sparse.lil_matrix((n_nodes, n_nodes**2))
    q=sc.sparse.lil_matrix((n_nodes, n_nodes**2))
    p[:, mask]= sc.sparse.lil_matrix(project_unit_ball(update,
                          is_sparse=False))
    p = p.tocsr()
    q[:,mask]= sc.sparse.lil_matrix(project_unit_cube(update,
                          is_sparse=False))
    q = q.tocsr()
    index_rev = [jj*n_nodes + ii  for ii in range(n_nodes) for jj in range(n_nodes)]
    #index_rev_mask = [(jj%n_nodes) * n_nodes + jj / n_nodes  for jj in mask]
    rk = {mask[i]: i for i in range(len(mask))}
    index_rev_mask = [rk[(jj%n_nodes)*n_nodes + jj/n_nodes]  for jj in mask]
    p = 0.5*(p - p[:,index_rev])
    q = 0.5*(q - q[:,index_rev])
    print("init p", p.max())
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
    while not converged:
        belly = (alpha * r + (1-alpha) * s).dot(delta_k.T)
        if verbose:    
            if logger is not None: logger.info("belly %f"%belly.max())
            else : print("belly ", belly.max())

        proj = project_DS2(B-lambd * belly.todense(),  max_it=max_iter_projection, eps = tol_projection)
        x_k = proj
        #x_k = project_DS_symmetric(np.array(inside),  max_it=max_iter_projection, eps = tol_projection)
        L_x = sc.sparse.csr_matrix((delta_k[:, mask].T.dot(proj.T)).T)
        update_p = p[:, mask] + 2.0 * alpha *lambd / gamma * L_x
        #update_p = p[:, mask] + 2.0 * alpha *lambd / gamma * sc.sparse.csc_matrix((delta_k.T.dot(proj.T)).T)[:, mask]
        #update_p = 0.5 * (update_p - update_p[:, index_rev_mask])
        t = project_unit_ball(update_p.todense(),
                              is_sparse=False)
        p = p.tolil()
        p[:, mask]= sc.sparse.lil_matrix(t)
        p = p.tocsr()
        
        #update_q = q[:, mask] + 2.0 * (1-alpha) / gamma * sc.sparse.csc_matrix((delta_k.T.dot(proj.T)).T)[:, mask]
        update_q = q[:, mask] + 2.0 * (1-alpha) / gamma * L_x
        #update_q = 0.5 * (update_q - update_q[:, index_rev_mask])
        t = project_unit_cube(update_q.todense(),
                              is_sparse=False)

        q = q.tolil()
        q[:, mask] = sc.sparse.lil_matrix(t)
        q = q.tocsr()
        if verbose: print(q.max(), q.min(), p.max(), p.min())

        #### Check convergence
        delta_x.append(np.linalg.norm( x_k - x_km1, 'fro')/np.linalg.norm(x_km1,'fro'))
        delta_p.append(sc.sparse.linalg.norm(p - p_old, 'fro'))
        delta_q.append(sc.sparse.linalg.norm(q - q_old, 'fro'))
        #converged = (delta_x[-1] < tol and it>1)\
        #             or (it > maxiterFISTA)
        converged = (math.sqrt((alpha**2 * sc.sparse.linalg.norm(p-p_old,'fro')**2 
                                + (1 - alpha)**2 * sc.sparse.linalg.norm(q-q_old,'fro')**2))
                     / np.max([0, math.sqrt((alpha**2 * sc.sparse.linalg.norm(p_old,'fro')**2 
                                 + (1 - alpha)**2 * sc.sparse.linalg.norm(q_old,'fro')**2))]) 
                     < tol)\
                     or (delta_x[-1] < tol and it>=3)\
                     or (it > maxiterFISTA)
        if verbose: 
            if logger is not None:  logger.info("norm : %f"%(math.sqrt((alpha**2 * sc.sparse.linalg.norm(p-p_old,'fro')**2 
                                + (1 - alpha)**2 * sc.sparse.linalg.norm(q-q_old,'fro')**2))
                     / np.max([0, math.sqrt((alpha**2 * sc.sparse.linalg.norm(p_old,'fro')**2 
                                 + (1 - alpha)**2 * sc.sparse.linalg.norm(q_old,'fro')**2))])))
            else: print("norm", math.sqrt((alpha**2 * sc.sparse.linalg.norm(p-p_old,'fro')**2 
                                + (1 - alpha)**2 * sc.sparse.linalg.norm(q-q_old,'fro')**2))
                     / np.max([0, math.sqrt((alpha**2 * sc.sparse.linalg.norm(p_old,'fro')**2 
                                 + (1 - alpha)**2 * sc.sparse.linalg.norm(q_old,'fro')**2))]))

        #dual.append(sc.sparse.linalg.norm(alpha * p[:, mask].dot((delta_k[:, mask]).T)\
        #            + (1 - alpha) * q[:, mask].dot((delta_k[:, mask]).T), 'fro'))
        t_kp1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k**2))

        r = p + (t_k - 1) / t_kp1 * (p - p_old)
        s = q + (t_k - 1) / t_kp1 * (q - q_old)
        t_k = t_kp1
        x_km1 = x_k
        p_old, q_old = copy.deepcopy(p), copy.deepcopy(q)
        it += 1

        if verbose:
             if logger is not None: 
                    logger.info(' %i: efficient rank x_k: %f, delta_x: %f'%(it,efficient_rank(x_k),delta_x[-1])
                               )
             else: 
                print(it,'efficient rank x_k', efficient_rank(x_k), 'delta', delta_x)

    toc0 = time.time()
    if verbose: 
        print("time:",time.time() - tic0)
        if logger is not None:
            logger.info("---------------------------------------------------")
    belly = (alpha * p+ (1-alpha) * q).dot(delta_k.T)
    x_k = project_DS2(B-lambd * belly.todense(),  max_it=max_iter_projection, eps = tol_projection)
    val = np.trace(x_k.T.dot(K.todense().dot(x_k)) 
                   - 2*(K.todense() -lambd *(delta_k.dot(alpha* p.T + (1.0 - alpha) * q.T)).todense().dot(x_k)))

    return x_k, toc0-tic0, delta_x, delta_p, delta_q, dual, val











def hcc_FISTA(K, pi_warm_start, lambd0, alpha =0.95,
              maxiterFISTA = 2000, tol=5*1e-3, debug_mode=True,
              lambda_spot = 0, verbose =False, logger=None):
    if debug_mode: verbose =True
    Y, pi_prev, pi_prev_old = [pi_warm_start] * 3
    evol_efficient_rank=[]
    L = 2 * sc.sparse.linalg.norm(K, 'fro')
    lambd = 2 * lambd0 / L
    t_k = 1
    tic = time.time()
    delta_pi=[]
    it = 0 
    converged = False
    if verbose:
        if logger is not None: logger.info("Beginning l=%f"%(lambd0))
        else: print("Beginning l=%f"%(lambd0))
    B = pi_prev
    inc = 0
    inc_rank = 0
    old_val = 1e18
    while not converged:
        g_t =  (K.todense().dot(B) - K.todense())
        #B=  project_DS2(B - g_t)#+np.abs(B - g_t))
        Z, time_taken, delta_x, _, _, dual, val = hcc_FISTA_denoise(K,
                                                                   pi_prev - 2.0/L * g_t,
                                                                   pi_prev,
                                                                   2.0/L*lambd0,
                                                                   alpha=alpha, 
                                                                   maxiterFISTA=maxiterFISTA,
                                                                   eta=1.0,
                                                                   tol=tol, 
                                                                   verbose=True,
                                                                   tol_projection=TOL_PROJ,
                                                                   logger=logger)
        pi_prev = Z
        if old_val < val:
            pi_prev = pi_prev_old
        else:
            old_val = val
        t_kp1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k**2))
        delta_pi.append(np.linalg.norm( pi_prev_old-pi_prev, 'fro')/np.linalg.norm( pi_prev_old, 'fro'))
        if delta_pi[-1]< tol:
            inc+=1
        else:
            inc=0
        if np.abs( efficient_rank(pi_prev_old)-efficient_rank(pi_prev))<2.0:
            inc_rank += 1
        else:
            inc_rank = 0
           
        #print delta_pi[-1]
        converged = (inc>2) or (it > maxiterFISTA)
        evol_efficient_rank += [efficient_rank(pi_prev)]
        
        B = pi_prev + (t_k)/t_kp1*(Z - pi_prev)+ (t_k-1)/t_kp1 * (pi_prev - pi_prev_old)
        pi_prev_old = pi_prev
        t_k = t_kp1
        it+=1
        if verbose:
            if logger is not None: logger.info("it:%i, convergence:%f, rk: %f)"%(it, delta_pi[-1],evol_efficient_rank[-1]))
            else: print(it, delta_pi[-1],evol_efficient_rank[-1])
        #if it ==1 : STOP

    print('-----------------------------------')
    if logger is not None: 
        logger.info("**********************************")
        logger.info("**********************************")
        logger.info("**********************************")
    toc = time.time()
    
    return pi_prev, toc-tic, evol_efficient_rank, delta_pi





