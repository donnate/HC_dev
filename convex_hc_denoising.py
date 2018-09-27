"""
Code for the dual FISTA algorithm for hierarchical
convex clustering
"""
import copy
import numpy as np
import scipy as sc
import time

from projections import *
from utils import *

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
    if type(K) == np.matrixlib.defmatrix.matrix:
        mask = np.array(K).flatten().reshape([-1,]).nonzero()[0]
    else:
        indices = K.nonzero()
        mask = []
        for i in range(K.nnz):
            mask += [indices[0][i]*n_nodes+indices[1][i]]
        mask = np.array(mask)
    lmax = np.max(np.square(K).sum(0)) 

    delta_k=sc.sparse.lil_matrix((n_nodes, n_nodes**2))
    for ii in range(n_nodes):
        ind =K[ii, :].nonzero()[1]
        #print ind
        indices = np.arange(0,n_nodes**2,n_nodes)
        delta_k[ii, ii * n_nodes +ind ] = K[ii, ind]
        for e in ind:
            delta_k[e, ii * n_nodes +e ] = -K[ii, e]
        delta_k[ii, ii + ii * n_nodes] = 0.0
    delta_k = delta_k.tocsc()

    gamma = 8 * max([alpha**2,(1-alpha)**2])*lmax * lambd**2
    ##gamma = 16 * max([alpha**2,(1-alpha)**2])*l_max * lambd**2
    if verbose: print("gamma =%f"%gamma)
    if verbose: print("lmax",lmax)
    I = sc.sparse.eye(n_nodes)
    p = sc.sparse.csc_matrix((n_nodes, n_nodes**2))
    q = sc.sparse.csc_matrix((n_nodes, n_nodes**2))

    # Initialize the dual variables
    t = project_unit_ball(x_k.dot((delta_k[:, mask]).A), is_sparse=False)

    p = p.tolil()

    p[:, mask]= sc.sparse.lil_matrix(t)
    p = p.tocsc()
    q = sc.sparse.csc_matrix((n_nodes, n_nodes**2))
    t = project_unit_cube( x_k.dot((delta_k[:, mask]).A), is_sparse=False)

    q = q.tolil()
    q[:, mask] = sc.sparse.lil_matrix(t)
    q = q.tocsc()

    p_old = copy.deepcopy(p)
    q_old = copy.deepcopy(q)

    t_k, it = 1, 1
    converged = False

    tic0 = time.time()
    delta_x = []
    delta_p = []
    delta_q = []
    dual = []
    it = 0
    eps_reg =1e-5
    while not converged:
        belly = (alpha * p[:, mask]+ (1-alpha) * q[:, mask]).dot((delta_k[:, mask]).T)
        if verbose:    
            if logger is not None: logger.info("belly %f"%belly.max())
            else : print("belly ", belly.max())

        inside = B - eta * lambd * belly.A
        x_k = project_DS2(np.array(inside),  max_it=max_iter_projection, eps = tol_projection)
        
        t = project_unit_ball(p[:, mask].A\
                              + 1.0 * alpha / gamma * x_k.dot((delta_k[:, mask]).A),
                              is_sparse=False)

        p = p.tolil()

        p[:, mask]= sc.sparse.lil_matrix(t)
        p = p.tocsc()

        t = project_unit_cube(q[:, mask].A\
                              + 1.0 * (1-alpha) / gamma * x_k.dot((delta_k[:, mask]).A),
                              is_sparse=False)

        q = q.tolil()
        q[:, mask] = sc.sparse.lil_matrix(t)
        q = q.tocsc()
        if verbose: print(q.max(), q.min(), p.max(), p.min())

        #### Check convergence
        delta_x.append(np.linalg.norm( x_k - x_km1, 'fro')/np.linalg.norm(x_km1,'fro'))
        delta_p.append(sc.
                       
                       sparse.linalg.norm(p - p_old, 'fro'))
        delta_q.append(sc.sparse.linalg.norm(q - q_old, 'fro'))
        converged = (delta_x[-1] < tol and it>1)\
                     or (it > maxiterFISTA)
        if verbose: 
            if logger is not None:  logger.info("norms : %f and %f"%(sc.sparse.linalg.norm(p-p_old,'fro'),
                                          sc.sparse.linalg.norm(q-q_old,'fro')))
            else: print("norm",sc.sparse.linalg.norm(p-p_old,'fro'),
                                          sc.sparse.linalg.norm(q-q_old,'fro'))
        if verbose: 
            if logger is not None: logger.info("norm X : %f, efficient rank: %f"%(np.linalg.norm(x_k-x_km1, 'fro'),
                                                                                  efficient_rank(x_k)))
            else: print("norm X : %f, efficient rank: %f"%(np.linalg.norm(x_k-x_km1, 'fro'),
                                                                                  efficient_rank(x_k)))

        dual.append(sc.sparse.linalg.norm(alpha * p[:, mask].dot((delta_k[:, mask]).T)\
                    + (1 - alpha) * q[:, mask].dot((delta_k[:, mask]).T), 'fro'))
        t_kp1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k**2))

        p = p + (t_k - 1) / t_kp1 * (p - p_old)
        q = q + (t_k - 1) / t_kp1 * (q - q_old)
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
    if verbose: print(time.time() - tic0)
    belly = alpha * p[:, mask].dot((delta_k[:, mask]).T)\
                    + (1-alpha) * q[:, mask].dot((delta_k[:, mask]).T)

    
    inside = B - eta * lambd * belly.A
    x_k = project_DS2(np.array(inside),  max_it=max_iter_projection, eps = tol_projection)

    return x_k, toc0-tic0, delta_x, delta_p, delta_q, dual


def hcc_FISTA(K, pi_warm_start, lambd0, alpha =0.95,
              maxiterFISTA = 2000, tol=5*1e-3, debug_mode=False,
              lambda_spot = 0, verbose =False, logger=None):
    if debug_mode: verbose =True
    Y, pi_prev, pi_prev_old = [pi_warm_start] * 3
    evol_efficient_rank=[]
    conv_p, conv_q, conv_x = {}, {}, {}
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
    while not converged:
        #STOP
        g_t = 2.0 / L * (K.todense().dot(B) - K.todense())
        B =  project_DS2(B - g_t, eps = 1e-4)#+np.abs(B - g_t))
      
        
        Z, time_taken, delta_x, delta_p, delta_q, dual = hcc_FISTA_denoise(K , B, pi_prev, 
                                                                           lambd, alpha=alpha, 
                                                                           maxiterFISTA=100,
                                                                           eta=1.0,
                                                                           tol= tol , verbose=False,
                                                                            tol_projection= 1e-4)
        if it>1:
            if ((efficient_rank(Z) - evol_efficient_rank[-1])/evol_efficient_rank[-1] >0 and
                np.linalg.norm( pi_prev_old-Z, 'fro')/np.linalg.norm( pi_prev_old, 'fro')>0.5):
                pi_prev = pi_prev_old
            else:
                pi_prev = Z
        else:
            pi_prev = Z
        
        #if efficient_rank(pi_prev)<45: STOP
        conv_p[it] = delta_p
        conv_q[it] = delta_q
        conv_x[it] = delta_x
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
        converged = (inc>4)  or (inc_rank > 10 and it >15) or (it > maxiterFISTA)
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
    if logger is not None: logger.info("'-----------------------------------")
    toc = time.time()
    return pi_prev, toc-tic, evol_efficient_rank, delta_pi
