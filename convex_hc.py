"""
Code for the dual FISTA algorithm for hierarchical
convex clustering
"""

import numpy as np
import scipy as sc
from projections import *
import time


def hcc_FISTA(K, pi_prev, lambd, alpha=0.5, maxiterFISTA=100,
              tol=1e-2, verbose=True):
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
    eps = tol * 1.0 / n_nodes * np.min([1, 1.0/lambd])
    eps2 = tol 
    mask = np.array(K.todense()).flatten().reshape([-1,]).nonzero()[0]
    l_max = np.sqrt(np.max(np.square(K).sum(0)) )

    delta_k=sc.sparse.lil_matrix((n_nodes, n_nodes**2))
    for ii in range(n_nodes):
        ind =K[ii, :].nonzero()[1]
        #print ind
        delta_k[ii, ii * n_nodes + ind] = K[ii, ind]
        delta_k[ii, ii + ind * n_nodes] = -K[ii, ind]
        delta_k[ii, ii + ii * n_nodes] = 0.0
    delta_k = delta_k.tocsc()

    gamma = np.sqrt(8) * l_max * lambd
    if verbose: print("gamma =%f"%gamma)
    I = sc.sparse.eye(n_nodes)
    p = sc.sparse.csc_matrix((n_nodes, n_nodes**2))
    q = sc.sparse.csc_matrix((n_nodes, n_nodes**2))

    # Initialize the dual variables
    t=project_unit_ball(sc.sparse.csc_matrix(x_k).dot(delta_k[:, mask]))
    p[:, mask].data, q[:,mask].data = t.data, t.data
    
    p_old = sc.sparse.csc_matrix((n_nodes, n_nodes**2))
    q_old = sc.sparse.csc_matrix((n_nodes, n_nodes**2))
    t_k, it = 1, 1
    converged = False

    tic0 = time.time()
    delta_x = []
    delta_p = []
    delta_q = []
    dual = []
    it = 0

    while not converged:
        if verbose: print(it)
        belly = (alpha * p[:, mask]+ (1-alpha) * q[:, mask]).dot((delta_k[:, mask]).T)
        if verbose:    print("belly ", belly.max())

        inside = x_k - 2.0 / gamma*((np.array(K.todense()).dot(x_k) - K)) - 2* lambd /gamma * belly.A
        x_k = project_DS2(np.array(inside))

        t = project_unit_ball(p[:, mask].A\
                              + 1 * alpha / gamma * x_k.dot((delta_k[:, mask]).A),
                              is_sparse=False)

        p = p.tolil()
        p[:, mask]= sc.sparse.lil_matrix(t)
        p = p.tocsc()

        t = project_unit_cube(q[:, mask].A\
                              + 1 * (1-alpha) / gamma * x_k.dot((delta_k[:, mask]).A),
                              is_sparse=False)

        q = q.tolil()
        q[:, mask] = sc.sparse.lil_matrix(t)
        q = q.tocsc()
        if verbose: print q.max(), q.min(), p.max(), p.min()

        #### Check convergence
        delta_x.append(np.linalg.norm( x_k - x_km1, 'fro'))
        delta_p.append(sc.sparse.linalg.norm(p - p_old, 'fro'))
        delta_q.append(sc.sparse.linalg.norm(q - q_old, 'fro'))
        converged = (delta_x[-1] < tol / np.sqrt(n_nodes)
                     and it >10)\
                     or (it > maxiterFISTA)
        if verbose: print("norm",sc.sparse.linalg.norm(p-p_old,'fro'),sc.sparse.linalg.norm(q-q_old,'fro'),eps)
        if verbose: print("norm X", np.linalg.norm(x_k-x_km1, 'fro'))

        dual.append(sc.sparse.linalg.norm(alpha * p[:, mask].dot((delta_k[:, mask]).T)\
                    + (1 - alpha) * q[:, mask].dot((delta_k[:, mask]).T), 'fro'))
        t_kp1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k**2))

        p = p + (t_k - 1) / t_kp1 * (p - p_old)
        q = q + (t_k - 1) / t_kp1 * (q - q_old)
        t_k = t_kp1
        x_km1 = x_k
        p_old, q_old = copy.deepcopy(p), copy.deepcopy(q)
        it += 1

    toc0 = time.time()
    if verbose: print(time.time() - tic0)
    belly = alpha * p[:, mask].dot((delta_k[:, mask]).T)\
                    + (1-alpha) * q[:, mask].dot((delta_k[:, mask]).T)

    inside = x_k - 2.0 / gamma * (K.dot(x_k) - K)- 2* lambd /gamma*belly
    x_k = project_DS2(np.array(inside))

    return x_k, toc0-tic0, delta_x, delta_p, delta_q, dual


def hcc_FISTA_scalable(K, W_prev, E_prev, lambd, nb_clusters, alpha=0.5,
                       maxiterFISTA=100, tol=1e-2, verbose=True):
    ''' Hierarchical clustering algorithm based on FISTA (dual)
    This is the matrix factorization based version, allowing the algorithm
    to scale up and handle potentially bigger matrices and showing the full
    regularization path
    Input: similarity matrix K assumed to be from a Mercer kernel (or at least PSD)
    Output: the regularized soft membership assignment matrix
    --------------------------------------------------------------------------
    
    INPUT:
    -----------------------------------------------------------
    K            :      the similarity matrix
    W_prev       :      initialization value for W (warm start)
    E_prev       :      initialization value for E (warm start)
    lambd        :      the level of regularization desired
    nb_clusters  :      maximum number of clusters expected
    alpha        :      parameter for the total variation mixed penalty
    maxiter      :      max number of iterations (updates in E and W)
    tol          :      tolerance level for the stopping criterion,
                        as a fraction of the number of nodes
    verbose      :      boolean: allow printing of various statistics
                        and intermediary parameters

    OUTPUT:
    -----------------------------------------------------------
    X           :      the corresponding soft clustering assignment matrix
    t           :      the time that the procedure took
    delta_W     :      list of updates in W (for each iteration)
    delta_E     :      list of updates in E (for each iteration)
    
    --------------------------------------------------------------------------
    '''
    # Initialization
   
    E, E_old = E_prev, E_prev
    W, W_old = W_prev, W_prev
    n_nodes, _ = pi_prev.shape
    eps = tol * 1.0 / n_nodes * np.min([1, 1.0/lambd])
    eps2 = tol 
    mask = np.array(K.todense()).flatten().reshape([-1,]).nonzero()[0]
    l_max = np.sqrt(np.max(np.square(K).sum(0)) )

    delta_k=sc.sparse.lil_matrix((n_nodes, n_nodes**2))
    for ii in range(n_nodes):
        ind =K[ii, :].nonzero()[1]
        #print ind
        delta_k[ii, ii * n_nodes + ind] = K[ii, ind]
        delta_k[ii, ii + ind * n_nodes] = -K[ii, ind]
        delta_k[ii, ii + ii * n_nodes] = 0.0
    delta_k = delta_k.tocsc()


    tic0 = time.time()
    delta_E = []
    delta_W = []
    dual = []
    it = 0

    while not converged:
            if verbose: print(it)
            W = update_W(W, E, K)
            E = update_E(W, E, K)
            delta_E.append(np.linalg.norm(E-E_old))
            delta_W.append(np.linalg.norm(W-W_old))
            converged = (it > maxiterFISTA)
    return x_k, toc0-tic0, delta_W, delta_E


def update_W(W, E, K):
    ''' Updates for the problem involving the W matrix
    This is simply a version of the Fast Projected Gradient Descent algorithm
    Input: similarity matrix K assumed to be from a Mercer kernel (or at least PSD)
    Output: the regularized soft membership assignment matrix
    --------------------------------------------------------------------------
    
    INPUT:
    -----------------------------------------------------------
    K            :      the similarity matrix
    W_prev       :      initialization value for W (warm start)
    E            :      value for E
    lambd        :      the level of regularization desired
    nb_clusters  :      maximum number of clusters expected
    alpha        :      parameter for the total variation mixed penalty
    maxiter      :      max number of iterations (updates in E and W)
    tol          :      tolerance level for the stopping criterion,
                        as a fraction of the number of nodes
    verbose      :      boolean: allow printing of various statistics
                        and intermediary parameters

    OUTPUT:
    -----------------------------------------------------------
    X           :      the corresponding soft clustering assignment matrix
    t           :      the time that the procedure took
    delta_W     :      list of updates in W (for each iteration)
    delta_E     :      list of updates in E (for each iteration)
    
    --------------------------------------------------------------------------
    '''
    x_k, x_km1, y_k = W_prev, W_prev, W_prev
    n_nodes, n_clusters = W_prev.shape
    eps = tol * 1.0 / n_nodes * np.min([1, 1.0/lambd])
    eps2 = tol 
    mask = np.array(K.todense()).flatten().reshape([-1,]).nonzero()[0]
    l_max = np.sqrt(np.max(np.square(K).sum(0)) )

    delta_k=sc.sparse.lil_matrix((n_nodes, n_nodes**2))
    for ii in range(n_nodes):
        ind =K[ii, :].nonzero()[1]
        #print ind
        delta_k[ii, ii * n_nodes + ind] = K[ii, ind]
        delta_k[ii, ii + ind * n_nodes] = -K[ii, ind]
        delta_k[ii, ii + ii * n_nodes] = 0.0
    delta_k = delta_k.tocsc()

    # Change this gamma
    gamma = np.sqrt(8) * l_max * lambd
    if verbose: print("gamma =%f"%gamma)
    t_k, it = 1, 1
    converged = False

    tic0 = time.time()
    delta_x = []
    dual = []
    it = 0

    while not converged:
            if verbose: print(it)
            belly = (alpha * p[:, mask]+ (1-alpha) * q[:, mask]).dot((delta_k[:, mask]).T)
            if verbose:    print("belly ", belly.max())

            if mode=="kernelized":
                inside = x_k - 2.0 / gamma*((np.array(K.todense()).dot(x_k) - K)) - 2* lambd /gamma * belly.A

                x_k = project_DS2(np.array(inside))
            else:
                inside =  K - lambd * belly
                x_k = project_DS_symmetric(np.array(inside))
            t = project_unit_ball(p[:, mask].A\
                                  + 1 * alpha / gamma * x_k.dot((delta_k[:, mask]).A),
                                  is_sparse=False)

            p = p.tolil()
            p[:, mask]= sc.sparse.lil_matrix(t)
            p = p.tocsc()

            t = project_unit_cube(q[:, mask].A\
                                  + 1 * (1-alpha) / gamma * x_k.dot((delta_k[:, mask]).A),
                                  is_sparse=False)

            q = q.tolil()
            q[:, mask] = sc.sparse.lil_matrix(t)
            q = q.tocsc()
            if verbose: print q.max(), q.min(), p.max(), p.min()

            #### Check convergence
            delta_x.append(np.linalg.norm( x_k - x_km1, 'fro'))
            converged = (delta_x[-1] < tol / np.sqrt(n_nodes)
                         and it >10)\
                         or (it > maxiterFISTA)
            if verbose: print("norm",sc.sparse.linalg.norm(p-p_old,'fro'),sc.sparse.linalg.norm(q-q_old,'fro'),eps)
            if verbose: print("norm X", np.linalg.norm(x_k-x_km1, 'fro'))

            t_kp1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k**2))

            y = x_k + (t_k - 1) / t_kp1 * (x_k - x_km1)
            t_k = t_kp1
            x_km1 = x_k
            it += 1

    toc0=time.time()
    print(time.time()-tic0)
    belly = alpha * p[:, mask].dot((delta_k[:, mask]).T)\
                    + (1-alpha) * q[:, mask].dot((delta_k[:, mask]).T)

    if mode=="kernelized":
        inside = x_k - 2.0 / gamma * (K.dot(x_k) - K)- 2* lambd /gamma*belly
        x_k = project_DS2(np.array(inside))

    else:
        inside = K - lambd * belly
        x_k = project_DS_symmetric(np.array(inside))
    return x_k


def update_E(W, E, K_orig, lambd, alpha=0.5, maxiterFISTA=100,
             tol=1e-2, verbose=True):
    ''' Updates for the problem involving the W matrix
    Input: similarity matrix K assumed to be from a Mercer kernel (or at least PSD)
    Output: the regularized soft membership assignment matrix
    --------------------------------------------------------------------------
    
    INPUT:
    -----------------------------------------------------------
    K_orig       :      the similarity matrix
    W            :      value for W
    E_prev       :      initialization value for E (warm start)
    lambd        :      the level of regularization desired
    alpha        :      parameter for the total variation mixed penalty
    maxiter      :      max number of iterations (updates in E and W)
    tol          :      tolerance level for the stopping criterion,
                        as a fraction of the number of nodes
    verbose      :      boolean: allow printing of various statistics
                        and intermediary parameters

    OUTPUT:
    -----------------------------------------------------------
    X           :      the corresponding soft clustering assignment matrix
    t           :      the time that the procedure took
    delta_W     :      list of updates in W (for each iteration)
    delta_E     :      list of updates in E (for each iteration)
    
    --------------------------------------------------------------------------
    '''
    # Initialization
    x_k, x_km1, y_k = E_prev, E_prev, E_prev
    n_clusters, n_nodes = E_prev.shape
    eps = tol * 1.0 / n_nodes * np.min([1, 1.0/lambd])
    eps2 = tol
    K = W.T.dot(K_orig.dot(W)) # we are essentially solving the same problem
                               # as in hcc_alg, with a modified kernel
    mask = np.array(K.todense()).flatten().reshape([-1,]).nonzero()[0]
    l_max = np.sqrt(np.max(np.square(K).sum(0)) )

    delta_k=sc.sparse.lil_matrix((n_nodes, n_nodes**2))
    for ii in range(n_nodes):
        ind =K[ii, :].nonzero()[1]
        #print ind
        delta_k[ii, ii * n_nodes + ind] = K[ii, ind]
        delta_k[ii, ii + ind * n_nodes] = -K[ii, ind]
        delta_k[ii, ii + ii * n_nodes] = 0.0
    delta_k = delta_k.tocsc()

    #### change this gamma constant 
    gamma = np.sqrt(8) * l_max * lambd
    if verbose: print("gamma =%f"%gamma)
    I = sc.sparse.eye(n_nodes)
    p = sc.sparse.csc_matrix((n_nodes, n_nodes**2))
    q = sc.sparse.csc_matrix((n_nodes, n_nodes**2))

    # Initialize the dual variables
    t=project_unit_ball(sc.sparse.csc_matrix(x_k).dot(delta_k[:, mask]))
    p[:, mask].data, q[:,mask].data = t.data, t.data
    
    p_old = sc.sparse.csc_matrix((n_nodes, n_nodes**2))
    q_old = sc.sparse.csc_matrix((n_nodes, n_nodes**2))
    t_k, it = 1, 1
    converged = False

    tic0 = time.time()
    delta_x = []
    delta_p = []
    delta_q = []
    dual = []
    it = 0

    while not converged:
            if verbose: print(it)
            belly = (alpha * p[:, mask]+ (1-alpha) * q[:, mask]).dot((delta_k[:, mask]).T)
            if verbose:    print("belly ", belly.max())

            inside = x_k - 2.0 / gamma*((np.array(K.todense()).dot(x_k) - K)) - 2* lambd /gamma * belly.A

            x_k = project_stochmat(np.array(inside), orient=1.0)
            t = project_unit_ball(p[:, mask].A\
                                  + 1 * alpha / gamma * x_k.dot((delta_k[:, mask]).A),
                                  is_sparse=False)

            p = p.tolil()
            p[:, mask]= sc.sparse.lil_matrix(t)
            p = p.tocsc()

            t = project_unit_cube(q[:, mask].A\
                                  + 1 * (1-alpha) / gamma * x_k.dot((delta_k[:, mask]).A),
                                  is_sparse=False)

            q = q.tolil()
            q[:, mask] = sc.sparse.lil_matrix(t)
            q = q.tocsc()
            if verbose: print q.max(), q.min(), p.max(), p.min()

            #### Check convergence
            delta_x.append(np.linalg.norm( x_k - x_km1, 'fro'))
            delta_p.append(sc.sparse.linalg.norm(p - p_old, 'fro'))
            delta_q.append(sc.sparse.linalg.norm(q - q_old, 'fro'))
            converged = (delta_x[-1] < tol / np.sqrt(n_nodes)
                         and it >10)\
                         or (it > maxiterFISTA)
            if verbose: print("norm",sc.sparse.linalg.norm(p-p_old,'fro'),sc.sparse.linalg.norm(q-q_old,'fro'),eps)
            if verbose: print("norm X", np.linalg.norm(x_k-x_km1, 'fro'))

            dual.append(sc.sparse.linalg.norm(alpha * p[:, mask].dot((delta_k[:, mask]).T)\
                        + (1 - alpha) * q[:, mask].dot((delta_k[:, mask]).T), 'fro'))
            t_kp1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k**2))

            p = p + (t_k - 1) / t_kp1 * (p - p_old)
            q = q + (t_k - 1) / t_kp1 * (q - q_old)
            t_k = t_kp1
            x_km1 = x_k
            p_old, q_old = copy.deepcopy(p), copy.deepcopy(q)
            it += 1

    toc0=time.time()
    belly = alpha * p[:, mask].dot((delta_k[:, mask]).T)\
                    + (1-alpha) * q[:, mask].dot((delta_k[:, mask]).T)

    inside = x_k - 2.0 / gamma * (K.dot(x_k) - K)- 2* lambd /gamma*belly
    x_k = project_stochmat(np.array(inside), orient=1.0)

    return x_k, toc0 -tic0, delta_x, delta_p, delta_q, dual

