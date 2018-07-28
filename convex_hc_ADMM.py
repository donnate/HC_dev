import copy
import numpy as np
from projections import *
import scipy as sc
import time


def hcc_ADMM(K, pi_prev, lambd, alpha=0.5, maxit_ADMM=100, rho=1.0,
             mode='kernelized', tol=1e-2, maxiter_ux=100, verbose=False):
    ''' Hierarchical clustering algorithm based on ADMM
        Input: similarity matrix K assumed to be from a Mercer kernel
        (or at least PSD)
        Output: the regularized soft membership assignment matrix
        ----------------------------------------------------------------------

        INPUT:
        -----------------------------------------------------------
        K            :      the similarity matrix
        pi_prev      :      initialization value for pi (warm start)
        lambd        :      the level of regularization desired
        alpha        :      parameter for the elastic net penalty
        maxit_ADMM   :      max number of iterations for ADMM
        mode         :      which problem to solve (kernel or
                            projection version)
        tol          :      tolerance level for the algorithm
                            (stopping criterion), as a fraction of
                            the n_nodes
        maxiter_ux   :      maxiter_ux

        OUTPUT:
        -----------------------------------------------------------
        X            :      corresponding soft clustering assignment matrix
        t            :      time that the procedure took
        ----------------------------------------------------------------------
    '''
    def update_X(X, Z, U, K, rho, delta, maxiter_ux =100):
        
        n_nodes, _ =K.shape
        eps = 0.01 / n_nodes
        converged = False
        L = (sc.sparse.linalg.norm(K, 'fro') + 4 * rho * n_nodes)
        t_k = 1
        X_k , X_km1 = X, X
        Y_k = sc.sparse.csc_matrix(X)
        it = 0
        while not converged:
            grad = K.dot(Y_k) - K +\
                   rho * (Y_k.dot(delta_k) + U - Z).dot(delta_k.T)
            X_k = iteration_proj_DS((Y_k - 1.0 / L *grad).todense())
            t_kp1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k**2))
            Y_k = sc.sparse.csc_matrix(X_k + (t_k-1) / t_kp1 * (X_k - X_km1))
            it += 1
            converged = (it > maxiter_ux) \
                         or np.linalg.norm(X_k - X_km1,'fro') < eps
            X_km1 = X_k
        return iteration_proj_DS(Y_k.todense())

    def update_Z(X, Z, U, K, rho, delta, alpha, lambd):
        # Let's assume that we are dealing with sparse matrices
        norm_Z = np.linalg.norm(X.dot(delta) + U)
        L = 1.0 + (1.0 - alpha) * lambd / (rho)
        mask = delta.nonzero
        Z_temp = 1.0 / L * (X.dot(delta) + U)
        #th = alpha * lambd / (rho + (1.0 - alpha) * lambd)
        th = np.min([alpha * lambd / ((rho + (1.0 - alpha) * lambd)*norm_X),2])
        thres = np.vectorize(lambda x: x -th * np.sign(x) if np.abs(x) >  th else 0)
        Z_temp.data = thres(Z_temp.data)
        return Z_temp

    def update_U(X, Z, U, delta):
        return U + (X.dot(delta) - Z)


    tic = time.time()
    n_nodes, _ = K.shape
    converged = False
    primal_res, dual_res = [], []
    eps1 = tol / np.sqrt(n_nodes)
    eps2 = tol
    it = 0

    K_tilde = copy.deepcopy(K)
    X = sc.sparse.csc_matrix(pi_prev)
    X_prev = copy.deepcopy(X)
    binarize = np.vectorize(lambda x: 1 if x > 1e-5 else 0)
    K_tilde.data = binarize(K_tilde.data)
    K_tilde = K_tilde.tocsr()
    delta_k=sc.sparse.lil_matrix((n_nodes, n_nodes**2))
    for ii in range(n_nodes):
        ind = K_tilde[ii, :].nonzero()[1]
        delta_k[ii, ii * n_nodes + ind] = 1.0
        delta_k[ii, ii + ind * n_nodes] = -1.0
        delta_k[ii, ii + ii * n_nodes] = 0.0
    delta_k = delta_k.tocsc()
    Z = sc.sparse.csc_matrix((n_nodes, n_nodes**2))
    U = sc.sparse.csc_matrix((n_nodes, n_nodes**2))
    while not converged:
        X = update_X(X, Z, U, K, rho, delta_k, maxiter_ux = maxiter_ux)
        X = sc.sparse.csc_matrix(X)
        Z = update_Z(X, Z, U, K, rho, delta_k, alpha, lambd)
        U = update_U(X, Z, U, delta_k)

        primal_res.append(sc.sparse.linalg.norm(X -X_prev, 'fro'))
        dual_res.append(sc.sparse.linalg.norm(X.dot(delta_k) - Z, 'fro'))
        it += 1
        X_prev = X
        converged = (it > maxit_ADMM) or\
                    (primal_res[-1] < eps1 and dual_res[-1] < eps2)
        if verbose: 
            print "Primal res= %f, Dual_res= %f, at iteration %i"%(primal_res[-1],
                                                                   dual_res[-1],
                                                                   it)

    toc = time.time()
    return X, toc-tic, Z, U, primal_res, dual_res