import copy
import numpy as np
from projections import *
import scipy as sc
import time

MAXIT_ADMM = 100
MAXITER_UX = 100
ALPHA = 0.5
TOL = 1e-2
RHO = 1.0


def hcc_ADMM(K, pi_prev, lambd, alpha=ALPHA, maxit_ADMM=MAXIT_ADMM, rho=RHO,
             mode='kernelized', tol=TOL, maxiter_ux=MAXITER_UX, verbose=False, logger = None):
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
    def update_X(X, Z, U, K, rho, indices_x, indices_y, neighbors, neighbors2, maxiter_ux =100):
        
        n_nodes, _ =K.shape
        eps = 0.01 * n_nodes
        converged = False
        L = np.sqrt(np.sum(K**2) + 4 * rho**2 * n_nodes**3)
        t_k = 1
        X_k , X_km1 = X, X
        Y_k = np.abs(X - 1.0 / n_nodes * np.diag(np.ones(n_nodes)))
        it = 0
        
        while not converged:
            #Y_k2 = Y_k #- 1.0 / n_nodes * np.diag(np.ones(n_nodes)))
            Y_k2 = iteration_proj_DS(Y_k + 1.0 / n_nodes * np.diag(np.ones(n_nodes)))
            Y_k2_temp = Y_k2[:, indices_x] - Y_k2[:, indices_y] 
            print(U.shape, Z.shape, Y_k2_temp.shape)
            inside = np.vstack([np.einsum('ij->i',(Y_k2_temp  + U - Z)[:, neighbors[i]]) \
                                - np.einsum('ij->i',(Y_k2_temp  + U -Z)[:, neighbors2[i]])
                               for i in range(n_nodes)]).T
            
            grad = K.dot(Y_k2) - K +\
                   rho * (inside)
            X_k = (1.0 - 1.0 / n_nodes) * iteration_proj_DS(Y_k - 1.0 / L *grad)
            t_kp1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k**2))
            Y_k = X_k + (t_k-1) / t_kp1 * (X_k - X_km1)
            it += 1
            print(np.linalg.norm(X_k - X_km1,'fro'))
            converged = (it > maxiter_ux) \
                         or np.linalg.norm(X_k - X_km1,'fro') < eps
            X_km1 = X_k
        return iteration_proj_DS(Y_k + 1.0 / n_nodes * np.diag(np.ones(n_nodes)))


    
    def update_Z(X, Z, U, K, rho, indices_x, indices_y, alpha, lambd):
        # Let's assume that we are dealing with sparse matrices
        #norm_Z = sc.sparse.linalg.norm(X.dot(delta) + U)
        
        Z_prev = copy.deepcopy(Z)
        n_nodes, _ = K.shape
        it_max = 20
        it = 0
        tol = 0.01 * len(indices_x)
        invert = np.vectorize(lambda x: 1.0/x if x >0.001 else 0)
        not_converged = True
        eta = 0.01
        norm_Z = invert(np.sqrt(np.sum(np.einsum('ij,ij->ij',Z_prev, Z_prev),0)))
        L = np.ones(len(norm_Z)) + (1.0 - alpha) * lambd / (rho) * norm_Z
        th  = np.vstack([alpha * lambd  * invert((rho + (1.0 - alpha) * lambd * norm_Z))]*n_nodes)
        print( th.shape, U.shape)
        Z_temp = np.einsum("j,ij->ij",invert(L) , X[:,indices_x]- X[:, indices_y]  + U)
        sign_Z = np.sign(Z_temp)
        prune = np.vectorize(lambda x: x if x>0 else 0)
        Z = np.einsum('ij, ij->ij', sign_Z,(prune(np.abs(Z_temp) - th)))
        return Z

    def update_U(X, Z, U, K, indices_x, indices_y):
        return U + (X[:,indices_x]- X[:, indices_y] - Z)


    tic = time.time()
    n_nodes, _ = K.shape
    converged = False
    primal_res, dual_res = [], []
    eps1 = tol * np.sqrt(n_nodes)
    eps2 = tol * np.sqrt(n_nodes)
    it = 0
    
    indices_x, indices_y  = K.nonzero()
    neighbors = [None] * n_nodes
    neighbors2 = [None] * n_nodes
    for ii in range(n_nodes):
        ind = np.where(indices_x == ii)[0]
        neighbors[ii] = ind
        ind2 = np.where(indices_y == ii)[0]
        neighbors2[ii] = ind2
        
    X = pi_prev
    X_prev = copy.deepcopy(X)

    Z = np.einsum('j,ij->ij', np.array(K[indices_x, indices_y]).flatten(),
                  X[:,indices_x]- X[:, indices_y])
    Z_prev = copy.deepcopy(Z)
    U = np.zeros((n_nodes, len(indices_x)))
    #print(U.shape, Z.shape, X.shape)
    plt.figure()
    sb.heatmap(X)
    plt.title("X at iteration %i "%0)
    plt.show()
    while not converged:
        tic1 = time.time()
        X = update_X(X, Z, U, K, rho, indices_x, indices_y, neighbors, neighbors2, maxiter_ux = maxiter_ux)
        Z = update_Z(X, Z, U, K, rho, indices_x, indices_y, alpha, lambd)
        U = update_U(X, Z, U, K, indices_x, indices_y)

        primal_res.append(sc.linalg.norm(X - X_prev, 'fro'))
        dual_res.append(sc.linalg.norm(Z_prev - Z, 'fro'))
        it += 1
        X_prev = X
        Z_prev = Z
        converged = (it > maxit_ADMM) or\
                    (primal_res[-1] < eps1 and dual_res[-1] < eps2)
        toc1 = time.time()
        if verbose: 
            if logger is not None:
                logger.info("Primal res= %f, Dual_res= %f, at iteration %i  %.3f s"%(primal_res[-1],
                                                                   dual_res[-1],
                                                                   it, toc1 - tic1))
            else:
                print("Primal res= %f, Dual_res= %f, at iteration %i in %.3f s"%(primal_res[-1],
                                                                   dual_res[-1],
                                                                   it, toc1 - tic1)
                     )
                plt.figure()
                sb.heatmap(X)
                plt.title("X at iteration %i "%it)
                plt.show()

    toc = time.time()
    return X, toc-tic, Z, U, primal_res, dual_res
