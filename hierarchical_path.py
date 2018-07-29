import numpy as np
import scipy as sc
from projections import *
from utils import *
from convex_hc_ADMM import *
from convex_hc_denoising import *
import time
import pickle


MAXIT_ADMM = 500
MAXIT_FISTA = 500
TOL= 5*1e-3
MAXITER_UX = 100
LAMBDA_MAX = 1e-3
N_LAMBDA = 20

def compute_reg_path(kernel, alpha, mode="ADMM", direction='up', tol= TOL,
                     lambda_max= LAMBDA_MAX, n_lambda=N_LAMBDA,
                     verbose= False, savefile =None, logger=None, **kwargs):
    ''' Computes the regularization path for K
    
        INPUT:
        -----------------------------------------------------------
        kernel            :      similarity matrix
        
        OUTPUT:
        -----------------------------------------------------------
        
    '''
    n_nodes, _ = kernel.shape
    eps = tol / np.sqrt(n_nodes)
    if direction == 'down':
        pi = {np.inf: 1.0 / n_nodes * np.ones((n_nodes, n_nodes))}
        x_init = pi[np.inf]
    else:
        pi = {0: np.eye(n_nodes)}
        x_init = pi[0]
    lambd = lambda_max
    time_alg = {}
    evol_rank = {}
    for it_lambda in range(n_lambda):
        if logger is not None: 
            logger.info('Starting lambda = %f'%lambd)
        else:
            print('Starting lambda = %f'%lambd)
        if mode == 'ADMM':
              x_k, time_alg[it_lambda],_, _, _, _ = hcc_ADMM(kernel, x_init, lambd,
                                                             alpha=alpha,
                                                             maxit_ADMM=MAXIT_ADMM,
                                                             tol=TOL,verbose=True,
                                                             maxiter_ux=MAXITER_UX,
                                                             logger=logger)
        else:
              x_k, time_alg[it_lambda], evol_rank[lambd], _ = hcc_FISTA(kernel, x_init, lambd,
                                                                        alpha=alpha, 
                                                                        maxiterFISTA=MAXIT_FISTA,
                                                                        tol = TOL,verbose=True,
                                                                        logger=logger)
        pi[lambd] = x_k
        x_init = x_k
        # Check divergence compare to previous
        if direction == 'down': 
            lambd *= 0.5
        else:
            lambd *= 2.0
        if verbose:
             print('finished lambda = %f'%lambd)
        if logger is not None: 
            logger.info('finished lambda = %f'%lambd)
            logger.info('--------------------------------')
        if savefile is not None:
            pickle.dump(pi,open(savefile,'wb'))
            

    return pi, time_alg, evol_rank



