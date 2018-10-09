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
LAMBDA0 = 1e-2

def compute_reg_path(kernel, alpha, pi_warm_start, mode="ADMM", direction='up', tol= TOL,lambd0 = LAMBDA0,
                     lambda_max= LAMBDA_MAX, n_lambda=N_LAMBDA,
                     verbose= False, savefile =None, logger=None, **kwargs):
    ''' Computes the regularization path for K
    
        INPUT:
        -----------------------------------------------------------
        kernel            :      similarity matrix
        
        OUTPUT:
        -----------------------------------------------------------
        
    '''
    n_nodes = kernel.shape[0]
    evol_efficient_rank={}
    pi = {}
    x_init = pi_warm_start
    lambd = lambd0
    for it_lambda in range(n_lambda):
        tic = time.time()
        if logger is not None: 
            logger.info('Starting lambda = %f'%lambd)
        else:
            print('Starting lambda = %f'%lambd)
        if mode == 'ADMM':
              x_k, _, _, _, _, _ = hcc_ADMM(kernel, x_init, lambd,
                                                             alpha=alpha,
                                                             maxit_ADMM=MAXIT_ADMM,
                                                             tol=TOL,verbose=True,
                                                             maxiter_ux=MAXITER_UX,
                                                             logger=logger)
        else:
              x_k, _, _ , _ = hcc_FISTA(kernel, x_init, lambd,
                                        alpha=alpha, 
                                        maxiterFISTA=MAXIT_FISTA,
                                        tol = TOL,verbose=True,
                                        logger=logger)
        pi[lambd] = x_k
        x_init = x_k
        evol_efficient_rank[lambd]  = efficient_rank(x_k)
        print('--------------------------')
        print('--------------------------')
        print('--------------------------')
        toc = time.time()
        pi[lambd0]={'pi':x_k, 'time':toc-tic}
        # Check divergence compare to previous

        if verbose:
             print('finished lambda = %f'%lambd)
        if logger is not None: 
            logger.info('finished lambda = %f'%lambd)
            logger.info('--------------------------------')
        if savefile is not None:
            pickle.dump(pi,open(savefile,'wb'))
        if direction == 'down': 
            lambd -= lambd0
        else:
            lambd += lambd0
            

    return pi, toc - tic, evol_efficient_rank



