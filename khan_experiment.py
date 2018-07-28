from __future__ import print_function

from argparse import ArgumentParser
import copy
import logging
import numpy as np
import os
import pandas as pd
import pickle
import random
from scipy.spatial.distance import cdist
import scipy as sc
import sklearn as sk
import sys
import time


from convex_hc_denoising import *
from convex_hc_ADMM import *
from projections import *
from utils import *

sys.stdout = sys.__stdout__ 
random.seed(2018)


if __name__ == '__main__':
    parser = ArgumentParser("Run evaluation on KHAN dataset.")
    parser.add_argument("-logger","--loggerfile",help="logger file name",default='synthetic_experiment_output/log_synthetic.log')
    parser.add_argument("-a","--alpha",help="alpha",default=0.95, type=float)
    parser.add_argument("-s","--sigma",help="bandwith for kernel",default=200.0, type=float)
    parser.add_argument("-l0","--lambd0",help="lambda 0 ",default=1e-3, type=float)
    parser.add_argument("-tol","--tol",help="tolerance for stopping criterion",default=5*1e-3, type=float)
    parser.add_argument("-nn","--n_neighbors",help="nb nearest_neighbors",default=10, type=int)
    args = parser.parse_args()

    logger = logging.getLogger('myapp')
    fh = logging.FileHandler(args.loggerfile)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh) 
    logger.setLevel(logging.DEBUG) # or any level you want

    ALPHA = args.alpha
    SIGMA = args.sigma
    N_NEIGHBORS = args.n_neighbors
    LAMBDA0 = args.lambd0
    TOL = args.tol

    data = pd.DataFrame.from_csv("data/khan_train.csv")
    D = np.exp(-cdist(data, data)**2/(2*SIGMA))
    nn = np.zeros(D.shape)
    for i in range(D.shape[0]):
        nn_n = [u for u in np.argsort(D[i,:])[-(N_NEIGHBORS+1):] if u!=1]
        nn[i, nn_n] = 1
    nn = nn +nn.T
    np.fill_diagonal(nn, 1)

    K = sc.sparse.csc_matrix(D * nn)
    n_nodes = K.shape[0]


    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
      

    n_nodes = K.shape[0]
    Y, pi, pi_prev = [np.eye(n_nodes)]*3
    evol_efficient_rank={}
    L = 2*sc.sparse.linalg.norm(K, 'fro')
    lambd0 = LAMBDA0
    lambd = 2*lambd0/ L
    maxiterFISTA = 120
    #L = 2*np.max(np.square(K.todense()).sum(1))
    res = {}
    tol= 5*1e-3
    t_k = 1
    conv_p, conv_q, conv_x = {}, {} , {}

    for l in range(20):
        tic = time.time()

        pi_prev_old=pi_prev
        delta_pi=[]
    
        it = 0 
        converged =False
        logger.info("Beginning l=%f"%(lambd0))
        lambd = 2*lambd0/ L
        B = pi_prev
        conv_p[lambd0], conv_q[lambd0], conv_x[lambd0] = {}, {} , {}
        evol_efficient_rank[lambd0] = []
        m_tm1, nu_tm1 = np.zeros(B.shape),np.zeros(B.shape)
        eta_t = 1.0
        inc  = 0
        inc_rank = 0
        while not converged:
            #STOP
            g_t = 2.0 / (L) * (K.todense().dot(B) - K.todense())
            B=  project_DS2(B - g_t)#+np.abs(B - g_t))
            Z, time_taken, delta_x, delta_p, delta_q, dual = hcc_FISTA_denoise(K, B,
                                                                               pi_prev,
                                                                               lambd,
                                                                               alpha=ALPHA, 
                                                                               maxiterFISTA=300,
                                                                               eta=eta_t,
                                                                               tol=TOL, 
                                                                               verbose=True,
                                                                               tol_projection=5*1e-5,
                                                                               logger=logger)
            pi_prev = Z
            if it > 2:
                if (np.linalg.norm( pi_prev_old-Z, 'fro')/np.linalg.norm( pi_prev_old, 'fro')>0.5
                   and efficient_rank(Z)>evol_efficient_rank[lambd0][-1]):
                    pi_prev =pi_prev_old

            conv_p[lambd0][it] = delta_p
            conv_q[lambd0][it] = delta_q
            conv_x[lambd0][it] = delta_x
            t_kp1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k**2))
            delta_pi.append(np.linalg.norm( pi_prev_old-pi_prev, 'fro')/np.linalg.norm( pi_prev_old, 'fro'))
            #print delta_pi[-1]
            if delta_pi[-1] < tol:
                inc += 1
            else:
                inc = 0
            if it > 0:
                if np.abs(efficient_rank(Z)-evol_efficient_rank[lambd0][-1])<0.5:
                    inc_rank += 1
                else:
                    inc_rank = 0
            converged = (inc >= 5) or (inc_rank > 20 and it > 50) or (it > maxiterFISTA)
            evol_efficient_rank[lambd0] += [efficient_rank(pi_prev)]
            B = pi_prev + (t_k) / t_kp1 * (Z - pi_prev)\
                + (t_k - 1) / t_kp1 * (pi_prev - pi_prev_old)
            pi_prev_old = pi_prev
            t_k = t_kp1
            it + =1
            logger.info('outer loop %i: conv: %f, rank: %f'%(it, delta_pi[-1], evol_efficient_rank[lambd0][-1]))
            logger.info('--------------------------')

        logger.info('--------------------------')
        logger.info('--------------------------')
        logger.info('--------------------------')
        toc = time.time()
        res[lambd0]={'pi':pi_prev, 'convergence': delta_pi, 'time':toc-tic}
        pickle.dump(res, open('KHAN_l1.pkl', 'wb'))
        lambd0 *= 2