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

sys.path.append('/scratch/users/cdonnat/convex_clustering/HC_dev')
from convex_hc_denoising import *
from convex_hc_ADMM import *
from projections import *
from utils import *
from utils_graphs import *

sys.stdout = sys.__stdout__ 
random.seed(2018)


if __name__ == '__main__':
    parser = ArgumentParser("Run evaluation on MNIST dataset.")
    parser.add_argument("-path2data","--path2data", help="path2data", default='/scratch/users/cdonnat/HC_data')
    parser.add_argument("-path2logs","--path2logs", help="path2logs", default='/scratch/users/cdonnat/convex_clustering/experiments/logs/')
    parser.add_argument("-logger","--loggerfile", help="logger file name", default='log_digits.log')
    parser.add_argument("-savefile","--savefile", help="save file name", default='digits_new.pkl')
    parser.add_argument("-a","--alpha", help="alpha", default=0.95, type=float)
    parser.add_argument("-a_reg","--alpha_reg", help="regularization for the similarity matrix", default=0.1, type=float)
    parser.add_argument("-type_lap","--type_lap", help="Which laplacian to use?", default="normalized_laplacian", type=str)   
    parser.add_argument("-s","--sigma",help="bandwith for kernel",default=200.0, type=float)
    parser.add_argument("-l0","--lambd0",help="lambda 0 ",default=1e-3, type=float)
    parser.add_argument("-tol","--tol",help="tolerance for stopping criterion",default=5*1e-3, type=float)
    parser.add_argument("-nn","--n_neighbors",help="nb nearest_neighbors",default=10, type=int)
    parser.add_argument("-max_iter_fista","--max_iter_fista",help="max_iter_fista",default=150, type=int)
    args = parser.parse_args()

    logger = logging.getLogger('myapp')
    fh = logging.FileHandler(args.loggerfile)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh) 
    logger.setLevel(logging.DEBUG) # or any level you want

    ALPHA = args.alpha
    ALPHA_REG = args.alpha_reg
    N_NEIGHBORS = args.n_neighbors
    LAMBDA0 = args.lambd0
    MAXITERFISTA = args.max_iter_fista
    PATH2DATA = args.path2data
    PATH2LOGS = args.path2logs
    SAVEFILE = PATH2LOGS + '/digits_alpha_' + str(ALPHA) + args.savefile
    LOGGER_FILE = PATH2LOGS +  '/digits_alpha_' + str(ALPHA) + args.loggerfile
    SIGMA = args.sigma
    TOL = args.tol
    TYPE_LAP = args.type_lap

    from sklearn.datasets import load_digits
    digits = load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    D = np.exp(-cdist(data, data)**2/(2*SIGMA))
    nn = np.zeros(D.shape)
    for i in range(D.shape[0]):
        nn_n = [u for u in np.argsort(D[i,:])[-(N_NEIGHBORS+1):] if u!=i]
        nn[i, nn_n] = 1
    nn = nn +nn.T
    nn[nn>1.0] = 1.0
    K = D * nn
    K = create_similarity_matrix(K, TYPE_LAP, ALPHA_REG)
    n_nodes = K.shape[0]
    
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
    lambd = 2 * lambd0/ L
    maxiterFISTA = 120
    #L = 2*np.max(np.square(K.todense()).sum(1))
    res = {}
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
        old_val = 1e18
        while not converged:
            #STOP
            g_t = 2.0 / (L) * (K.todense().dot(B) - K.todense())
            B=  project_DS2(B - g_t)#+np.abs(B - g_t))
            Z, time_taken, delta_x, delta_p, delta_q, dual, val = hcc_FISTA_denoise(K, B,
                                                                               pi_prev,
                                                                               lambd,
                                                                               alpha=ALPHA, 
                                                                               maxiterFISTA=MAXITERFISTA,
                                                                               eta=eta_t,
                                                                               tol=TOL, 
                                                                               verbose=True,
                                                                               tol_projection=1e-2*TOL,
                                                                               logger=logger)
    
            pi_prev = Z
            if  old_val < val:
                pi_prev = pi_prev_old
            else:
                old_val = val

            conv_p[lambd0][it] = delta_p
            conv_q[lambd0][it] = delta_q
            conv_x[lambd0][it] = delta_x
            t_kp1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k**2))
            delta_pi.append(np.linalg.norm( pi_prev_old-pi_prev, 'fro')/np.linalg.norm( pi_prev_old, 'fro'))
            #print delta_pi[-1]
            if delta_pi[-1] < TOL:
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
            it +=1
            logger.info('outer loop %i: conv: %f, rank: %f'%(it, delta_pi[-1], evol_efficient_rank[lambd0][-1]))
            logger.info('--------------------------')

        logger.info('--------------------------')
        logger.info('--------------------------')
        logger.info('--------------------------')
        toc = time.time()
        res[lambd0]={'pi':pi_prev, 'convergence': delta_pi, 'time':toc-tic}
        pickle.dump(res, open(SAVEFILE, 'wb'))
        lambd0 *= 2
    logger.info('**********************************')
    logger.info('DONE')
