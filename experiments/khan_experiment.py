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
from utils_graphs import *

sys.stdout = sys.__stdout__ 
random.seed(2018)


if __name__ == '__main__':
    parser = ArgumentParser("Run evaluation on KHAN dataset.")
    parser.add_argument("-path2data","--path2data", help="path2data", default='/scratch/users/cdonnat/data/HC_data')
    parser.add_argument("-path2data","--path2logs", help="path2logs", default='/scratch/users/cdonnat/convex_clustering/experiments/logs/')
    parser.add_argument("-logger","--loggerfile", help="logger file name", default='khan.log')
    parser.add_argument("-savefile","--savefile", help="save file name", default='khan.pkl')
    parser.add_argument("-a","--alpha", help="alpha", default=0.95, type=float)
    parser.add_argument("-a_reg","--alpha_reg", help="regularization for the similarity matrix", default=0.1, type=float)
    parser.add_argument("-type_lap","--type_lap", help="Which laplacian to use?", default="normalized_laplacian", type=str)
    parser.add_argument("-s","--sigma", help="bandwith for kernel", default=200.0, type=float)
    parser.add_argument("-l0","--lambd0", help="lambda 0 ", default=1e-3, type=float)
    parser.add_argument("-tol","--tol", help="tolerance for stopping criterion", default=5*1e-3, type=float)
    parser.add_argument("-nn","--n_neighbors", help="nb nearest_neighbors", default=10, type=int)
    parser.add_argument("-t","--is_train", help="use the training set(1) or test set (0)?", default=1, type=int)
    parser.add_argument("-max_iter_fista", "--max_iter_fista", help="max_iter_fista", default=300, type=int)
    args = parser.parse_args()


    ALPHA = args.alpha
    ALPHA_REG = args.alpha_reg
    N_NEIGHBORS = args.n_neighbors
    LAMBDA0 = args.lambd0
    MAXITERFISTA = args.max_iter_fista
    PATH2DATA = args.path2data
    PATH2LOGS = args.path2logs
    SAVEFILE = args.savefile
    SIGMA = args.sigma
    TOL = args.tol
    TYPE_LAP = args.type_lap
    USE_TRAINING_SET = args.is_train    
    
    if USE_TRAINING_SET == 1:
        data = pd.DataFrame.from_csv(PATH2DATA + "/khan_train.csv")
        SAVEFILE = PATH2LOGS + '/train_alpha_' + str(ALPHA) + args.savefile
        LOGGER_FILE = PATH2LOGS +  '/train_alpha_' + str(ALPHA) + args.loggerfile
    else:
        data = pd.DataFrame.from_csv(PATH2DATA +  "/khan_test.csv")
        SAVEFILE = PATH2LOGS + '/test_alpha_' + str(ALPHA) + args.savefile
        LOGGER_FILE = PATH2LOGS + '/test_alpha_' + str(ALPHA) + args.loggerfile
    
    logger = logging.getLogger('myapp')
    fh = logging.FileHandler(LOGGER_FILE)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh) 
    logger.setLevel(logging.DEBUG) # or any level you want
    
    
    D = np.exp(-cdist(data, data)**2 / (2 * SIGMA))
    nn = np.zeros(D.shape)
    for i in range(D.shape[0]):
        nn_n = [u for u in np.argsort(D[i, :])[-(N_NEIGHBORS + 1):] if u!= 1]
        nn[i, nn_n] = 1
    nn = nn + nn.T
    nn[nn > 1.0] = 1.0
    K = D * nn
    K = create_similarity_matrix(K, TYPE_LAP, ALPHA_REG)



    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
      

    n_nodes = K.shape[0]
    Y, pi, pi_prev = [np.eye(n_nodes)] * 3
    evol_efficient_rank={}
    L = 2 * sc.sparse.linalg.norm(K, 'fro')
    res = {}
    t_k = 1
    conv_p, conv_q, conv_x = {}, {} , {}
    value_taken = {}

    value_taken = {}
    for lambd0 in np.sort([0.001,0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
                           0.08, 0.09, 0.1, 0.13, 0.15, 0.17, 0.2,0.22, 0.24, 
                           0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 
                           0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.6, 0.65, 
                           0.7, 0.8 1.0, 2.0, 3.0, 3.5, 4, 5, 6, 7, 8, 9, 10,
                           15, 20, 30, 40, 60, 100, 200, 500, 1000, 1e4]):

        tic = time.time()

        pi_prev_old=pi_prev
        delta_pi=[]
    
        it = 0 
        converged =False
        logger.info("Beginning l=%f"%(lambd0))
        B = pi_prev
        conv_p[lambd0], conv_q[lambd0], conv_x[lambd0] = {}, {} , {}
        evol_efficient_rank[lambd0] = []
        m_tm1, nu_tm1 = np.zeros(B.shape),np.zeros(B.shape)
        eta_t = 1.0
        inc = 0
        inc_rank = 0
        step_size = 2.0/L
        value_taken[lambd0] = [1e10]
        while not converged:
            #STOP
            g_t =  (K.todense().dot(B) - K.todense())
            Z, time_taken, delta_x, delta_p, delta_q, dual, val = hcc_FISTA_denoise(K, B - step_size * g_t,
                                                                               pi_prev,
                                                                               step_size*lambd0,
                                                                               alpha=ALPHA, 
                                                                               maxiterFISTA=MAXITERFISTA,
                                                                               eta=eta_t,
                                                                               tol=TOL, 
                                                                               verbose=True,
                                                                               tol_projection=1e-2*TOL,
                                                                               logger=logger)
            pi_prev = Z
            if value_taken[lambd0][-1] < val:
                pi_prev = pi_prev_old
            else:
                old_val = val

            conv_p[lambd0][it] = delta_p
            conv_q[lambd0][it] = delta_q
            conv_x[lambd0][it] = delta_x
            t_kp1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k**2))
            delta_pi.append(np.sum(np.abs(pi_prev_old-pi_prev))/n_nodes)
            #print delta_pi[-1]
            if delta_pi[-1] < TOL:
                inc += 1
            else:
                inc = 0
            converged = (inc >= 5) or (it > n2)
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

