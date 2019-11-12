from __future__ import print_function

from argparse import ArgumentParser
import copy
import logging
import numpy as np
import networkx as nx
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
    parser = ArgumentParser("Run evaluation on protein dataset.")
    parser.add_argument("-logger","--loggerfile", help="logger file name", default='log_protein.log')
    parser.add_argument("-savefile","--savefile", help="save file name", default='protein.pkl')
    parser.add_argument("-a","--alpha", help="alpha", default=0.95, type=float)
    parser.add_argument("-s","--sigma", help="bandwith for kernel", default=200.0, type=float)
    parser.add_argument("-l0","--lambd0", help="lambda 0 ", default=1e-3, type=float)
    parser.add_argument("-tol","--tol", help="tolerance for stopping criterion", default=5*1e-3, type=float)
    parser.add_argument("-nn","--n_neighbors", help="nb nearest_neighbors", default=10, type=int)
    parser.add_argument("-t","--is_train", help="use the training set(1) or test set (0)?", default=1, type=int)
    parser.add_argument("-max_iter_fista", "--max_iter_fista", help="max_iter_fista", default=150, type=int)
    args = parser.parse_args()

    SIGMA = args.sigma
    N_NEIGHBORS = args.n_neighbors
    LAMBDA0 = args.lambd0
    TOL = args.tol
    ALPHA = args.alpha
    MAXITERFISTA = args.max_iter_fista
    NAME_EXPERIMENT = 'protein_experiment_alpha_' + str(ALPHA)
    SAVEFILE = args.savefile
    LOG_FILE = args.loggerfile
    INPUTFILE = '/scratch/users/cdonnat/HC_data/protein_edges.csv'

    logger = logging.getLogger('myapp')
    fh = logging.FileHandler(LOG_FILE)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG) # or any level you want




    edgelist = pd.DataFrame.from_csv(INPUTFILE)
    G = nx.from_pandas_edgelist(edgelist, source ='source' , target ='target', edge_attr= ['weight'] ).to_undirected()
    K = nx.adjacency_matrix(G)
    K = K.T.dot(K)
    sqrtv = np.vectorize(lambda x: 1.0/np.sqrt(x) if x > 1e-10 else 0.0)
    Deg = np.diagflat(sqrtv(K.diagonal()))
    K = sc.sparse.csc_matrix(Deg.dot(K.dot(Deg)))
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
    #L = 2*np.max(np.square(K.todense()).sum(1))
    res = {}
    t_k = 1
    conv_p, conv_q, conv_x = {}, {} , {}
    value_taken = {}

    for l in range(20):
        tic = time.time()
        value_taken[lambd0]=[1e18]
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
            B=  project_DS2(B - g_t)#+np.abs(B - g_t)) #x_k, toc0-tic0, delta_x, delta_p, delta_q, dual, val
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
            if value_taken[lambd0][-1] < val:
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
                if np.abs(efficient_rank(Z)-evol_efficient_rank[lambd0][-1])<2:
                    inc_rank += 1
                else:
                    inc_rank = 0
            converged = (inc >= 5) or (inc_rank > 10 and it > 14) or (it > MAXITERFISTA2)
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
