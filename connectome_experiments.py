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
from scipy.stats import scoreatpercentile
import scipy as sc
import sklearn as sk
import sys
import time


from convex_hc_denoising import *
from convex_hc_ADMM import *
from hierarchical_path import *
from projections import *
from utils import *

sys.stdout = sys.__stdout__ 
random.seed(2018)
RHO = 1.0



def mtx_from_utr(utr,complete=True):
    """
    create full weight matrix from upper triangle, taking mean across first axis
    complete: should bottom triangle be completed as well?
    """
    mtx=np.zeros((630,630))
    if len(utr.shape)>1:
        mtx[np.triu_indices_from(mtx,1)]=utr.mean(axis=0)
    else:
        mtx[np.triu_indices_from(mtx,1)]=utr

    if complete:
        mtx=mtx+mtx.T
    return(mtx)


if __name__ == '__main__':
    parser = ArgumentParser("Run evaluation on connectome dataset.")
    parser.add_argument("-logger","--loggerfile", help="logger file name", default='final_log_connectome_DS_')
    parser.add_argument("-savefile","--savefile", help="save file name", default='01')
    parser.add_argument("-i","--inputfile", help="input file name in the data folder",
                        default='data/data_fmri/rsfmri/corrdata.npy')
    parser.add_argument("-a","--alpha", help="alpha", default=0.95, type=float)
    parser.add_argument("-s","--sigma", help="bandwith for kernel", default=200.0, type=float)
    parser.add_argument("-l0","--lambd0", help="lambda 0 ",default=1e-3, type=float)
    parser.add_argument("-tol","--tol", help="tolerance for stopping criterion", default=1e-2, type=float)
    parser.add_argument("-nn","--n_neighbors", help="nb nearest_neighbors", default=10, type=int)
    parser.add_argument("-max_iter_fista","--max_iter_fista",help="max_iter_fista", default=150, type=int)
    parser.add_argument("-algo", "--algorithm", default="FISTA")
    parser.add_argument("-w", "--which_session", default=0, type=int)
    args = parser.parse_args()

    
    INPUTFILE = args.inputfile
    ALGO = args.algorithm
    WHICH_SESSION = args.which_session
    SAVEFILE = 'results/final_connectome_results_' +args.loggerfile + '_' + str(WHICH_SESSION) + '.pkl'
    LOG_FILE = 'logs/' + args.loggerfile + '_' + str(WHICH_SESSION) + '.log'
    ALPHA = args.alpha
    SIGMA = args.sigma
    N_NEIGHBORS = args.n_neighbors
    LAMBDA0 = args.lambd0
    TOL = args.tol
    MAXITERFISTA = args.max_iter_fista
    

    
    
    logger = logging.getLogger('myapp')
    fh = logging.FileHandler(LOG_FILE)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh) 
    logger.setLevel(logging.DEBUG) # or any level you want



    corrdata = np.load(INPUTFILE)
    # load the correlation data. This matrix contains
    # the upper triangle of the correlation matrix between each of the 630
    # regions, across each of the 84 good sessions

    meancorr=np.mean(corrdata,0)
    density=0.1 
    cutoff=scoreatpercentile(meancorr,100-density*100)
    
    #### Threshold the correlation matrix
    i = WHICH_SESSION
    SAVEFILE = 'data/results_' +args.loggerfile + '_' + str(i) + '.pkl'
    adjmtx=mtx_from_utr(corrdata[i ,:])
    adjmtx[adjmtx<cutoff]=0
    #adjmtx[adjmtx>0]=1
    np.fill_diagonal(adjmtx, 1)  ### this has to yield a similarity matrix
    K = sc.sparse.csr_matrix(adjmtx)
    K = K.T.dot(K)
    sqrtv = np.vectorize(lambda x: 1.0/np.sqrt(x) if x > 1e-10 else 0.0)
    Deg = np.diagflat(sqrtv(K.diagonal()))
    K = sc.sparse.csc_matrix(Deg.dot(K.dot(Deg)))
    n_nodes = K.shape[0]



    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")

    pi_prev = np.eye(n_nodes)
    pi, time, evol_rank = compute_reg_path(K, ALPHA, pi_warm_start=pi_prev, mode= 'FISTA', verbose=True,
                                          logger = logger, savefile=SAVEFILE)
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("DONE")
    
    
    