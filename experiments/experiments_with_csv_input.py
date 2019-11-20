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

sys.path.append("/scratch/users/cdonnat/convex_clustering/HC_dev")
from convex_hc_denoising import *
from convex_hc_ADMM_nn_sparse import *
from hierarchical_path import *
from projections import *
from utils import *
from utils_graphs import *

sys.stdout = sys.__stdout__ 
random.seed(2018)
RHO = 1.0

if __name__ == '__main__':
    parser = ArgumentParser("Run evaluation on synthetic dataset.")
    parser.add_argument("-logger","--loggerfile",help="logger file name",default='log_synthetic.log')
    parser.add_argument("-path2data","--path2data", help="path2data", default='/scratch/users/cdonnat/data/HC_data')
    parser.add_argument("-path2logs","--path2logs", help="path2logs", default='/scratch/users/cdonnat/convex_clustering/HC_dev/experiments/logs/')
    parser.add_argument("-savefile","--savefile",help="save file name",default='01')
    parser.add_argument("-i","--inputfile",help="input file name in the data folder",default='synthetic.csv')
    parser.add_argument("-a","--alpha", help="alpha", default=0.95, type=float)
    parser.add_argument("-a_reg","--alpha_reg", help="regularization for the similarity matrix", default=0.1, type=float)
    parser.add_argument("-type_lap","--type_lap", help="Which laplacian to use?", default="normalized_laplacian", type=str)
    parser.add_argument("-s","--sigma", help="bandwith for kernel", default=200.0, type=float)
    parser.add_argument("-l0","--lambd0", help="lambda 0 ", default=1e-3, type=float)
    parser.add_argument("-tol","--tol", help="tolerance for stopping criterion", default=5*1e-3, type=float)
    parser.add_argument("-nn","--n_neighbors", help="nb nearest_neighbors", default=10, type=int)
    parser.add_argument("-t","--is_train", help="use the training set(1) or test set (0)?", default=1, type=int)
    parser.add_argument("-max_iter_fista", "--max_iter_fista", help="max_iter_fista", default=150, type=int)
    parser.add_argument("-algo", "--algorithm", default="FISTA")
    args = parser.parse_args()

    ALPHA = args.alpha
    ALPHA_REG = args.alpha_reg
    N_NEIGHBORS = args.n_neighbors
    LAMBDA0 = args.lambd0
    LOGGER_FILE = args.loggerfile
    MAXITERFISTA = args.max_iter_fista
    PATH2DATA = args.path2data
    PATH2LOGS = args.path2logs
    SAVEFILE = args.savefile
    SIGMA = args.sigma
    TOL = args.tol
    TYPE_LAP = args.type_lap
    
    INPUTFILE = args.inputfile
    SAVEFILE = args.savefile
    ALGO = args.algorithm

    data = pd.DataFrame.from_csv(INPUTFILE)
    print(data.values)
    print([TYPE_LAP, ALPHA_REG])
    K = create_similarity_matrix(data.values, TYPE_LAP, ALPHA_REG)
    n_nodes = K.shape[0]
                                 
    logger = logging.getLogger('myapp')
    fh = logging.FileHandler(LOGGER_FILE)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh) 
    logger.setLevel(logging.DEBUG) # or any level you want 

    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")

    pi_prev = np.eye(n_nodes)
    pi, time, evol_rank = compute_reg_path(K, ALPHA, pi_warm_start=pi_prev, mode=ALGO,
                                           verbose=True, logger=logger, savefile=SAVEFILE)
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("DONE")
    
    
    
