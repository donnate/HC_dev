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
RHO = 1.0

if __name__ == '__main__':
    parser = ArgumentParser("Run evaluation on MNIST dataset.")
    parser.add_argument("-logger","--loggerfile",help="logger file name",default='log_synthetic.log')
    parser.add_argument("-savefile","--savefile",help="save file name",default='tests.pkl')
    parser.add_argument("-i","--inputfile",help="input file name in the data folder",default='tests.csv')
    parser.add_argument("-a","--alpha",help="alpha",default=0.95, type=float)
    parser.add_argument("-s","--sigma",help="bandwith for kernel",default=200.0, type=float)
    parser.add_argument("-l0","--lambd0",help="lambda 0 ",default=1e-3, type=float)
    parser.add_argument("-tol","--tol",help="tolerance for stopping criterion",default=5*1e-3, type=float)
    parser.add_argument("-nn","--n_neighbors",help="nb nearest_neighbors",default=10, type=int)
    parser.add_argument("-max_iter_fista","--max_iter_fista",help="max_iter_fista",default=150, type=int)
    parser.add_argument("-algo", "--algorithm", default="FISTA")
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
    MAXITERFISTA = args.max_iter_fista
    SAVEFILE = args.savefile
    INPUTFILE = args.inputfile
	ALGO = args.algorithm
	
    data = pd.DataFrame.from_csv("data/"+ INPUTFILE +".csv")
    n_nodes = K.shape[0]
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    from hierarchical_path import *
    pi_prev = np.eye(n_nodes)
    pi_ADMM, time_ADMM = compute_reg_path(K, alpha, mode= ALGO, verbose=True,
                                          logger = logger)
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("DONE")
    
    
    