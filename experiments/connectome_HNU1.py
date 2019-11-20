from __future__ import print_function

from argparse import ArgumentParser
import copy
import logging
import networkx as nx
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


sys.path.append('../')
from convex_hc_denoising import *
from convex_hc_ADMM import *
from hierarchical_path import *
from projections import *
from utils import *

sys.stdout = sys.__stdout__ 
random.seed(2018)
RHO = 1.0




if __name__ == '__main__':
    parser = ArgumentParser("Run evaluation on connectome dataset.")
    parser.add_argument("-logger","--loggerfile", help="logger file name", default='final_log_connectome_DS_')
    parser.add_argument("-savefile","--savefile", help="save file name", default='01')
    parser.add_argument("-i","--inputfile", help="input file name in the data folder",
                        default='/Users/cdonnat/Dropbox/NeuroscienceFall18/HNU1/')
    parser.add_argument("-a","--alpha", help="alpha", default=0.95, type=float)
    parser.add_argument("-a_reg","--alpha_reg", help="regularization for the similarity matrix", default=0.1, type=float)
    parser.add_argument("-type_lap","--type_lap", help="Which laplacian to use?", default="normalized_laplacian", type=str)
    parser.add_argument("-s","--sigma", help="bandwith for kernel", default=200.0, type=float)
    parser.add_argument("-l0","--lambd0", help="lambda 0 ",default=1e-3, type=float)
    parser.add_argument("-tol","--tol", help="tolerance for stopping criterion", default=1e-2, type=float)
    parser.add_argument("-nn","--n_neighbors", help="nb nearest_neighbors", default=10, type=int)
    parser.add_argument("-max_iter_fista","--max_iter_fista",help="max_iter_fista", default=150, type=int)
    parser.add_argument("-algo", "--algorithm", default="FISTA")
    parser.add_argument("-w", "--which_session", default=1, type=int)
    parser.add_argument("-subj", "--which_subject", default=25427, type=int)
    parser.add_argument("-sess", "--which_session", default=1, type=int)
    args = parser.parse_args()


    INPUTFILE = args.inputfile
    ALGO = args.algorithm
    WHICH_SESSION = args.which_session
    WHICH_SUBJECT = args.which_subject
    SAVEFILE = 'final_connectomeHNU1_results_' +args.loggerfile+ str(WHICH_SUBJECT) + '_' + str(WHICH_SESSION) + '.pkl'
    LOG_FILE = 'logs_HNU1_' + args.loggerfile + '_' + str(WHICH_SUBJECT) + '_' + str(WHICH_SESSION) + '.log'
    ALPHA = args.alpha
    SIGMA = args.sigma
    N_NEIGHBORS = args.n_neighbors
    LAMBDA0 = args.lambd0
    TOL = args.tol
    MAXITERFISTA = args.max_iter_fista
    ALPHA_REG = args.alpha_reg
    TYPE_LAP = args.type_lap

    logger = logging.getLogger('myapp')
    fh = logging.FileHandler(LOG_FILE)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh) 
    logger.setLevel(logging.DEBUG) # or any level you want

    name_file = 'sub-00' + str(s) + '_ses-' + str(session)\
                                                    + '_bold_CPAC200_res-2x2x2_variant-mean_timeseries.npz'
    name_file_dwi = 'sub-00'+str(WHICH_SUBJECT)+'_ses-'+str(WHICH_SESSION)+'_dwi_CPAC200.gpickle'
    key = 'sub_' + str(WHICH_SUBJECT)+'_ses-'+str(WHICH_SESSION)
    print(key)
    try:
        graphs_dwi = pickle.load(open('/scratch/users/cdonnat/data/HNU1/'+name_file_dwi, 'rb'))
    except:
        print("graph not found for", key)





    adjmtx = nx.adjacency_matrix(graphs_dwi).todense()

    #adjmtx[adjmtx>0]=1
    #np.fill_diagonal(adjmtx, 1)  ### this has to yield a similarity matrix
    K = create_similarity_matrix(adjmtx, TYPE_LAP, ALPHA_REG)
    n_nodes = K.shape[0]



    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")

    pi_prev = np.eye(n_nodes)
    pi, time, evol_rank = compute_reg_path(K, ALPHA, pi_warm_start=pi_prev, mode=ALGO,
                                           verbose=True,
                                           logger = logger, savefile=SAVEFILE, rho=RHO)
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("DONE")
