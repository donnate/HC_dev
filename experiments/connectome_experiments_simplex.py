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
from utils_graphs import *

sys.stdout = sys.__stdout__ 
random.seed(2018)




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
    parser.add_argument("-path2data","--path2data", help="path2data", default='/scratch/users/cdonnat/data/HC_data')
    parser.add_argument("-path2logs","--path2logs", help="path2logs", default='/scratch/users/cdonnat/convex_clustering/experiments/logs/')
    parser.add_argument("-logger","--loggerfile", help="logger file name", default='log_digits.log')
    parser.add_argument("-savefile","--savefile", help="save file name", default='digits_new.pkl')
    parser.add_argument("-a","--alpha", help="alpha", default=0.95, type=float)
    parser.add_argument("-a_reg","--alpha_reg", help="regularization for the similarity matrix", default=0.1, type=float)
    parser.add_argument("-type_lap","--type_lap", help="Which laplacian to use?", default="normalized_laplacian", type=str)
    parser.add_argument("-tol","--tol", help="tolerance for stopping criterion", default=1e-2, type=float)
    parser.add_argument("-max_iter_fista","--max_iter_fista",help="max_iter_fista", default=30, type=int)
    parser.add_argument("-algo", "--algorithm", default="FISTA")
    parser.add_argument("-w", "--which_session", default=0, type=int)
    parser.add_argument("-rho", "--rho", default=1.0, type=float)
    args = parser.parse_args()

    
    ALPHA = args.alpha
    ALPHA_REG = args.alpha_reg
    N_NEIGHBORS = args.n_neighbors
    LAMBDA0 = args.lambd0
    MAXITERFISTA = args.max_iter_fista
    PATH2DATA = args.path2data
    PATH2LOGS = args.path2logs
    SAVEFILE = PATH2LOGS + '/connectome_alpha_' + str(ALPHA) + args.savefile
    LOGGER_FILE = PATH2LOGS +  '/connectome_alpha_' + str(ALPHA) + args.loggerfile
    RHO = args.rho
    SIGMA = args.sigma
    TOL = args.tol
    TYPE_LAP = args.type_lap
    

    
    
    logger = logging.getLogger('myapp')
    fh = logging.FileHandler(LOGGER_FILE)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh) 
    logger.setLevel(logging.DEBUG) # or any level you want



    corrdata = np.load(PATH2DATA + "/" + INPUTFILE)
    # load the correlation data. This matrix contains
    # the upper triangle of the correlation matrix between each of the 630
    # regions, across each of the 84 good sessions
    parceldata = pd.read_csv(PATH2DATA + 'parcel_data.txt',
                          header=None,sep='\t')
    parceldata.replace(to_replace='na', value='Subcortical', inplace=True)
    parceldata.replace(to_replace='Zero', value='Unassigned', inplace=True)
    parceldata.columns=['num','hemis','X','Y','Z','lobe','sublobe',
                        'power','yeo7','yeo17']

    meancorr=np.mean(corrdata,0)
    density=0.1 
    cutoff=scoreatpercentile(meancorr,100-density*100)
    
    #### Threshold the correlation matrix
    i = WHICH_SESSION
    adjmtx=mtx_from_utr(corrdata[i ,:])
    adjmtx[adjmtx<cutoff]=0
    #adjmtx[adjmtx>0]=1

    K = create_similarity_matrix(adjmtx, TYPE_LAP, ALPHA_REG)



    n_nodes = K.shape[0]
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")

    pi_prev = np.eye(n_nodes)
    pi, time, evol_rank = compute_reg_path(K, ALPHA, pi_warm_start=pi_prev, mode= 'simplex', verbose=True,
                                          logger = logger, savefile=SAVEFILE, rho=RHO)
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("DONE")
    
    
    
