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
    parser.add_argument("-logger","--loggerfile", help="logger file name", default='connectome3_')
    parser.add_argument("-savefile","--savefile", help="save file name", default='01')
    parser.add_argument("-i","--inputfile", help="input file name in the data folder",
                        default='data/rsfmri/corrdata.npy')
    parser.add_argument("-a","--alpha", help="alpha", default=0.95, type=float)
    parser.add_argument("-l0","--lambd0", help="lambda 0 ",default=1e-3, type=float)
    parser.add_argument("-tol","--tol", help="tolerance for stopping criterion", default=1e-2, type=float)
    parser.add_argument("-max_iter_fista","--max_iter_fista",help="max_iter_fista", default=30, type=int)
    parser.add_argument("-algo", "--algorithm", default="FISTA")
    parser.add_argument("-w", "--which_session", default=0, type=int)
    args = parser.parse_args()

    
    INPUTFILE = args.inputfile
    ALGO = args.algorithm
    WHICH_SESSION = args.which_session
    SAVEFILE = 'data/results_simplex_' +args.loggerfile + '_' + str(WHICH_SESSION) + '.pkl'
    LOG_FILE = 'logs/simplex_' + args.loggerfile + '_' + str(WHICH_SESSION) + '.log'
    ALPHA = args.alpha
    LAMBDA0 = args.lambd0
    TOL = args.tol
    MAXITERFISTA = args.max_iter_fista
    

    
    
    logger = logging.getLogger('myapp')
    fh = logging.FileHandler(LOG_FILE)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh) 
    logger.setLevel(logging.DEBUG) # or any level you want



    corrdata = np.load("data/"+ INPUTFILE)
    # load the correlation data. This matrix contains
    # the upper triangle of the correlation matrix between each of the 630
    # regions, across each of the 84 good sessions
    parceldata = pd.read_csv('data/data/parcel_data.txt',
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
    np.fill_diagonal(adjmtx, 1)  ### this has to yield a similarity matrix
    K = sc.sparse.csr_matrix(adjmtx)



    n_nodes = K.shape[0]
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")

    pi_prev = np.eye(n_nodes)
    pi, time, evol_rank = compute_reg_path(K, ALPHA, pi_warm_start=pi_prev, mode= 'simplex', verbose=True,
                                          logger = logger, savefile=SAVEFILE)
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("*********************************************************************")
    logger.info("DONE")
    
    
    