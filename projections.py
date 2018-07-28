"""
Code for all the projection operations/ algorithms
"""
import numpy as np
import scipy as sc
import copy 


def project_unit_ball(m, is_sparse=True):
    ''' Projection onto the l2 ball
        INPUT:
        -----------------------------------------------------------
        m           :      matrix    
    
        OUTPUT:
        -----------------------------------------------------------
        Pi_m        :      projection of m unto the unit ball
    '''
    if is_sparse:
        norm_m = copy.deepcopy(m)
        norm_m.data = np.square(norm_m.data)
        thr = np.vectorize(lambda x: max(x,1))
        normed = thr(np.array(norm_m.sum(0)) )
        d = sc.sparse.diags(1.0 / np.sqrt(normed.flatten()), 0, format='lil')
        return (m.tolil().dot(d)).tocsc()
    else:
        def proj_vector(m):
            norm_m = np.linalg.norm(m)
            return m / np.max([norm_m,1])
        return np.apply_along_axis(proj_vector,0,m)


def project_unit_cube(m, is_sparse=True):
    ''' Projection onto the l1 cube
        INPUT:
        -----------------------------------------------------------
        m           :      matrix    
    
        OUTPUT:
        -----------------------------------------------------------
        Pi_m        :      projection of m unto the unit cube
    '''
    thr = np.vectorize(lambda x: np.sign(x) * np.min([abs(x),1]))
    if is_sparse:
        mm = copy.deepcopy(m)
        if mm.nnz > 0:
            mm.data = thr(mm.data)
        return mm
        
    else:
        thr =np.vectorize(lambda x: x if abs(x) < 1 else (1.0 if x>1 else -1.0))
        return thr(np.array(m))


def iteration_proj_DSS(Y):
    ''' Projection onto the set of doubly stochastic matrices
    '''
    n = Y.shape[0]
    pos = np.vectorize(lambda x: max(x,0))
    Y2 = Y + 1.0 / n * ( np.eye(n)- Y + 1.0 / n * np.ones((n,n)).dot(Y)).dot(np.ones((n,n)))\
            - 1.0 / n * np.ones((n,n)).dot(Y)
    Y2 = pos(Y2) 
    return Y2


def project_DS_symmetric(G_tilde, max_it=500, eps=0.01):
    ''' Projection onto the set of doubly stochastic symmetric
        matrices
        INPUT:
        -----------------------------------------------------------
        G_tilde     :      matrix    
        max_it      :      max number of iterations
        eps         :      convergence tolerance
    
        OUTPUT:
        -----------------------------------------------------------
        Y           :      projection of m unto the set od DSS matrices
    '''
    converged = False
    n = G_tilde.shape[0]
    Y = G_tilde
    I = np.eye(n)
    it = 0
    while not converged:
        Y = iteration_proj_DSS(Y)
        converged = (np.mean(np.abs(Y.sum(0) - 1)) < eps
                     and np.mean(np.abs(Y.sum(1) - 1)) < eps)\
                     or (it > max_it)
        it += 1
    return Y


def iteration_proj_DS(Y):
    ''' Projection onto the set of doubly stochastic matrices

    '''
    n = Y.shape[0]
    u = np.ones(n)
    I = np.eye(n)
    P1 = Y + (1.0 / n * (I - Y) + 1.0 / n**2 *\
             Y.sum() * I).dot(np.ones((n,n)))\
             - 1.0 / n * np.ones((n,n)).dot(Y)
    Y2 = 0.5 * (P1 + np.abs(P1)) 
    return Y2


def project_DS2(G_tilde, max_it=100, eps=0.01):
    ''' Projection onto the set of doubly stochastic matrices
        INPUT:
        -----------------------------------------------------------
        G_tilde     :      matrix    
        max_it      :      max number of iterations
        eps         :      convergence tolerance

        OUTPUT:
        -----------------------------------------------------------
        Y           :      projection of m unto the DS set
    '''
    n = G_tilde.shape[0]
    u = np.ones(n)
    I = np.eye(n)
    Y = G_tilde
    Y_old = G_tilde
    G_converged = False
    it_G = 0
    while not G_converged:
        Y = iteration_proj_DS(Y)
        G_converged = (np.mean(np.abs(Y.sum(0) - 1)) < eps \
                             and np.mean(np.abs(Y.sum(1) - 1)) < eps)\
                             or (it_G > max_it)
        Y_old = Y
        it_G += 1
        #print(it_G)
    #print("Projection attained in %i iterations."%it_G)
    #print(Y.sum(0), Y.sum(1))
    return Y

def project_simplex(v, z=1.0):
    ''' Projection of a vector v unto the scaled simplex
        INPUT:
        -----------------------------------------------------------
        v         :     vector to be projected on the simplex
        z         :     scaling of the simplex

        OUTPUT:
        -----------------------------------------------------------
        y_tilde   :     projection of y unto the simplex
    '''
    if z<=0:
        print 'error: z must be positive'
        return None
    mu = np.sort(v)[::-1]
    temp = [mu[j - 1] - 1.0 / j * (np.sum(mu[: j]) - z)
            for j in range(1, len(mu) + 1)]
    cand_index = [j for j in range(len(temp)) if temp[j] > 0]
    rho = np.max(cand_index) + 1
    theta = 1.0 / rho * (np.sum(mu[: rho]) - z)
    def prune(x):
        return max(x - theta, 0)
    vprune = np.vectorize(prune)
    return np.array([max(v[i] - theta, 0) for i in range(len(v))])


def project_stochmat(G_tilde, orient=1):
    ''' Projection onto the set of stochastic matrices
        INPUT:
        -----------------------------------------------------------
        G_tilde     :      matrix
        orient      :      0 (row) or 1 (column): direction in which
                           the matrix is stochastic

        OUTPUT:
        -----------------------------------------------------------
        Y           :      projection of m unto the DS set
    '''
    n = G_tilde.shape[0]
    Y = np.apply_along_axis(project_simplex, orient, G_tilde)
    return Y