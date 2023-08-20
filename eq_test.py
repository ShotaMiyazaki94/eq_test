import numpy as np
from numpy import random
from scipy.spatial.distance import pdist, cdist

def eq_test(x,y,nboot=1000):
    """
    Multi-dimentional equality test of distributions using energy L1-distances of the samples.
    This code is based on the paper, "DISCO analysis: A nonparametric extension of analysis of variance", Maria L. Rizzo, Gábor J. Székely (2010)
    https://projecteuclid.org/journals/annals-of-applied-statistics/volume-4/issue-2/DISCO-analysis-A-nonparametric-extension-of-analysis-of-variance/10.1214/09-AOAS245.full
    
    Parameters
    ----------
    x : ndarray, shape (n1, p)
        Data of sample 1 with p dimensions
    y : ndarray, shape (n2, p)
        Data of sample 2 with p dimensions
        The sample sizes of n1 and n2 can be different.
    nboot : None or int
        Number of bootstrap resample for estimating the empirical p-value.

    Returns
    -------
    p : float
        The permutation p-value that rejects the null hypothesis that "The distributions of the two samples are identical".
        You may increase nboot for reducing the p-value.
    
    F : float
        The DISCO F_alpha value calculated using the two samples
    
    -----------
    Usage:
    x1 = np.random.normal(0,0.1,10)
    y1 = np.random.normal(0,0.1,10)
    x2 = np.random.normal(1,0.1,300)
    y2 = np.random.normal(0,0.1,300)
    result = eq_test(np.c_[x1,y1],np.c_[x2,y2])
    
    """
    Nx,Ny,N = len(x),len(y),len(x)+len(y)
    stack   = np.vstack([x, y])
    F1_en   = energy(stack[:Nx], stack[Nx:])
    F1_boot = np.zeros(nboot)
    rand    = random.permutation
    for i in range(nboot):
        idx = rand(N)
        F1_boot[i] = energy2(stack[idx[:Nx]],stack[idx[Nx:]])
    p = (1+np.sum(F1_boot >= F1_en)) / (1+nboot)
    return p,F1_en
    
def energy(x, y):
    n, m = len(x), len(y)
    dx, dy, dxy = pdist(x,'minkowski',p=1), pdist(y,'minkowski',p=1), cdist(x, y,'minkowski',p=1)
    S1 = (2*np.sum(dxy)/(n*m) - np.sum(dx)/(n*n) - np.sum(dy)/(m*m)) * (n*m)/(n+m)
    W1 = 0.5*(np.sum(dx)/n+np.sum(dy)/m)
    F1 = S1/W1
    return F1
