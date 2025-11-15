# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 12:36:44 2025

@author: lseibert
About utiltiy and macro models
"""
import numpy as np

def utility_CRRA(x, gamma):
    '''
    Return utility of 

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    gamma : float
        intertemporal coefficicent of substitution.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if gamma == 1:
        return np.where(x > 1e-12, np.log(x), np.log(1e-12))
    else:
        return ((x)**(1-gamma)) / (1 - gamma)
    
    
def marg_util_CRRA(x, gamma):
    if gamma == 1:
        return (1/x)
    else:
        return x**(-gamma)
    
def comp_util_grid(kgrid, kgrid, alpha, gamma):

    c = kgrid[:, None]**alpha - kgrid[None, :]

    # mask out negative consumption
    utility = np.full_like(c, -1e10, dtype=float)
    positive_mask = c > 0
    utility[positive_mask] = util(c[positive_mask])
    return utility

def marg_prod():
    theta * alpha * k**(alpha-1)
