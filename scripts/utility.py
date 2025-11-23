# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 12:36:44 2025

@author: lseibert
About utiltiy and macro models
"""
import numpy as np

def utility_CRRA(c, gamma):
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
    u = np.full_like(c, -np.inf, dtype=float)
    pos_c = c > 0 
    
    if gamma == 1:
        u[pos_c] = np.log(c[pos_c])
    else:
        u[pos_c] = ((c[pos_c])**(1-gamma)) / (1 - gamma)
    return u
    
def marg_util_CRRA(x, gamma):
    if gamma == 1:
        return (1/x)
    else:
        return x**(-gamma)
    
def comp_util_grid(kgrid, alpha, gamma):

    c = kgrid[:, None]**alpha - kgrid[None, :]

    # mask out negative consumption
    utility = np.full_like(c, -1e10, dtype=float)
    positive_mask = c > 0
    utility[positive_mask] = util(c[positive_mask])
    return utility

def marg_prod():
    theta * alpha * k**(alpha-1)
