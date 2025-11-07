# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 15:59:25 2025

@author: lseibert
script to tackle dynamic programming problems
with aggregate uncertainty
"""
# import packages
import numpy as np
from scipy.stats import norm

class discretize_AR1:
    """
    Class to discretize an AR(1) process using Tauchen or Rouwenhorst methods.

    Attributes
    ----------
    rho : float
        AR(1) persistence parameter (0 <= |rho| < 1). 
    mu : float
        Unconditional mean of the AR(1) process.
    sigma : float
        Standard deviation of the AR(1) innovations.

    Methods
    -------
    Tauchen(N, m=3)
        Discretizes the AR(1) process using Tauchen's method.
    Rouwenhorst(N)
        Discretizes the AR(1) process using Rouwenhorst's method.
    """

    def __init__(self, rho, mu, sigma):
        self.rho = rho
        self.mu = mu
        self.sigma = sigma

    def Tauchen(self, N, m=3):
        """
        Discretize the AR(1) process using Tauchen's method.
    
        Tauchen's method approximates a continuous AR(1) process with a discrete Markov chain.
        It is particularly useful for solving dynamic programming problems numerically.
    
        Parameters
        ----------
        N : int
            Number of discrete states (grid points) for the AR(1) process.
        m : float, optional
            Scaling parameter that sets the grid width in multiples of the unconditional standard deviation.
            Default is 3.
    
        Uses (from class attributes)
        ---------------------------
        rho : float
            AR(1) persistence parameter.
        mu : float
            Unconditional mean of the AR(1) process.
        sigma : float
            Standard deviation of AR(1) innovations.
    
        Returns
        -------
        zgrid : ndarray of shape (N,)
            Grid of discrete state values.
        Q : ndarray of shape (N, N)
            Transition probability matrix. Each row sums to 1.
        """

        z_max = self.mu + (m * self.sigma)/(np.sqrt(1-self.rho**2))
        z_min = -z_max + 2*self.mu
        zgrid = np.linspace(z_min, z_max, N)
        dz = (zgrid[1] - zgrid[0])    
        
        # Mean conditional on today's state
        mean = self.rho*zgrid[:, None] + (1-self.rho)*self.mu
        upper_std = (zgrid[None, :] + dz/2 - mean) / self.sigma
        lower_std = (zgrid[None, :] - dz/2 - mean) / self.sigma

        # Transition matrix
        Q = norm.cdf(upper_std) - norm.cdf(lower_std)     
        # Edge cases
        Q[:, 0] = norm.cdf((zgrid[0] + dz/2 - mean.flatten()) / self.sigma)  # First grid
        Q[:, -1] = 1 - norm.cdf((zgrid[-1] - dz/2 - mean.flatten()) / self.sigma)  # Last grid
         
        return zgrid, Q

    def Rouwenhorst(self, N):
        """
        Discretize the AR(1) process using Rouwenhorst's method.
    
        Rouwenhorst's method approximates a continuous AR(1) process with a discrete Markov chain.
        It preserves the first two moments of the process well even for highly persistent AR(1) processes 
        (rho close to 1) and works well with a small number of states.
    
        Parameters
        ----------
        N : int
            Number of discrete states (grid points) for the AR(1) process.
    
        Uses (from class attributes)
        ---------------------------
        rho : float
            AR(1) persistence parameter.
        mu : float
            Unconditional mean of the AR(1) process.
        sigma : float
            Standard deviation of AR(1) innovations.
    
        Returns
        -------
        zgrid : ndarray of shape (N,)
            Grid of discrete state values for the AR(1) process.
        Q : ndarray of shape (N, N)
            Transition probability matrix. Each row sums to 1.
        """

        # define grid
        z_end = np.sqrt(N-1) * self.sigma / (np.sqrt(1-self.rho**2))
        zgrid = np.linspace(-z_end, z_end, N) + self.mu

        #Step 2: Compute the transition matrix P recursively: Let
        Q = self._generate_matrix_rouwen(N)

        return zgrid, Q

    def _generate_matrix_rouwen(self, N):
        '''
        Recursive generation of the transition matrix for Rouwen's Method'

        Parameters
        ----------
        N : Int
            Number of final states.

        Returns
        -------
        Matrix
            Transition Matrix.

        '''
        p = (1 + self.rho) / 2
        if N == 1:
            return np.array([[1.0]])
        elif N == 2:
            # Base case
            return np.array([[p, 1-p],
                            [1-p, p]])
        else:
            # Recursive construction
            Q_prior = self._generate_matrix_rouwen(N-1)
            Q = np.zeros((N, N))
            topl = np.pad(Q_prior, ((0,1),(0,1)), 'constant')
            topr = np.pad(Q_prior, ((0,1),(1,0)), 'constant')
            botl = np.pad(Q_prior, ((1,0),(0,1)), 'constant')
            botr = np.pad(Q_prior, ((1,0),(1,0)), 'constant')
            
            Q = p*topl + (1-p)*topr + (1-p)*botl + p*botr
            Q[1:-1, :] /= 2
            
            return Q
        
    def simulate_time_series(self, T, zgrid, Q, initial_state_idx=None, seed=42):
         """
         Simulate a time series based on the discretized AR(1) process.
    
         Parameters
         ----------
         T : int
             Number of time periods to simulate.
         zgrid : ndarray
             The grid of discrete state values from Tauchen or Rouwenhorst.
         Q : ndarray
             The transition probability matrix from Tauchen or Rouwenhorst.
         initial_state_idx : int, optional
             The starting index of the grid. Default is None, which selects a random starting index.
    
         Returns
         -------
         path : ndarray
             Simulated time series of the AR(1) process.
         """
         np.random.seed(seed)
         if initial_state_idx is None:
             initial_state_idx = np.random.randint(0, len(zgrid))
         
         path = np.zeros(T, dtype=int)
         path[0] = zgrid[initial_state_idx]
         
         for t in range(1, T):
             # Transition to the next state based on the current state and transition probabilities
             path[t] = np.random.choice(len(zgrid), p=Q[path[t-1]])
         
         return zgrid[path]