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
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
from functools import wraps


def time_method(func):
    """Decorator to time class methods."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = func(self, *args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f" {func.__name__} took {elapsed:.3f} seconds")
        return result
    return wrapper


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
        
    @time_method
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
    
    @time_method
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
         seed: int, optional
             Random seed for reproducibility (default 44).
    
         Returns
         -------
         path : ndarray
             Simulated time series of the discretized AR(1) process.
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
     
class agg_uncertainty:
    '''
    Solving and simmulating a depreciation ramsey problem in an aggregate uncertainty environment.
    '''
    def __init__(self, beta, alpha, delta, state, Q):
        '''
        Setting up economic model parameters.

        Parameters
        ----------
        beta : int
            Discount factor.
        alpha : int
            Capital elasticity.
        delta : int
            Depreciation rate.
        state : array
            State values of discretized AR1. Use class discretize_AR1 to discretize values.
        Q : Transition Matrix
            Transition matrix of the shock state space.

        Returns
        -------
        None.

        '''
        self.beta = beta
        self.alpha = alpha
        self.delta = delta
        self.state = state
        self.nz = len(state)
        self.Q = Q

    @time_method
    def solve_EGM(self, kgrid, tol=1e-10, max_iter=1000):
        '''.
        Solving the aggregate uncertainty economy vie endogeneous grid method with a state space consisting of productivity and capital states.

        Parameters
        ----------
        self: Model parameter
        
        kgrid : 1d array
            Capital grid values.
        tol : TYPE, optional
            Policy convergence tolerance level . The default is 1e-10.
        max_iter : int, optional
            Maximum iterations for convergence on the policy. The default is 1000.

        Returns
        -------
        TYPE
            Implied policy function of form g(z,k) shape (nz, nk).

        '''
        nk = len(kgrid)
        nz = self.nz
        k_policy = np.tile(kgrid * 0.3, (nz, 1))  # Save 30% of capital
        k_policy_new = np.zeros((nz, nk))
        theta = self.state
        
        for i in range(max_iter):
            # 
            Emu = expected_marginal_utility(kgrid, theta, k_policy, self.Q, self.alpha, self.delta)

            # Today's consumption implied by Euler equation
            c_today = 1 / (self.beta * Emu)  # shape: (nz, nk)
            
            # Today's resources needed
            cash_at_hand = c_today + kgrid[None, :]  # cash at hand
            # Update policy using interpolation
            for z_i in range(nz):
                # Interpolate: resources_today -> capital_today
                policy_interp = interp1d(cash_at_hand[z_i, :], kgrid, kind='linear',
                       bounds_error=False, 
                       fill_value=(kgrid[0], kgrid[-1]))
                
                x_fixed = theta[z_i] * kgrid**self.alpha + (1 - self.delta) * kgrid
                k_policy_new[z_i, :] = policy_interp(x_fixed)


            # Check convergence
            if np.linalg.norm(k_policy_new - k_policy)<(1+np.linalg.norm(k_policy))*tol:
                print(f"Converged in {i} iterations")
                return k_policy_new
            
            k_policy = k_policy_new.copy()
        
        print("Max iterations reached")
        return k_policy
    
    
    def simulate_economy(self, kgrid, T, burn_in, plot=False, seed = 44, initial_state_idx = None):
        '''
        Simulate a time path of an aggregate uncertainty economy.

        Parameters
        ----------
        kgrid : TYPE
            DESCRIPTION.
        T : TYPE
            DESCRIPTION.
        burn_in : TYPE
            DESCRIPTION.
        plot : TYPE, optional
            DESCRIPTION. The default is False.
        seed : TYPE, optional
            Random seed for reproducibility (default 44).
        initial_state_idx : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        k_path: array
            Forward simulated cpaital path.
        c_path: array
            Forward simulated consumption path.
        y_path: array
            Forward simulated output path.
        path: array
            Markov shock state indices path.

        '''
        # get the policy of the model
        policy = self.solve_EGM(kgrid)
        
        path = generate_shock_path(self.Q, T=T, seed=seed, initial_state_idx=initial_state_idx)
        k_path   = np.empty(T)
        cons     = np.empty(T)
        output   = np.empty(T)
        k_path[0] = kgrid[0] # initial value for capital
        
        for t in range(T):
            z  = self.state[path[t]]
            # ressources
            yt = z * (k_path[t]**self.alpha)
            x_t = yt + (1 - self.delta) * k_path[t]
            # decision variable
            k_next = np.interp(k_path[t], kgrid, policy[path[t], :])
            cons[t] = x_t - k_next # consumption implied by the budget constraint
            output[t] = yt
                        
            if t < T-1:
                # next state variabels
                k_path[t+1] = k_next
                
        if plot:
            self._plot_economy(k_path, cons, output, path, burn_in)
        return (
            k_path[burn_in:], 
            cons[burn_in:], 
            output[burn_in:], 
            path[burn_in:]
            )

    def _plot_economy(self, k_path, cons, output, path, burn_in):
        """Plots simulated economy time series."""
        T = len(k_path)
        t = np.arange(T)
    
        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        fig.suptitle("Simulated Economy Over Time", fontsize=14)
    
        axs[0].plot(t, self.state[path[t]], color="gray")
        axs[0].set_ylabel("Shock State $z_t$")
        
        axs[1].plot(t, output, label="Output", color="tab:blue")
        axs[1].set_ylabel("Output")
        
        axs[2].plot(t, cons, label="Consumption", color="tab:green")
        axs[2].set_ylabel("Consumption")
        
        axs[3].plot(t, k_path, label="Capital", color="tab:red")
        axs[3].set_ylabel("Capital $k_t$")
        axs[3].set_xlabel("Time")
    
        # Mark burn-in period visually
        for ax in axs:
            ax.axvline(burn_in, color="black", linestyle="--", alpha=0.5)
            ax.legend()
    
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        
    def static_EEE(self, k_policy, kgrid, kfine): #WiP to vectorize and use of expected_marginal_utility
        '''
        Compute the Euler Equation Error of a policy function of shape (nz, nk) for each grid point
        
        Parameters
        ----------
        policy_a : flexible
            Default: optimal next year capital.
            If interpolating False: interpolated policy function that maps a to a_prime 
        kgrid : array
            Capital grid for today's state value.
        kfine : array
            Finer capital grid to evaluate performance of policy at interpolated points. 
        interpolating:
            if False computes interpolated policy function with cubic splines.
        
        Returns
        -------
        Euler Equation Error for each state of shape (nz, nk).
        '''
        
        euler_error = np.zeros((self.nz, len(kfine)))

        for z_i, z in enumerate(self.state):
            policy = interp1d(kgrid, k_policy[z_i, :], kind='cubic', fill_value='extrapolate')
            k_next = policy(kfine)
            # consumption today given budget constraint implied by policy
            cash = z * kfine**self.alpha + (1 - self.delta) * kfine
            c = cash - k_next
            # compute implied consumption from Euler equation
            cash_next = self.state[:, None] * k_next**self.alpha + (1 - self.delta) * k_next
            c_next = cash_next - policy(k_next)
            MPK_prime = self.alpha * self.state[:, None] * k_next**(self.alpha-1) + 1 - self.delta
            MU_prime = MPK_prime / c_next
            Emu = self.Q[z_i, :] @ MU_prime # expected marginal utility
            c_implied = 1 / (self.beta * Emu)
            
            # Euler error as relative deviation
            euler_error[z_i, :] = np.abs((c - c_implied) / c_implied)
        
        return euler_error

    @time_method
    def _dynamic_EE(self, k_policy, kgrid, T, burn_in, seed=44, initial_state_idx=None):
        """
        Generate a time series implied by the Euler equation with log utility given a policy.
    
        Parameters
        ----------
        k_policy : ndarray
            Policy function for capital, shape (nz, nk).
        kgrid : ndarray
            Grid for capital, shape (nk,).
        T : int
            Number of periods to simulate.
        Burnin : int
            Number of periods to burn in.
        seed : int, optional
            Random seed for reproducibility (default 44).
        initial_state_idx : int, optional
            Initial index of the exogenous state (default: random).
    
        Returns
        -------
        k_imp : ndarray
            Implied capital path over time.
        c_imp : ndarray
            Implied consumption path over time.
        z_level : ndarray
            Realizations of the exogenous state over time.
        """
    
        path = generate_shock_path(self.Q, T, seed, initial_state_idx)
        k_imp   = np.empty(T)
        c_imp     = np.empty(T)
        k_imp[0] = kgrid[0] # initial value for capital
        
        
        for t in range(T):
            # state shock level
            z = self.state[path[t]]
            # state shock determines policy
            policy = interp1d(kgrid, k_policy[path[t], :], kind='cubic', fill_value='extrapolate')
            k_current = np.array([k_imp[t]])
            k_next = policy(k_current)

            # compute implied consumption from Euler equation
            Emu = expected_marginal_utility(k_current, self.state, k_next, self.Q[path[t]], self.alpha, self.delta)
            c_imp[t] = 1 / (self.beta * Emu)
            
            if t < T-1: 
                k_imp[t+1] = z * k_imp[t] ** self.alpha + (1 - self.delta) * k_imp[t] - c_imp[t]
                
        return (k_imp[burn_in:], c_imp[burn_in:], path[burn_in:])
    
    
    @time_method
    def _dynamic_EE_old(self, k_policy, kgrid, T, burn_in, seed=44, initial_state_idx=None): #WiP
        """
        Generate a time series implied by the Euler equation with log utility.
    
        Parameters
        ----------
        k_policy : ndarray
            Policy function for capital, shape (nz, nk).
        kgrid : ndarray
            Grid for capital, shape (nk,).
        T : int
            Number of periods to simulate.
        seed : int, optional
            Random seed for reproducibility (default 44).
        initial_state_idx : int, optional
            Initial index of the exogenous state (default: random).
    
        Returns
        -------
        k_imp : ndarray
            Implied capital path over time.
        c_imp : ndarray
            Implied consumption path over time (from Euler equation).
        z_level : ndarray
            Realizations of the exogenous state over time.
        """
        np.random.seed(seed)
        
        if initial_state_idx is None:
            initial_state_idx = np.random.randint(0, self.nz)
    
        path = np.zeros(T, dtype=int)
        k_imp   = np.empty(T)
        z_level  = np.empty(T)
        c_imp     = np.empty(T)

        
        path[0] = initial_state_idx
        z_level[0] = self.state[initial_state_idx]
        k_imp[0] = kgrid[10] # initial value for capital
        cdf_Q = np.cumsum(self.Q, axis=1)  # Compute CDF for each row
        
        
        for t in range(T):
            # state shock determines policy
            policy = interp1d(kgrid, k_policy[path[t], :], kind='cubic', fill_value='extrapolate')
            k_next = policy(k_imp[t])
            
            # compute implied consumption from Euler equation
            cash_next = self.state[:, None] * k_next**self.alpha + (1 - self.delta) * k_next
            c_next = cash_next - policy(k_next)
            MPK_prime = self.alpha * self.state[:, None] * k_imp[t]**(self.alpha-1) + 1 - self.delta
            MU_prime = MPK_prime / c_next
            Emu = self.Q[path[t], :] @ MU_prime
            c_imp[t] = 1 / (self.beta * Emu)
            
            if t < T-1: 
                u = np.random.rand()  # Uniform random number between 0 and 1
                path[t+1] = np.searchsorted(cdf_Q[path[t]], u) # shock
                z_level[t+1] = self.state[path[t+1]]
                k_imp[t+1] = z_level[t] * k_imp[t] ** self.alpha + (1 - self.delta) * k_imp[t] - c_imp[t]
                
        return (k_imp[burn_in:], c_imp[burn_in:], path[burn_in:])
    
def generate_shock_path(Q, T, seed=44, initial_state_idx=None):
    '''
    Generate a time series of shock indices.

    Parameters
    ----------
    Q : TYPE
        Transition Matrix. Rows must sum up to 1
    T : TYPE
        Time points.
    seed : TYPE, optional
        Random seed for reproducibility (default 44).
    initial_state_idx : TYPE, optional
       Initial index of the exogenous state (default: random).
       
    Returns
    -------
    path : Array of integers
        Time path of a discretized AR1.

    '''
    np.random.seed(seed)
    nz = Q.shape[0]
    if initial_state_idx is None:
        initial_state_idx = np.random.randint(0, nz)
    path = np.zeros(T, dtype=int)
    path[0] = initial_state_idx

    cdf_Q = np.cumsum(Q, axis=1)
    for t in range(T - 1):
        u = np.random.rand()
        path[t + 1] = np.searchsorted(cdf_Q[path[t]], u)
    return path

def expected_marginal_utility(kgrid, theta, k_policy, Q, alpha, delta):
    """
    Compute expected marginal utility for each state and capital for log utility. Underlying strucutre assumes a discretized marcov chain.

    Parameters
    ----------
    kgrid : (nk,) array
        Capital grid.
    theta : (nz,) array
        Productivity states.
    k_policy : (nz, nk) array
        Next-period capital policy.
    Q : (nz, nz) array
        Transition matrix.
    alpha : float
        Capital share in production.
    delta : float
        Depreciation rate.

    Returns
    -------
    Emu : (nz, nk) array
        Expected marginal utility.
    """
    # Next-period consumption
    c_next = theta[:, None] * kgrid[None, :]**alpha + (1 - delta) * kgrid[None, :] - k_policy

    # Marginal product of capital next period
    MPK_prime = (1 - delta) + theta[:, None] * alpha * kgrid[None, :]**(alpha - 1)

    # Marginal utility next period
    MU_prime = MPK_prime / c_next

    # Expected marginal utility
    Emu = Q @ MU_prime
    return Emu

    
    