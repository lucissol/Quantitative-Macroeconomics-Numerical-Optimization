#%% import modules
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

#%% Value function iteration
class ValueFunctionMethods:
    '''
    A class implementing various value function iteration methods for dynamic programming
    '''
    
    def __init__(self, beta, alpha, state_grid, choice_grid=None):
        '''
        Parameters
        ----------
        beta : TYPE
            DESCRIPTION.
        alpha : TYPE
            DESCRIPTION.
        kgrid : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.beta = beta
        self.alpha = alpha
        self.kgrid = state_grid
        self.choice_grid = choice_grid if choice_grid is not None else state_grid
        self.nk = len(state_grid)
        self.u_matrix = self._compute_utility_matrix()
        
    @staticmethod
    def _log_util(x):
        '''Get log utility'''
        return np.where(x > 1e-12, np.log(x), np.log(1e-12))
    
    def _compute_utility_matrix(self):
        '''
        Parameters
        ----------
        a_current : Vector
            Asset state space.
        a_next : Vector
            Choice grid.
        alpha : float
            Parameter.

        Returns
        -------
        utility : float
            Invariant log utility for asset grid.

        '''
        c = self.kgrid[:, None]**self.alpha - self.choice_grid[None, :]

        # mask out negative consumption
        utility = np.full_like(c, -1e10, dtype=float)
        positive_mask = c > 0
        utility[positive_mask] = self._log_util(c[positive_mask])
        return utility
    
    
    @staticmethod
    def solve_value_function_sparse(u, beta, nk, policy_new):
        """Sparse solver for value function iteration"""
        # Build sparse transition matrix efficiently
        Q = sp.csr_matrix((np.ones(nk), (np.arange(nk), policy_new)), shape=(nk, nk))
        
        # Build sparse system: (I - beta * Q)
        A = sp.identity(nk) - beta * Q
        
        # Extract utilities for optimal policy
        u_policy = u[np.arange(nk), policy_new]
        
        # Solve sparse system
        return spla.spsolve(A, u_policy)

    def PFIhoward_exact(self, Vguess, tol = 1e-9, max_iter = 1000, verbose=True):
        '''
        Parameters
        ----------
        Vguess : Float
            First guess.
        beta : Float
            Parameter.
        alpha : Float
            Parameter.
        a_current : Float
            Asset state space.
        a_next : Float
            Choice grid.
        tol : TYPE, optional
            Tolerance level for Value Iteration. The default is 10e-9.
        max_iter : Int, optional
            Maximum number of iterations. The default is 10000.
    
        Raises
        ------
        ValueError
            Check asset space.
    
        Returns
        -------
        V : float
            Value Function.
        a_policy : float
            Policy Function.
        time_s : float
            Full time call to return.
        it : float
            Number of iterations needed.
        '''
        
        start = time.perf_counter()
        V = Vguess.copy()
        policy_old = np.ones(self.nk) * 0.5  # Non-integer to avoid immediate convergence
        if verbose:
            print("Starting Howard Policy Function Iteration...")
            
        for it in range(max_iter):
            phi = self.u_matrix + self.beta*V[np.newaxis, :] # matrix with full values of dimension n x nprime
            policy_new = np.argmax(phi, axis=1) # take max out of each row and store optimal policy index
            # policy evaluation check with sparse solver
            V = self.solve_value_function_sparse(self.u_matrix, self.beta, self.nk, policy_new)
    
            #check convergance criterium for policy
            policy_diff = np.sum(policy_new != policy_old)
            if policy_diff == 0:  # Equivalent to array_equal but faster
                end = time.perf_counter()
                time_s = end - start
                # store policy function
                a_opt = self.choice_grid[policy_new]
                if verbose:
                    print(f"..Converged! in {time_s} seconds\n")
                    print(f"Number of iterations: {it}")
                return V, a_opt, time_s, it
            else:
                policy_old = policy_new.copy() # update guess
        # If we reach here, maximum iterations were exceeded
        end = time.perf_counter()
        time_s = end - start
        a_opt = self.choice_grid[policy_new]
        
        if verbose:
            print(f"Warning: Maximum iterations ({max_iter}) reached after {time_s:.4f} seconds")
        
        return V, a_opt, time_s, max_iter