# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 17:26:47 2025
Script to run ayagari model
@author: lseibert
"""
import numpy as np
from functools import wraps
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, identity
from scipy.interpolate import interp1d
from scipy import optimize as opt
import time
import numba as njit

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


class Economy:
    def __init__(self, alpha, beta, delta, states, Q):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.states = states
        self.Q = Q
        
class Household:
    def __init__(self, Economy, sigma, na, b):
        self.econ = Economy
        self.sigma = sigma
        self.b = b      
        self.agrid = construct_grid(
                        b=b,
                        max_state=np.max(self.econ.states),
                        na=na
                        )
        print(f"Defined asset grid on {np.min(self.agrid):.2f} - {np.max(self.agrid):.2f}")
        print(f"For more info, access it class {type(self).__name__} stored youre_class_name.agrid")
        
    @time_method
    def _solve_EGM(self, r, w, max_iter=1000, tol=1e-8, lamb = 0.7, verbose=False):
        na = len(self.agrid)
        nz = len(self.econ.states)
        k_policy = np.zeros((nz, na))
        k_policy[:] = self.agrid * 0.5
        k_policy_new = np.zeros((nz, na))
        xgrid = comp_ressources(self.agrid, r, w, self.econ.states)
        diff_list = np.zeros(max_iter)
        constr = np.zeros(max_iter)

        for i in range(max_iter):
            c_next = xgrid - k_policy
            # compute implied consumption today
            # compute C_t**-sigma = E(c-t+1**-sigma) * (R+beta)
            marginal_utility = Emu(c_next, r, self.sigma, self.econ.beta, self.econ.Q)
            # invert ot get c_today!
            c_today = (marginal_utility) ** (-1/self.sigma)
            # compute implied cash at hand
            cash_imp = c_today + self.agrid
            for z_i in range(nz):

                # Interpolate: resources_today -> capital_today
                policy_interp = interp1d(cash_imp[z_i, :], self.agrid, kind='linear',
                       bounds_error=False, fill_value="extrapolate") 
                k_policy_new[z_i, :] = policy_interp(xgrid[z_i, :])
                binding_indices = xgrid[z_i, :] < cash_imp[z_i, 0]
                k_policy_new[z_i, binding_indices] = -self.b           
            # lowest endogenous resource point (cash_imp[z_i, 0]) means 
            # the constraint is binding. cash_imp is unconstraiend since agrid[0] == -b
            # Identify points where we are poorer than the poorest unconstrained agent

            # Check convergence
            diff = np.linalg.norm(k_policy_new - k_policy)
            diff_list[i] = diff
            if diff < (np.linalg.norm(k_policy))*tol:
                self.policy = OptimalPolicy(cash_imp, self.agrid, self.b)
                if verbose:
                    print(f"Converged in {i} iterations")
                    print("Policy saved under self.policy!")
                return None
            k_policy = k_policy_new * lamb + k_policy * (1 - lamb)
        print("Max iterations reached")
        print(diff)
        print(f"required value: {(1+np.linalg.norm(k_policy))*tol}")
        return diff_list, constr
    
    def _solve_EGM_njit(self, r, w):
        
        agrid, cash_imp =_solve_EGM_numba(
            agrid = self.agrid,
            states = self.econ.states,
            Q = self.econ.Q,
            r = r,
            w = w,
            b = self.b,
            beta = self.econ.beta,
            sigma = self.sigma
            )
        return (OptimalPolicy(self.agrid, cash_imp, self.b))
        
    def _simulate_stationary_distribution(
            self,
            r,
            w,
            verbose=False,
            H=100, 
            T=500, 
            burn_in=200
        ):
        """
        Monte Carlo simulation of invariant distribution in Aiyagari model.
        """
            # 1. Setup
        nz = len(self.econ.states)
        
        # Initialize assets at random grid points
        a = np.random.choice(self.agrid, size=H) 
        
        # Initialize states (randomly across z)
        z_idx = np.random.choice(nz, size=H)
        
        # Pre-calculate Cumulative Probability for Markov transition
        Q_cumsum = np.cumsum(self.econ.Q, axis=1)
        K_series = []
        if verbose:
            print(f"Simulating {H} agents for {T} periods...")
    
        # 2. Simulation Loop
        for t in range(T):
            
            # Vectorized Markov Transition
            # This compares random draws against the CDF of the current row.
            r_draws = np.random.rand(H)
            
            # Method: for every agent i, find where r_draws[i] fits in Q_cumsum[z_idx[i]]
            # Vectorized trick: Summing booleans. 
            # (Broadcasting is tricky here, let's use a robust approach)
            
            # Look up the specific CDF row for each agent
            current_cdfs = Q_cumsum[z_idx] # Shape (H, nz)
            
            # Find the first index where CDF > random draw
            # argmax returns the first True index because True > False
            z_next = (current_cdfs > r_draws[:, None]).argmax(axis=1)
            z_idx = z_next
        
            # policy maps cash -> assets
            cash = (1+r)*a + w*self.econ.states[z_idx]
            a_prime = self.policy.simulate_policy(z_idx, cash) # or use self.policy.__call__() -> USP of __call__
    
            a = a_prime
        
            if t > burn_in:
                K_series.append(np.mean(a))
        
        # Return the final distribution AND the smoothed aggregate capital
        mean_K = np.mean(K_series) 
        return a, mean_K

def Emu(c, r, sigma, beta, Q):
    # for negative consumption select small value so that marginal value goes against infinity
    epsilon_clip = 1e-10 
    c_clipped = np.maximum(c, epsilon_clip)
    
    # Calculate Marginal Utility for each future state (c'^-sigma)
    C_z = c_clipped**(-sigma)
    C_z_scaled = C_z * (1+r) * beta
    return Q @ C_z_scaled

 @njit
 def interp_linear_1d(x_grid, y_grid, x_query):
     """
     Manual linear interpolation compatible with Numba.
     Maps x_query points to values based on (x_grid, y_grid).
     Assumes x_grid is sorted.
     """
     res = np.empty_like(x_query)
     # We use searchsorted (binary search) which is very fast
     indices = np.searchsorted(x_grid, x_query)
     
     for i in range(len(x_query)):
         idx = indices[i]
         
         # Handle extrapolation (bounds)
         if idx == 0:
             res[i] = y_grid[0]
         elif idx >= len(x_grid):
             res[i] = y_grid[-1]
         else:
             # Linear interpolation formula
             x0 = x_grid[idx - 1]
             x1 = x_grid[idx]
             y0 = y_grid[idx - 1]
             y1 = y_grid[idx]
             
             weight = (x_query[i] - x0) / (x1 - x0)
             res[i] = y0 + (y1 - y0) * weight
             
     return res
 @njit
 def _solve_EGM_numba(agrid, states, Q, r, w, b, beta, sigma, max_iter=1000, tol=1e-8, lamb = 0.7, verbose=False):
     na = len(agrid)
     nz = len(states)
     k_policy = np.zeros((nz, na))
     k_policy[:] = agrid * 0.5
     k_policy_new = np.zeros((nz, na))
     xgrid = comp_ressources(agrid, r, w, states)
     
     for i in range(max_iter):
         c_next = xgrid - k_policy
         # compute implied consumption today
         # compute C_t**-sigma = E(c-t+1**-sigma) * (R+beta)
         marginal_utility = Emu(c_next, r, sigma, beta, Q)
         # invert ot get c_today!
         c_today = (marginal_utility) ** (-1/sigma)
         # compute implied cash at hand
         cash_imp = c_today + agrid
         for z_i in range(nz):

             # Interpolate: resources_today -> capital_today
             policy_interp = interp1d(cash_imp[z_i, :], agrid, kind='linear',
                    bounds_error=False, fill_value="extrapolate") 
             k_policy_new[z_i, :] = policy_interp(xgrid[z_i, :])
             binding_indices = xgrid[z_i, :] < cash_imp[z_i, 0]
             k_policy_new[z_i, binding_indices] = -b
         # lowest endogenous resource point (cash_imp[z_i, 0]) means 
         # the constraint is binding. cash_imp is unconstraiend since agrid[0] == -b
         # Identify points where we are poorer than the poorest unconstrained agent

         # Check convergence
         diff = np.max(np.abs(k_policy_new - k_policy))
         if diff < (np.linalg.norm(k_policy))*tol:
             if verbose:
                 print(f"Converged in {i} iterations")
                 print("Policy saved under self.policy!")
             return agrid, cash_imp, b
         k_policy = k_policy_new * lamb + k_policy * (1 - lamb)
     print("Max iterations reached")
     print(diff)
     print(f"required value: {(1+np.linalg.norm(k_policy))*tol}")
     return diff_list, constr

    
def comp_ressources(agrid, r, w, states):
    return (1+r) * agrid[None, :] + w * states[:, None]


def construct_grid(b, max_state, na, w_ref=2, zeta=0.15):
    '''
    Return a power grid that approximately matches the target asset possibilites of the agents.
    Upper bound determined by 10 * maximum savings

    Parameters
    ----------
    b : TYPE
        DESCRIPTION.
    w : TYPE
        DESCRIPTION.
    max_state : TYPE
        DESCRIPTION.
    na : TYPE
        DESCRIPTION.
    zeta : TYPE, optional
        DESCRIPTION. The default is 0.15.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    a_max = 21 * w_ref * max_state 
    s = np.linspace(0, 1, na)
    zeta = 0.15
    return -b + (a_max + b) * s**(1/zeta)

        
class GE:
    def __init__(self, household):
        self.hh = household
        self.econ = household.econ
        

    def capital_demand(self, r):
        alpha = self.econ.alpha
        delta = self.econ.delta
        return (alpha / (r + delta))**(1/(1-alpha))

    def capital_supply(self, r):
        K_d = self.capital_demand(r)
        w = (1 - self.econ.alpha) * (self.econ.alpha / (r + self.econ.delta))**(self.econ.alpha / (1 - self.econ.alpha))
        self.hh._solve_EGM(r, w)
        self.a, K_s = self.hh._simulate_stationary_distribution(r, w, T = 5000, burn_in = 1500)
        # track
        self.history['r'].append(r)
        self.history['K_s'].append(K_s)
        self.history['K_d'].append(K_d)
        self.history['w'].append(w)
        return K_s

    def market_clearing(self, r):
        self.iter_count += 1
        print(f"Solving for r: {r:.10f}")
        K_d = self.capital_demand(r)
        K_s = self.capital_supply(r)
        return K_d - K_s
    
    @time_method
    def solve(self, r_min=None, r_max=None):
        self.iter_count = 0
        self.history = {'r':[], 'K_s':[], 'K_d':[], 'w':[]}
        eps = 1e-10
        if r_min is None:
            r_min = -self.econ.delta + eps
        if r_max is None:
            r_max = (1 / self.econ.beta) - 1 -eps
        r_star = opt.brentq(self.market_clearing, r_min, r_max, xtol = 1e-10)
        print(f"Brent finished after {self.iter_count} iterations.")
        return r_star
    
class OptimalPolicy:
    def __init__(self, cash, a_prime, b_constraint):
        """
        a_prime: The fixed grid of assets chosen for tomorrow (1D array)
        cash: The endogenous resource grid computed by EGM (nz, na)
        b_constraint: The borrowing limit (positive number, e.g., 0.0 or 2.0)
                      The constraint is a' >= -b_constraint
        """
        self.a_prime = a_prime
        self.nz = cash.shape[0]
        self.na = cash.shape[1]
        self.b_constraint = b_constraint
        self.interpolators = []
        if a_prime.ndim != 1:
            raise ValueError("a_prime (Y-axis) must be a 1D array (e.g., agrid).")
        for z_i in range(self.nz):
            # We map Resources (x) -> Assets (y)
            f = interp1d(
                cash[z_i, :],  # x: Endogenous resources
                a_prime,                 # y: Chosen assets
                kind='linear',
                bounds_error=False,
                fill_value="extrapolate"
            )
            self.interpolators.append(f) # check interpolation constraitn when applying interpolation

    def __call__(self, xgrid):
        """
        The standard call method: Looks up the policy for every point in a 2D Cash-on-Hand grid.
        This is primarily for TESTING, VISUALIZATION, and GRID LOOKUP.
        
        Parameters:
            xgrid: The (nz, na) Cash-on-Hand matrix you want to query.
                   
        Returns:
            a_prime_matrix: The resulting (nz, na) matrix of chosen assets (a').
        """
        if xgrid.shape != (self.nz, self.na):
            raise ValueError(f"Input xgrid must be of shape ({self.nz}, {self.na}).")
            
        # Flattening to 1d inputs
        cash_flat = xgrid.flatten()
        z_idx_flat = np.repeat(np.arange(self.nz), self.na)

        # internally call optimizer with grid values and corresponding states
        a_prime_flat = self.simulate_policy(z_idx_flat, cash_flat)

        # reshape into Shape (nz, na)
        a_prime_matrix = a_prime_flat.reshape(self.nz, self.na)
        
        return a_prime_matrix

    def simulate_policy(self, z_idx, current_cash):
        """
        Optimized method for Monte Carlo Simulation.
        Takes 1D vectors of agent states and cash-on-hand.
        
        Parameters:
            z_idx: 1D array of agent state indices (0, 1, 2, ...)
            current_cash: 1D array of agent cash-on-hand values.
            
        Returns:
            final_a_prime: 1D array of chosen assets (a').
        """
        # Ensure inputs are arrays, even if scalar
        current_cash = np.atleast_1d(current_cash)
        z_idx = np.atleast_1d(z_idx)
        a_prime = np.zeros_like(current_cash, dtype=np.float64)

        # Vectorized Interpolation
        for z in range(self.nz):
            mask = (z_idx == z)
            
            if np.any(mask):
                # Interpolate cash for agents in state z
                a_prime[mask] = self.interpolators[z](current_cash[mask])

        # Enforce Constraint!
        final_a_prime = np.maximum(a_prime, -self.b_constraint)
        
        # Return item if original input was scalar, else return array
        return final_a_prime.item() if final_a_prime.size == 1 and z_idx.size == 1 else final_a_prime

    
# work in progess, solving the aiyagari model with Howard's improvement Algorithm and sparse matrices
class test: # WiP
    '''
    Solving and simmulating a depreciation ramsey problem in an aggregate uncertainty environment.
    '''
    def __init__(self, gamma, beta, alpha, delta, agrid, state, Q):
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
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.delta = delta
        self.agrid = agrid
        self.na = len(agrid)
        self.state = state
        self.nz = len(state)
        self.Q = Q


    def H_PFI(self, agrid, w, r, tol=1e-10, max_iter=1000):
        '''
        '''
        na = len(agrid)
        nz = self.nz
        policy_idx = np.zeros((na, nz), dtype = int)
        agrid_2d  = agrid[:, None]
        print("agrid_2d shape:", agrid_2d.shape)
        eps_states = self.state[None, :]
        cash = (1 + r) * agrid_2d  + w * eps_states
        print("cash shape:", cash.shape)
        V = np.zeros((na, nz))

        #a_idx_flat = np.repeat(np.arange(na), nz)
        z_idx_flat = np.tile(np.arange(nz), na)
                
        print("Start iteration")
        for it in range(max_iter):
            a_prime = np.take(agrid, policy_idx)   # shape (na, nz)
            cons = cash - a_prime
            u_pi = utility_CRRA(cons, self.gamma)  # 2D utility grid
            print(np.min(u_pi))
            try:
                V_flat = self._solve_sparse_Q(u_pi, policy_idx, z_idx_flat)
                print("Linear solve successful")
                print("Printing linear solver")
                print(V_flat)
            except Exception as e:
                print(f"Linear solve FAILED: {e}")
            V = V_flat.reshape((na, nz), order='F')  # back to 2D
            print("V shape:", V.shape)
            print("vlaue")
            print(V)
            print("unique policy_idx:", np.unique(policy_idx))

            phi = cash[:, None, :] - agrid[None, :, None] + self.beta * V[None, :, :]
            print("phi shape:", phi.shape)

            policy_new = np.argmax(phi, axis=1).squeeze()
            #if np.array_equal(policy_new, policy_idx):
            if np.array_equal(policy_new, policy_idx):
                return V, np.take(agrid, policy_idx)
            else:
                policy_idx = policy_new
        print("Did not converge")
        print(policy_new.dtype, policy_idx.dtype)

        return policy_idx, policy_new
    
    def _construct_sparse_Q(self, policy, z_idx_flat):
        # now construct the Q_pi matrix
        policy_flat = policy.ravel(order='F')

        # next state indices for all possible next shocks
        next_states = policy_flat[:, None] + np.arange(self.nz)[None, :] * self.na
        assert np.all(next_states < self.na * self.nz)
        assert np.all(next_states >= 0)

        # probabilities
        probs = self.Q[z_idx_flat, :] 
        
        # build COO
        rows = np.repeat(np.arange(self.na*self.nz), self.nz)
        cols = next_states.ravel()
        data = probs.ravel()
        
        return csr_matrix((data, (rows, cols)), shape=(self.na*self.nz, self.na*self.nz))
        
        
    def _solve_sparse_Q(self, u_pi, policy, z_idx_flat):
        # Solve linear system
        u_flat = u_pi.ravel(order='F')
        Q_pi = self._construct_sparse_Q(policy, z_idx_flat)
        print(Q_pi)
        
        return spsolve(identity(self.na*self.nz) - self.beta * Q_pi, u_flat)
