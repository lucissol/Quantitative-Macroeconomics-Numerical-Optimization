# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 12:45:50 2025

@author:
Louis Brune - 8514718
João Cordeiro - 8509509
Luca Seibert - 8511270 
"""
#%% Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.optimize import newton

#%% set key parameters
beta = 0.99
alpha = 0.3
na = 500 # fully discretized risk (symmetric "need only one na")
# compute steady state
k_ss = (alpha/beta)**(1/1-alpha)
k_min = 10e-10 # avoid points close to zero to avoid interpolation issues of the log utility (the value function becoming to steep) and with the log grid
k_max = 1.5 * k_ss # upper bound
print(f"Using the steady state value: {k_ss}")
agrid = np.linspace(k_min, k_max, na) # asset grid
V0 = np.zeros(len(agrid)) # uninformative value guess
#%% part a) analytical solutions
a_ana = alpha*beta * agrid**alpha
b_true = alpha/(1.0 - alpha*beta) # just to plot V(k)
a_true = (np.log(1.0 - alpha*beta) + beta*b_true*np.log(alpha*beta)) / (1.0 - beta) # just to plot V(k)
V_ana = a_true + b_true*np.log(agrid) 
#%% part b)
def util(x):
    '''
    Parameters
    ----------
    x : Matrix
        Apply log utility on x.

    Returns
    -------
    TYPE
        Log utility for values larger 1e-12.

    '''
    return np.where(x > 1e-12, np.log(x), np.log(1e-12)) # allowing for matrix input

def get_logutil_invar(a_current, a_next, alpha):
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
    c = a_current[:, None]**alpha - a_next[None, :]

    # mask out negative consumption
    utility = np.full_like(c, -1e10, dtype=float)
    positive_mask = c > 0
    utility[positive_mask] = util(c[positive_mask])
    return utility


def ValueFiter(Vguess, beta, alpha, a_current, a_next, vectorized=True, tol = 10e-9, maxit = 10000):
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
    maxit : Int, optional
        Maximum number of iterations. The default is 10000.

    Raises
    ------
    ValueError
        Check asset space.

    Returns
    -------
    Vn : float
        Value Function.
    a : float
        Policy Function.
    time_s : float
        Full time call to return.
    it : float
        Number of iterations needed.
    '''
    start = time.perf_counter()
    # Value Function Iteration for two dimensional asset grid
    V = Vguess.copy() # copy in the first guess
    u = get_logutil_invar(a_current, a_next, alpha)
    #robustness
    if len(V) != len(a_current):
        raise ValueError("Length of value function and choice grid does not match")
    
    if vectorized == False: # If Value Function solved in for loop, preallocate objects
        phi = np.zeros((len(a_current),len(a_current)))
        a = np.zeros(len(a_current))
        Vn = np.zeros(len(a_current))
        
    print("Starting Iteration...")
    for it in range(maxit):
        if vectorized: 
            phi = u + beta*V[np.newaxis, :] # matrix with full values of dimension n x nprime
            a_idx = np.argmax(phi, axis=1) # take max out of each row and store index
            a = np.take_along_axis(a_next, a_idx, axis=-1) # go through each axis and store index of max value 
            # go through each row and store max value indexed by opt policy
            Vn = np.take_along_axis(phi, a_idx[:,np.newaxis], axis=-1).squeeze()
        else: # for loop
            for i in range(len(a_current)):
                phi_i   = u[i,:] + beta*V
                a_idx   = np.argmax(phi_i)
                a[i]    = agrid[a_idx]
                Vn[i]   = phi_i[a_idx]
                
        #check convergance criterium
        if np.linalg.norm(Vn - V) < (1 + np.linalg.norm(V)) * tol:
            end = time.perf_counter()
            time_s = end - start
            print(f"..Converged! in {time_s} seconds\n")
            print(f"Number of iterations: {it}")
            return Vn, a, time_s, it
        else: #keep iterating
            V = Vn.copy() # update guess
#%% part a) iterate and store values
###-! Storing Values !-###
VF = ValueFiter(V0, beta, alpha, agrid, agrid)
VF_slow = ValueFiter(V0, beta, alpha, agrid, agrid, False)
#%% plot results for a)
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.title("Value Function")
plt.plot(agrid, V_ana, label="Analytical solution", linestyle="-", color="orange")
plt.plot(agrid, VF[0], label="VFI", linestyle="--", color="blue")
plt.xlabel('Assets')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.title("Policy Function")
plt.plot(agrid, a_ana, label="Analytical solution", linestyle="-", color="orange")
plt.plot(agrid, VF[1], label="VFI",linestyle="--", color="blue")
plt.xlabel('Assets')
plt.ylabel('Optimal policy')
plt.grid(True)
plt.legend()
#%% c) & d)
###-! preparing part c) and d) !-###
# defining log grid
agrid_log = np.exp(np.linspace(np.log(k_min), np.log(k_max), na))

# Power-spaced grid
s = np.linspace(0, 1, na)
zeta = 0.15
agrid_pow = k_min + (k_max - k_min) * s**(1/zeta)
#%% part c)
VF_log = ValueFiter(V0, beta, alpha, agrid_log, agrid_log)
#%% part d)
VF_pow = ValueFiter(V0, beta, alpha, agrid_pow, agrid_pow)
#%% compare grids
sns.scatterplot(agrid, label='Equidistant Grid')
sns.scatterplot(agrid_log, label='Log Grid')
sns.scatterplot(agrid_pow, label=f'Power Grid (zeta = {zeta})')
plt.xlabel('Grid Index')
plt.ylabel('Assets choices')
plt.legend()
plt.show()
# computing differences from the analytical solution
mean_a_log = np.abs(np.mean(VF_log[1] - a_ana))
mean_a_equi = np.abs(np.mean(VF[1] - a_ana))
mean_a_pow = np.abs(np.mean(VF_pow[1] - a_ana))
max_a_log= np.abs(np.max(VF_log[1] - a_ana))
max_a_equi = np.abs(np.max(VF[1] - a_ana))
max_a_pow = np.abs(np.max(VF_pow[1] - a_ana))
print("Deviation from analytical solution \n")
print(f"Equidistant grid \n mean:{mean_a_equi}, max: {max_a_equi}")
print(f"Log grid \n mean:{mean_a_log}, max: {max_a_log}")
print(f"Power grid \n mean:{mean_a_pow}, max: {max_a_pow}")
#%% Value function comparison of the three grid methods
plt.figure(figsize=(10, 6))
plt.plot(agrid, VF[0], label='Equidistant Grid', color='green', linestyle='-')
plt.plot(agrid_log, VF_log[0], label='Log Grid', color='blue', linestyle='--')
plt.plot(agrid_pow, VF_pow[0], label='Power Grid', color='red', linestyle=(0, (5, 2, 1, 2)))

plt.title('Comparison of Asset Grids')
plt.xlabel('Index')
plt.ylabel('Value function')
plt.legend()
plt.grid(True)
plt.show()
#%% Value function comparison of the three grid methods (different plotting style)
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(agrid, VF[0], label='Linear Grid')
plt.title('Value Function (Equidistant Grid)')
plt.xlabel('Assets')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(agrid_log, VF_log[0], label='Log Grid', color='orange')
plt.title('Value Function (Log Grid)')
plt.xlabel('Assets')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(agrid_pow, VF_pow[0], label='Log Grid', color ='green')
plt.title('Value Function (Power Grid)')
plt.xlabel('Assets')
plt.grid(True)

plt.tight_layout()
plt.show()
#%% Policy function comparison of the three grid methods
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(agrid, VF[1], label='Equidistant',linewidth=2, color="blue")
plt.plot(agrid, a_ana, label='Analytical',linewidth=1, linestyle="-", color="orange")
plt.title('Equidistant Grid')
plt.xlabel('Assets')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(agrid_log, VF_log[1], label='Log',linewidth=2, color="blue")
plt.plot(agrid, a_ana, label='Analytical', linewidth=1, linestyle="-", color="orange")
plt.title('Log Grid')
plt.xlabel('Assets')
plt.grid(True)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(agrid_pow, VF_pow[1], label='Power',linewidth=2, color="blue")
plt.plot(agrid, a_ana, label='Analytical',linewidth=1, linestyle="-", color="orange")
plt.title('Power Grid')
plt.xlabel('Assets')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
#%% About monotonicity and concavity
# Idea: once we have found the optimal k' for k we can restrict for higher k the search by starting with k' of the lower k
# Concavity: Decreasing value function indicates that we overshoot the maximum. Upper bound!
def VFiter_NKGM(Vguess, beta, alpha, a_current, a_next, tol = 10e-9, maxit=10000):
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
    maxit : Int, optional
        Maximum number of iterations. The default is 10000.

    Raises
    ------
    ValueError
        Check asset space.

    Returns
    -------
    Vn : float
        Value Function.
    a_policy : float
        Policy Function.
    time_s : float
        Full time call to return.
    it : float
        Number of iterations needed.
    '''
    
    start = time.perf_counter()
    V = Vguess.copy() # copy in the first guess
    u = get_logutil_invar(a_current, a_next, alpha)
    # robustness
    if len(V) != len(a_current):
        raise ValueError("Length of value function and choice grid does not match")
    # storing values
    phi = np.zeros_like(u)
    Vn = np.zeros_like(V)
    a_policy = np.zeros_like(a_current)

    print("Starting Iteration...")
    for it in range(maxit):
        lower_bound = 0 # for first iteration of j (choice grid) = 0 (use full grid)
        for i in range(len(a_current)):
            for j in range(lower_bound, len(a_next)):
                phi[i, j] = u[i, j] + beta*V[j] # update value function
                if j > lower_bound and phi[i, j] < phi[i, j - 1]: # check whether value function is declining
                    break # break into next current capital value iteration
            # maximize in the optimal search radius where we expect the solution to be in
            j_max = np.argmax(phi[i, lower_bound:j+1]) + lower_bound # add lower_bound to adjust for the smaller search area of the index
            a_policy[i] = a_next[j_max] # save policy 
            Vn[i] = phi[i, j_max] # save value 
            # Update search bound for monotonicity exploit in next iteration
            lower_bound = j_max
        #check convergance criterium
        if np.linalg.norm(Vn - V) < (1 + np.linalg.norm(V)) * tol:
            end = time.perf_counter()
            time_s = end - start
            print(f"..Converged! in {time_s} seconds\n")
            print(f"Number of iterations: {it}")
            return Vn, a_policy, time_s, it
        else: #keep iterating
            V = Vn.copy() # update guess

#%% e) compare runtime
VF_2 = VFiter_NKGM(V0, beta, alpha, agrid, agrid)
print(f"Comparing Runtimes \n VFI: {VF[2]}\n VFI_exploited: {VF_2[2]}\n VFI_loop: {VF_slow[2]}\n ")

#%% next part Howards improvement
def PFIhoward(Vguess, beta, alpha, a_current, a_next, tol = 10e-9, policy_iterate = 1500, max_iter = 1000):
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
    policy_iterate : Int, optional
        Number of iterations on the policy. The default is 1500.
    max_iter : Int, optional
        Maximum number of iterations. The default is 10000.

    Raises
    ------
    ValueError
        Check asset space.

    Returns
    -------
    V : float
        Value function.
    a_opt : float
        Policy function.
    time_s : float
        Full time call to return.
     it : float
         Number of iterations needed.
     '''
     
    start = time.perf_counter()
    # Value Function Iteration for two dimensional asset grid
    V = Vguess.copy() # copy in the first guess
    u = get_logutil_invar(a_current, a_next, alpha)
    policy_old = np.ones(len(a_current)) * 0.5 # non-integer to avoid convergance after first iteration
    
    if len(V) != len(a_current):
        raise ValueError("Length of value function and choice grid does not match")
        
    print("Starting Iteration...")
    for it in range(max_iter):
        phi = u + beta*V[np.newaxis, :] # matrix with full values of dimension n x nprime
        policy_new = np.argmax(phi, axis=1) # take max out of each row and store optimal policy index
        for pol_iter in range(policy_iterate):
            # iterate on the policy for policy_iterate times
            # update policy iteratively by computing utility value of optimal policy today + discounted next period of sticking to this policy
            V_policy = u[np.arange(len(a_current)), policy_new] + beta * V[policy_new]
            # update new Value function and iterate
            V = V_policy.copy()
            
        #check convergance criterium for policy
        if np.array_equal(policy_new, policy_old):
            end = time.perf_counter()
            time_s = end - start
            print(f"..Converged! in {time_s} seconds\n")
            print(f"Number of iterations: {it}")
            # store policy function
            a_opt = a_next[policy_new]
            return V, a_opt, time_s, it # return value function and optimal policy
        
        else: #keep iterating
            policy_old = policy_new.copy() # update guess

#%% Howards improvement with direct solution of the linear system
# look up sparse matrices to aoid many computations of zeros
def PFIhoward_exact(Vguess, beta, alpha, a_current, a_next, tol = 1e-9, max_iter = 1000):
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
    # Howard's improvement solving the linear system of equations directly
    u = get_logutil_invar(a_current, a_next, alpha) # invariant utility
    V = Vguess.copy() # copy in the first guess
    n = len(a_current)
    nprime = len(a_next)
    policy_old = np.ones(len(a_current)) * 0.5 # non-integer to avoid convergance after first iteration (e.g., optimal policy zero for entire grid)

    if u.shape != (len(a_current), len(a_next)): # compare utility grid with asst grid
        raise ValueError("Utility grid should be of shape (N, N) for asset grid of length N.")
    if len(V) != len(a_current):
        raise ValueError("Length of value function and choice grid does not match")
        
    print("Starting Iteration...")
    for it in range(max_iter):
        phi = u + beta*V[np.newaxis, :] # matrix with full values of dimension n x nprime
        policy_new = np.argmax(phi, axis=1) # take max out of each row and store optimal policy index
        # instead of iterating on the policy we follow Heer & Maußner and solve a system of equations instead
        # transition matrix
        # solve linear sysstem instead of iterating
        Q = np.zeros((n, nprime)) # transition matrix
        # update for each row i the corresponding optimal policy 
        Q[np.arange(n), policy_new] = 1  # Deterministic transition
        # extract optimal utility for the policy values of interest to ease matrix multiplication
        u_policy = u[np.arange(n), policy_new] # in line with transition matrix Q
        # invert directly now
        # use direct solvers
        V = np.linalg.inv(np.eye(n) - beta * Q) @ u_policy  # as if we iterated on the optimal policy forever!

        #check convergance criterium for policy
        if np.array_equal(policy_new, policy_old):
            end = time.perf_counter()
            time_s = end - start
            print(f"..Converged! in {time_s} seconds\n")
            print(f"Number of iterations: {it}")
            # store policy function
            a_opt = a_next[policy_new]
            return V, a_opt, time_s, it # return value function and optimal policy
        else: #keep iterating
            policy_old = policy_new.copy() # update guess
#%% compute howard's improvement
Howard1 = PFIhoward(V0, beta, alpha, agrid, agrid)
Howard_exact = PFIhoward_exact(V0, beta, alpha, agrid, agrid)
#%% part f) compare runtimes of all methods from part 1
print(f"Comparing Runtimes\n VFI: {VF[2]}\n VFI_exploited: {VF_2[2]}\n VFI_loop: {VF_slow[2]}\n Howard: {Howard1[2]}\n Howard_exact: {Howard_exact[2]}\n ")
print(f"Comparing Iterations \n VFI: {VF[3]}\n VFI_exploited: {VF_2[3]}\n VFI_loop: {VF_slow[3]}\n Howard: {Howard1[3]}\n Howard_exact: {Howard_exact[3]}\n ")
start_util = time.perf_counter()
get_logutil_invar(agrid, agrid, alpha)
end_util = time.perf_counter()
print(f"Computing the invariant utility matrix requires {end_util-start_util} s")
#%% Question 2

#%% Time Iteration
#The idea is that if I save tomorrow that much I should save today this much. Next iteration we have an updated policy. This is the same as VFI, just for the policy going backwards on the euler

# necessary functions
def f(k):
    k_safe = np.maximum(k, 1e-12)
    return k_safe**alpha
def inv_f(k):
    k_safe = np.maximum(k, 1e-12)
    return k_safe**(1/alpha)
def f_prime(k):
    k_safe = np.maximum(k, 1e-12)
    return alpha * k_safe**(alpha-1)
def u_prime(c):
    c_safe = np.maximum(c, 1e-12)
    return 1/c_safe
# utility function already defined


# transforming the euler equation into a root finding problem
def time_iter(savings_guess, beta, alpha, a_current, tol = 10e-9, max_iter = 1000): 
    '''
    Time iteration on the euler equation to extract optimal polic given a choice grid.
    Using cubic interpolation with newton's method for root findig
    
    Parameters
    ----------
    savings_guess : float
        Constant savings guess. Should be between 0 and 1
    beta : float
        Parameter.
    alpha : float
        Parameter.
    a_current : float
        Asset state space.
    tol : float, optional
        Tolerance level for Time Iteration & Newton Root Finder. The default is 10e-9.
    max_iter : TYPE, optional
        Maximum number of iterations. The default is 1000.

    Returns
    -------
    g_new : float
        Policy Function.
    time_s : float
        Full time call to return.
    it : float
        Number of iterations needed.
    '''
    
    start = time.perf_counter()
    g_old = savings_guess * a_current**alpha
    
    print("Starting Iteration...")
    for it in range(max_iter):
        g_new = np.zeros_like(a_current) # reset next period's optimal asset choice
        # update interpolation function mapping state space into optimal asset next period with policy guess
        g_interp_cubic = interp1d(a_current, g_old, kind='cubic',
                         bounds_error=False, fill_value='extrapolate')
        
        for i, a in enumerate(a_current):
            def iteration(k_prime):
                #interpolate grid for next period's policy function
                policy_tomorrow = g_interp_cubic(k_prime) # we need to interpolate so that whatever vlaue our root finder analyzes we have a value for the next period's policy
                LHS = u_prime(f(a) - k_prime)
                RHS = beta*u_prime(f(k_prime)-policy_tomorrow)*f_prime(k_prime)
                return RHS - LHS #transforming euler equation into a root finding problem

            # Solve for optimal policy
            initial_guess = g_old[i] if g_old[i] < k_max else 0.5 * k_max
            # find k_prime that returns RHS - LHS = 0 (approximately)
            kp_optimal = newton(iteration, initial_guess, 
                  fprime=None,  # Let Newton approximate derivative
                  tol=tol, maxiter=50)
            g_new[i] = kp_optimal
        
        # Check convergence
        error = np.max(np.abs(g_new - g_old))        
        if error < tol:
            end = time.perf_counter()
            time_s = end - start
            print(f"..Converged! in {time_s} seconds\n")
            print(f"Number of iterations: {it}")
            return g_new, time_s, it
        
        # Update for next iteration
        g_old = g_new.copy()
    return g_new
        
#%% a) compare time iteration with analytical solution
agrid_2 = np.linspace(k_min, k_max, 200) # equidistant grid with 200 observations
a_ana_2 = alpha*beta * agrid_2**alpha # analytical solution defined on the new and smaller grid
#%%
savings_guess = 0.1
policy_time = time_iter(savings_guess, beta, alpha, agrid_2)
#%%
plt.plot(agrid_2, a_ana_2, label="Analytical solution",linestyle="-", color="blue")
plt.plot(agrid_2, policy_time[0], label="Time Iteration", linestyle="--", color="orange")
plt.xlabel("Assets")
plt.ylabel("Optimal policy")
plt.grid()
plt.legend()
plt.show()
print(f"Maximum deviation from analytical solution for time iteration policy: {np.abs(np.max(policy_time[0]-a_ana_2))}")
print(f"Maximum deviation from analytical solution for standard VFI: {np.abs(np.max(VF[1]-a_ana))}")
#%%
# transforming the euler equation into a fixed point iteration
def FPIter(savings_guess, beta, alpha, a_current, damping, tol=10e-9, max_iter=1000):
    '''
    Parameters
    ----------
    savings_guess : float
        Constant savings guess. Should be between 0 and 1
    beta : float
        Parameter.
    alpha : float
        Parameter.
    a_current : float
        Asset state space.
    damping : Int
        Dampening parameter for Fixed Point Iteration. Should be between 0 and 1.
    tol : float, optional
        Tolerance level for Time Iteration & Newton Root Finder. The default is 10e-9.
    max_iter : TYPE, optional
        Maximum number of iterations. The default is 1000.
        
    Returns
    -------
    g_new : float
        Policy Function.
    time_s : float
        Full time call to return.
    it : float
        Number of iterations needed.
    '''
    start = time.perf_counter()
    # get stable values for iterations
    a_current_safe = np.maximum(a_current, 1e-12)
    g_old = np.clip(savings_guess * a_current_safe**alpha, 1e-12, None)
    assert 0 < damping <= 1, "damping must be between 0 and 1"
    
    print("Starting Iteration...")
    for it in range(max_iter):
        g_new = np.zeros_like(a_current)
        # using linear splines here - update interpolation function
        g_interp_linear = interp1d(a_current, g_old, kind='linear', 
                               bounds_error=False, fill_value='extrapolate')
        # map x into y as a policy! so that for any x_tilde we get y_tilde in the same style
        
        for i, a in enumerate(a_current):
            #interpolate grid for next period's policy function
            a_prime = g_old[i]
            a_double_prime = g_interp_linear(a_prime) # a_prime is likely not on our grid anymore!
            
            # returning closed form solution step for step
            denominator = (beta*u_prime(f(a_prime) - a_double_prime)*f_prime(a_prime))
            denominator = np.clip(denominator, 1e-12, 1e12)
            closedform_g = f(a) - 1 / denominator
            
            # update values assuring for stable division
            max_feasible = f(a) * 0.999999  # cannot save more than production
            closed_form_update = np.clip(closedform_g, 1e-12, max_feasible)
            g_new[i] = damping * g_old[i] + (1-damping) * closed_form_update
            
        
        # Check convergence
        max_diff = np.max(np.abs(g_new - g_old))
        if it % 100 == 0:  # Added progress monitoring
            print(f"Iteration {it}, max difference: {max_diff:.2e}")
            
        if max_diff < tol:
            end = time.perf_counter()
            time_s = end - start
            print(f"..Converged! in {time_s} seconds\n")
            print(f"Number of iterations: {it}")
            return g_new, time_s, it
        else:
             # Update for next iteration
             g_old = g_new.copy()
    print(f"Warning: Failed to converge after {max_iter} iterations")
    print(f"Final max difference: {max_diff:.2e}")
    return g_new
#%%
fpi = FPIter(savings_guess, beta, alpha, agrid_2, damping=0.7)
#%%
plt.plot(agrid_2, a_ana_2, label="Analytical solution",linestyle="-", color="blue")
plt.plot(agrid_2, fpi[0], label="Fixed Point Iteration", linestyle="--", color="orange")
plt.xlabel("Assets")
plt.ylabel("Optimal policy")
plt.grid()
plt.legend()
plt.show()
print(f"Maximum deviation from analytical solution for fixed point iteration policy: {np.abs(np.max(fpi[0]-a_ana_2))}")
#%%
# we start with a policy guess and  a k prime grid. Then, we ask what value of k would let us chooce the k prime grid. We update the policy and use the  policy to compute k prime prime with our k prime grid?
# So we guess a policy, compute kprime and prime prime, and optimize to get the k today. for k_today we have a policy that maps k to k prime. This is what we want to interpolate to in the next iteration assigning k prime prime to k prime
def EGM_iter(savings_guess, beta, alpha, a_next, tol=10e-9, max_iter=1000 ):
    '''
    Endogenous grid method. using Linear Splines to interpolate Policy.
    Parameters
    ----------
    savings_guess : float
        Constant savings guess. Should be between 0 and 1
    beta : float
        Parameter.
    alpha : float
        Parameter.
    a_next : float
        Asset state space next period from which we revert back to today's assets.
    tol : float, optional
        Tolerance level for Time Iteration & Newton Root Finder. The default is 10e-9.
    max_iter : TYPE, optional
        Maximum number of iterations. The default is 1000.

    Returns
    -------
    policy_function : float
            Interpolation function that maps state space into optimal next period's asset holding. Cubic splines
    time_s : float
        Full time call to return.
    it : float
        Number of iterations needed.
    '''
    start = time.perf_counter()
    # we need a policy guess
    a_next_safe = np.maximum(a_next, 1e-12)
    g_old = np.clip(savings_guess * a_next_safe**alpha, 1e-12, None)
    # first guess for kprimep
    print("Starting Iteration...")
    for it in range(max_iter):
        # update policy for last period
        g_new = np.zeros_like(a_next)
        # update interpolation function
        g_interp_linear = interp1d(g_old, a_next, kind='linear', 
                               bounds_error=False, fill_value='extrapolate')
        
        for j, a_prime in enumerate(a_next):
            # get a_double_prime given the policy_new
            a_double_prime = g_interp_linear(a_prime)
            # computing consumption in closed form (for log utility!)
            c_next = f(a_prime) - a_double_prime
            c = c_next / (beta * f_prime(a_prime))
            # for the closed form solution we must invert the production function
            f_a = a_prime + c
            g_new[j] = inv_f(f_a)
          
        max_diff = np.max(np.abs(g_new - g_old))
        if it % 100 == 0:  # progress monitoring
            print(f"Iteration {it}, max difference: {max_diff:.2e}")
            
        if max_diff < tol:
            end = time.perf_counter()
            time_s = end - start
            print(f"..Converged! in {time_s} seconds\n")
            print(f"Number of iterations: {it}")
            # return interpolated policy -> use cubic splines since we are out of iteration
            # and can invest a bit more time to return a smoother policy
            policy_function = interp1d(g_new, a_next, kind='cubic',
                         bounds_error=False, fill_value='extrapolate')
            return policy_function, time_s, it
        else:
             # Update for next iteration
             g_old = g_new.copy()
#%% question 2 d)
EGM_policy = EGM_iter(savings_guess, beta, alpha, agrid_2, tol=10e-9, max_iter=1000)
#carefull! EGM_iter returns a interpolated policy function! Use with EGM_policy[0](grid) to get grid mapping
#%%
plt.plot(agrid_2, a_ana_2, label="Analytical solution",linestyle="-", color="blue")
plt.plot(agrid_2, EGM_policy[0](agrid_2), label="EGM", linestyle="--", color="orange")
plt.xlabel("Assets")
plt.ylabel("Optimal policy")
plt.grid()
plt.legend()
plt.show()
print(f"Maximum deviation from analytical solution for EGM policy: {np.abs(np.max(EGM_policy[0](agrid_2)-a_ana_2))}")
#%% Euler error
# We've computed all except the euler error so lets write a function for it
def EulerError(policy_a, a_current, beta, alpha, interpolating = True):
    '''
    Parameters
    ----------
    policy_a : TYPE
        Default: optimal next year capital.
        If interpolating False: interpolated policy function that maps a to a_prime 
    a_current : TYPE
        Capital grid for today's state value.
    beta : TYPE
        Discount factor.
    alpha : TYPE
        Model parameter.
    interpolating:
        if False computes interpolated policy function with cubic splines.
    
    Returns
    -------
    Mean and Max Euler Equation Error.
    '''
    if interpolating == True:
        if len(policy_a) != len(a_current):
            raise ValueError("Dimensions of policy mapping wrong!")
        policy_fct = interp1d(a_current, policy_a, kind='cubic',
                 bounds_error=False, fill_value='extrapolate')
    else:
        policy_fct = policy_a

    # euler for log utility
    c = f(a_current) - policy_fct(a_current)
    c_next = f(policy_fct(a_current)) - policy_fct(policy_fct(a_current))
    c_implied = c_next / (beta * f_prime(policy_fct(a_current)))
    
    # euler error
    euler_error = np.abs((c - c_implied)/c_implied)
    mean_EE = np.mean(euler_error)
    max_EE = np.max(euler_error)
    return mean_EE, max_EE

#%% Comparison of methods | Speed & Accuracy d)
# dataframe to store results in
results = pd.DataFrame({
    'Method': ['VFI', 'TI', 'FPI', 'EGM'],
    'Time (s)': [0.0] * 4,
    'Iter': [0] * 4,
    'MeanEE': [0.0] * 4,
    'MaxEE': [0.0] * 4
})

# we computed already everything by now, so lets write into the dataframe
# call everything again for cleanless and overview
a_gridVF = np.linspace(k_min, k_max, 2000)
a_gridEE = np.linspace(k_min, k_max, 200)
# for the guesses we use solid ones for both methods to treat them fairly (I hope)
V_guess = util(f(a_gridVF) - a_gridVF) / (1-beta) # naive guess
policy_saving_guess = 0.1 # naive guess -> save 10% of output / consume the rest
#%%
print("Applying all computational methods | including slow version of VFI. Expected runtime 30 seconds!")
VFI = ValueFiter(V_guess, beta, alpha, a_gridVF, a_gridVF, tol = 10e-9, maxit=10000)
VFI_0guess = ValueFiter(np.zeros(len(a_gridVF)), beta, alpha, a_gridVF, a_gridVF, tol = 10e-9, maxit=10000)
TI = time_iter(policy_saving_guess, beta, alpha, a_gridEE, tol=10e-9, max_iter=1000)
FPI = FPIter(policy_saving_guess, beta, alpha, a_gridEE, damping=0.7, tol=10e-9, max_iter=1000)
EGM = EGM_iter(policy_saving_guess, beta, alpha, a_gridEE, tol=10e-9, max_iter=1000)
#%% compute euler errors
VFIEE = EulerError(VFI[1], a_gridVF, beta, alpha, interpolating = True)
TIEE = EulerError(TI[0], a_gridEE, beta, alpha, interpolating = True)
FPIEE = EulerError(FPI[0], a_gridEE, beta, alpha, interpolating = True)
EGMEE = EulerError(EGM[0], a_gridEE, beta, alpha, interpolating = False) # already have interpolated policy for EGM!
#%% Update the results DataFrame
results.loc[results['Method'] == 'VFI', 'Time (s)'] = VFI[2]
results.loc[results['Method'] == 'VFI', 'Iter'] = VFI[3]
results.loc[results['Method'] == 'VFI', 'MeanEE'] = VFIEE[0]
results.loc[results['Method'] == 'VFI', 'MaxEE'] = VFIEE[1]

results.loc[results['Method'] == 'TI', 'Time (s)'] = TI[1]
results.loc[results['Method'] == 'TI', 'Iter'] = TI[2]
results.loc[results['Method'] == 'TI', 'MeanEE'] = TIEE[0]
results.loc[results['Method'] == 'TI', 'MaxEE'] = TIEE[1]

results.loc[results['Method'] == 'FPI', 'Time (s)'] = FPI[1]
results.loc[results['Method'] == 'FPI', 'Iter'] = FPI[2]
results.loc[results['Method'] == 'FPI', 'MeanEE'] = FPIEE[0]
results.loc[results['Method'] == 'FPI', 'MaxEE'] = FPIEE[1]

results.loc[results['Method'] == 'EGM', 'Time (s)'] = EGM[1]
results.loc[results['Method'] == 'EGM', 'Iter'] = EGM[2]
results.loc[results['Method'] == 'EGM', 'MeanEE'] = EGMEE[0]
results.loc[results['Method'] == 'EGM', 'MaxEE'] = EGMEE[1]

# beauty treatment
results_formatted = results.copy()
results_formatted['Time (s)'] = results_formatted['Time (s)'].round(4)
results_formatted['MeanEE'] = results_formatted['MeanEE'].apply(lambda x: f"{x:.2e}")
results_formatted['MaxEE'] = results_formatted['MaxEE'].apply(lambda x: f"{x:.2e}")

print("Comparison Results:")
print(results_formatted.to_string(index=False))
print(f"Note that VFI with a zero initial value guess needs {VFI_0guess[3]} iterations and {VFI_0guess[2]}s!")
#%% e) comparing the EGM method with a log grid
a_gridEE_log = np.exp(np.linspace(np.log(k_min), np.log(k_max), 200))
EGM_log = EGM_iter(policy_saving_guess, beta, alpha, a_gridEE_log, tol=10e-9, max_iter=1000)
EGMEE_log = EulerError(EGM_log[0], a_gridEE, beta, alpha, interpolating = False)
#%%
print(EGMEE, EGMEE_log)
# drastic improvement
