# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 17:27:48 2025

@author: lseibert
"""
#%%
import numpy as np
from scripts.functions import ValueFunctionMethods
from scripts.stochastic_euler import discretize_AR1
from scripts.stochastic_euler import agg_uncertainty
import matplotlib.pyplot as plt
from scripts.utility import utility_CRRA
from scripts.aiyagari import *

#%%
beta = 0.96
alpha = 0.36
delta = 0.08
sigma = 1
rho = 0.9
b = 0
sigma_eps = 0.1743
mu_eps = (- sigma_eps / (2 * (1-rho**2))) 
mu_eps_x = mu_eps * (1- rho)
eps_process = discretize_AR1(rho, mu_eps_x, sigma_eps)
na = 4

#%% a) setting up the grids
eps_grid_log, eps_Q = eps_process.Rouwenhorst(2)
eps_grid = np.exp(eps_grid_log)
print(eps_grid)
#%%
agrid = construct_grid(b, w, np.max(eps_grid), na)
#%%
aiy = HANK(alpha, beta, delta, sigma, b, na, eps_grid, eps_Q)
#%%

# L is fixed thus given r we pin down K, an together determine w. 
# Given r and w solve the consumption problem of the households
# Find the invariant distribution of assets
# optimal policy & invariant distribution determine the aggregate capital supply
# interpoalte this to get the supply curve
# capital demand available in closed form! 

#r = 0.05 # real interest rate
# actually lets start from K! sinc er = MPK - delta
#%%
r = 0.1
L = 1
K_d = (r + delta)**(1/(1 - alpha)) * L
w = (1-alpha) * (K_d/L)**(alpha)
xgrid = (1+r) * agrid[None, :] + w * eps_grid[:, None]
test, interp = aiy._solve_EGM(agrid, r, w, b, max_iter=250)
#%%
test
#%% optimalpolicy class
np.maximum(interp[0](xgrid[0,:]), -b)

#%%
np.maximum(interp[1](xgrid[1,:]), -b)
#%%
aiy.policy(xgrid)
#%%

aiy.policy.simulate_policy(np.zeros_like(agrid), agrid)
#%%
# Before (Fails):
# np.max(aiy.policy(xgrid, agrid))

# After (Correct):
a_prime_matrix = aiy.policy(eps_grid, xgrid[1,:])
#%%
np.max(a_prime_matrix)
#%%
#%%
L = 1 
r_min = 0.005
r_max = 0.05
r_grid = np.linspace(r_min, r_max, 20)
points = []
for r in r_grid:
    K_d = (alpha/(r + delta))**(1/(1 - alpha)) * L
    w = (1-alpha) * (K_d/L)**(alpha)
    aiy._solve_EGM(agrid, r, w, b, max_iter=250)
    a, k_s = aiy._simulate_stationary_distribution(agrid, w, r, H=10000, T=150, burn_in=50)
    points.append(k_s)
#%% plot
k_min = np.min(points)
k_max = np.max(points)
capital_grid = np.linspace(0.1, k_max, 100)
#interpolate supply
r_sup = np.interp(capital_grid, points, r_grid)
# demand
r_demand = alpha * capital_grid**(alpha-1) - delta
#plt.plot(capital_grid, ) # xx line at y level 1/beta   -1
#%%
plt.axhline(y=1/beta - 1, color='k', linestyle='--', label='r = 1/beta - 1')
plt.plot(capital_grid, r_demand, label="Capital demand")
plt.plot(capital_grid, r_sup, label="Capital supply")
plt.xlabel("Capital (K)")
plt.ylabel("Interest rate (r)")

# --- SCALING FIX ---
plt.ylim([-0.01, 0.1])

plt.title("Capital Market Equilibrium")
plt.legend()
plt.grid(True)
plt.show() # Make sure to include plt.show() to display the plot


#%%
beta = 0.96
alpha = 0.36
delta = 0.08
sigma = 1
rho = 0.9
b = 0
sigma_eps = 0.1743
mu_eps = (- sigma_eps / (2 * (1-rho**2))) 
mu_eps_x = mu_eps * (1- rho)
eps_process = discretize_AR1(rho, mu_eps_x, sigma_eps)
na = 1000
#%% a) setting up the grids
eps_grid_log, eps_Q = eps_process.Rouwenhorst(10)
eps_grid = np.exp(eps_grid_log)
print(eps_grid)
#%%
econ = Economy(alpha, beta, delta, eps_grid, eps_Q)
#%%
hh = Household(econ, sigma, na, b)

#%%
ge = GE(hh)

#%%
r_star = ge.solve(r_min=0)
print(f"Equilibrium interest rate: {r_star:.4f}")
print(f"Capital supply/demand at r*: {ge.history['K_s'][-1]:.2f} / {ge.history['K_d'][-1]:.2f}")
#%%

plt.plot(ge.history['K_s'], ge.history['r'], 'o-', label='K_s')
plt.plot(ge.history['K_d'], ge.history['r'], 'x-', label='K_d')
plt.xlabel("Capital")
plt.ylabel("Interest rate r")
plt.legend()
plt.show()
#%%
kd = (alpha / (r_star + delta))**(1/(1-alpha))
wd = (1 - alpha) * (kd)**alpha
agrid = hh.agrid
#%%
hh._solve_EGM(r_star, wd)
#%%
xgrid = (1+r_star) * agrid[None, :] + wd * eps_grid[:, None]
#%%
hh.policy(xgrid)