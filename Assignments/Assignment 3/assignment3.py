# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 17:27:48 2025

@author: lseibert
"""
#%%
import numpy as np
from scripts.stochastic_euler import discretize_AR1
import matplotlib.pyplot as plt
from scripts.aiyagari import OptimalPolicy, Economy, Household, GE
#%%
alpha = 0.36
beta = 0.96
delta = 0.08

rho = 0.9
sigma_eps = 0.1743
mu_eps = (- sigma_eps / (2 * (1-rho**2))) 
mu_eps_x = mu_eps * (1- rho)
eps_process = discretize_AR1(rho, mu_eps_x, sigma_eps)
eps_grid_log, eps_Q = eps_process.Rouwenhorst(10)
eps_grid = np.exp(eps_grid_log)
print(eps_grid)

economy_values = Economy(alpha, beta, delta, eps_grid, eps_Q)

sigma = 1
na = 10000
b = 0
household_aiy = Household(economy_values, sigma, na, b)
#%%
L = 1 
r_min = 0.0005
r_max = (1/beta - 1 - 1e-10)
r_grid = np.linspace(r_min, r_max, 15)
points = []
for r in r_grid:
    K_d = (alpha/(r + delta))**(1/(1 - alpha)) * L
    w = (1-alpha) * (K_d/L)**(alpha)
    household_aiy._solve_EGM(r, w, max_iter=1000, verbose = True)
    a, k_s = household_aiy._simulate_stationary_distribution(r, w, H=10000, T=150, burn_in=50)
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

#%% setting up the grids
ge = GE(household_aiy)

#%%
r_star = ge.solve(r_min=0)
print(f"Equilibrium interest rate: {r_star:.4f}")
print(f"Capital supply/demand at r*: {ge.history['K_s'][-1]:.2f} / {ge.history['K_d'][-1]:.2f}")
#%%
index = np.arange(0, len(ge.history['K_s']))

plt.plot(index, ge.history['r'], 'x-', label='r')
plt.xlabel("Capital")
plt.ylabel("Interest rate r")
plt.legend()
plt.ylim([-0.01, 0.1])
plt.show()
#%%
plt.plot(index, ge.history['K_s'], 'o-', label='K_s')
plt.plot(index, ge.history['K_d'], 'x-', label='K_d')
plt.legend()
plt.show()
#%%
plt.plot(ge.history['K_s'], ge.history['r'], 'o-', label='K_s')
plt.plot(ge.history['K_d'], ge.history['r'], 'x-', label='K_d')

#%% what happens when sigma = 2?
household_aiy2 = Household(economy_values, sigma=2, na=na, b=b)
ge2 = GE(household_aiy2)

#%%
r_star = ge2.solve(r_min=0)
print(f"Equilibrium interest rate: {r_star:.4f}")
print(f"Capital supply/demand at r*: {ge.history['K_s'][-1]:.2f} / {ge.history['K_d'][-1]:.2f}")
#%%
index2 = np.arange(0, len(ge2.history['K_s']))
plt.plot(index2, ge2.history['r'], 'x-', label='r')
plt.xlabel("Capital")
plt.ylabel("Interest rate r")
plt.legend()
plt.ylim([-0.01, 0.1])
plt.show()
#%%
plt.plot(index2, ge2.history['K_s'], 'o-', label='K_s')
plt.plot(index2, ge2.history['K_d'], 'x-', label='K_d')
plt.legend()
plt.show()
#%%
plt.plot(ge2.history['K_s'], ge2.history['r'], 'o-', label='K_s')
plt.plot(ge2.history['K_d'], ge2.history['r'], 'x-', label='K_d')

# answer: sigma reflects the intertemporal elasticity of substitution meaning that with sigma = 2, households react more sensitive to interest rate changes
# As a result aggregate savings increase and the interest decreases in favor for the firms!
#%%
plt.hist(ge.a, label = "sigma 2")
plt.legend()
plt.show()

#%%
plt.hist(ge2.a, label = "sigma 2")
plt.legend()
plt.show()
# collection of results:
    # sigma increases risk aversion of households
    # reflects in lower interest rate
    # more households try to accumulate assets thus more compressed right tail
    # because of lower interest rate, maximum savings decrease!