# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 17:27:48 2025

@author: lseibert
"""
#%%
import numpy as np
from scripts.stochastic_euler import discretize_AR1
import matplotlib.pyplot as plt
from scripts.aiyagari import Economy, Household, GE, GE_tax
#%%
alpha = 0.36
beta = 0.96
delta = 0.08

rho = 0.9
sigma_eps = 0.1743
mu_eps = (- sigma_eps / (2 * (1-rho**2))) 
mu_eps_x = mu_eps * (1- rho)
eps_process = discretize_AR1(rho, mu_eps_x, sigma_eps)
eps_grid_log, eps_Q = eps_process.Rouwenhorst(3)
eps_grid = np.exp(eps_grid_log)
print(eps_grid)

economy_values = Economy(alpha, beta, delta, eps_grid, eps_Q)

sigma = 1
na = 1500
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
# --- SCALING Region of Interest ---
plt.ylim([-0.01, 0.1])

plt.title("Capital Market Equilibrium")
plt.legend()
plt.grid(True)
plt.show()

#%% setting up the grids
ge = GE(household_aiy)

#%% solving for GE in no tax version
r_star = ge.solve(r_min=0, r_max=np.max(r_sup))
print(f"Equilibrium interest rate: {r_star:.4f}")
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

#%%
plt.axhline(y=1/beta - 1, color='k', linestyle='--', label='r = 1/beta - 1')

plt.plot(capital_grid, r_demand, label="Capital demand")
plt.plot(capital_grid, r_sup, label="Capital supply")
plt.xlabel("Capital (K)")
plt.ylabel("Interest rate (r)")
plt.plot(ge.history['K_d'], ge.history['r'], 'x-', label='K_d-r path')
plt.ylim([-0.01, 0.1])

plt.title("Capital Market Equilibrium - convergence path capital demand & r")
plt.legend()
plt.grid(True)
plt.show()
#%%
plt.axhline(y=1/beta - 1, color='k', linestyle='--', label='r = 1/beta - 1')

plt.plot(capital_grid, r_demand, label="Capital demand")
plt.plot(capital_grid, r_sup, label="Capital supply")
plt.xlabel("Capital (K)")
plt.ylabel("Interest rate (r)")
plt.plot(ge.history['K_s'], ge.history['r'], 'x-', label='K_s-r path')
plt.ylim([-0.01, 0.1])

plt.title("Capital Market Equilibrium - convergence path capital supply & r")
plt.legend()
plt.grid(True)
plt.show()
#%% what happens when sigma = 2?
###############################################################################################
household_aiy2 = Household(economy_values, sigma=2, na=na, b=b)
ge2 = GE(household_aiy2)

#%%
r_star = ge2.solve(r_min=0)
print(f"Equilibrium interest rate: {r_star:.4f}")
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

# answer: sigma reflects the risk aversion of the households meaning that with sigma = 2
# their utility function is more concave now and thus they want to save more in order to tackle consumption uncertiatny in the future
# As a result aggregate savings increase and the interest decreases in favor for the firms!
#%%
plt.hist(ge.a, label = "sigma 1")
plt.title("Invariant asset distribution")
plt.legend()
plt.show()

#%%
plt.hist(ge2.a, label = "sigma 2")
plt.title("Invariant asset distribution")
plt.legend()
plt.show()
# collection of results:
    # sigma increases risk aversion of households
    # higher aggregate savings lead to lower interest rate
    # more households try to accumulate assets thus more compressed right tail
    # because of lower interest rate, maximum savings decrease!
###############################################################################################
#%% Aiyagari with taxes!
household_aiy_tax = Household(economy_values, sigma, na, b)
ge2_tax = GE_tax(household_aiy_tax, G = 0.25)

#%%
L = 1
tau_list = [0.00, 0.05, 0.10, 0.15] # taxes 
r_min = 0.0005
r_max = 0.06 # something below 1/beta - 1 
r_grid = np.linspace(r_min, r_max, 15)

# Storage
demand_points = []
supply_curves = {}

for tau in tau_list:
    supply_curves[tau] = []  

for r in r_grid:
    K_d = (alpha/(r + delta))**(1/(1 - alpha)) * L
    demand_points.append((K_d, r))
    for tau in tau_list:
        w = (1-alpha) * (K_d/L)**alpha
        r_taxed = (1 - tau) * r
        w_taxed = (1 - tau) * w
        household_aiy._solve_EGM(r_taxed, w_taxed,
                                 max_iter=500, verbose=False)
        a, k_s = household_aiy._simulate_stationary_distribution(
                        r_taxed, w_taxed,
                        H=8000, T=200, burn_in=50)

        supply_curves[tau].append((k_s, r))
#%% plot
plt.figure(figsize=(10,6))

K_d_vec = [p[0] for p in demand_points]
r_vec   = [p[1] for p in demand_points]

colors = ['b', 'g', 'r', 'orange']

for tau, color in zip(tau_list, colors):
    K_s_vec = [p[0] for p in supply_curves[tau]]
    plt.plot(K_s_vec, r_grid, color=color, linestyle='--',
             label=f"Supply (tau={tau})")

plt.xlabel("Capital K")
plt.ylabel("Interest rate r")
plt.title("Capital Market Equilibrium for Different Tax Rates τ")
plt.grid(True)
plt.legend()
plt.ylim([0, (r_max + 0.1)])
plt.show()
# what we see here is that the higher the equilibrium interest rate, the less aggregate savings we have
# and thus we require a higher tau value to finance the G spendings!
#%% c)
imp_tau = np.zeros(len(r_grid))
for i, r in enumerate(r_grid):
    K_d, w = ge2_tax.capital_demand(r)
    imp_tau[i] = ge2_tax.implied_tau(r, K_d, w)
r_grid_fine = np.linspace(np.min(r_grid), np.max(r_grid), 100)
taus = np.interp(r_grid_fine, r_grid, imp_tau)
#%%
plt.figure(figsize=(8,5))
plt.plot(r_grid_fine, taus, 'b-', linewidth=2)
plt.xlabel("Interest rate r")
plt.ylabel("Tax rate tau")
plt.title("Tau as a function of r using capital demand")
plt.grid(True)
plt.show()
# answer: tau is increasing in r! Since higher vlaues of r imply lower capital demand
# 
# the government must increase higher taxes to finance its spendings
#%% solving the general equilibirum model
collect_tax = ge2_tax.solve(r_min = 0.02, r_max = 0.045)
# solve() took 937.977 seconds - expected run time!!! ~~ 16 minutes!
#changed the inner loop convergence criterium -> now running only 15 times to speed up inner convergence and hopefully get stable r
# change to lower grid points and fewer shock space for quick checkup
#%%
index3 = np.arange(0, len(ge2_tax.history['K_s']))
plt.plot(index3, ge2_tax.history['r'], 'x-', label='r')
plt.plot(index3, ge2_tax.history['tau'], 'x-', label='tau')
plt.xlabel("Capital")
plt.ylabel("Interest rate r")
plt.legend()
plt.ylim([-0.01, 0.1])
plt.show()
# Note that interest rate is higher since households face "real interest rate" (1+ (1-tau)r)
# whereas firms still observe interest rate r!
#%%
plt.plot(index3, ge2_tax.history['K_s'], 'o-', label='K_s')
plt.plot(index3, ge2_tax.history['K_d'], 'x-', label='K_d')
plt.legend()
plt.show()
#%%
plt.plot(ge2_tax.history['K_s'], ge2_tax.history['r'], 'o-', label='K_s')
plt.plot(ge2_tax.history['K_d'], ge2_tax.history['r'], 'x-', label='K_d')
plt.show()
