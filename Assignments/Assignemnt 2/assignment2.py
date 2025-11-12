# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 09:48:50 2025

@author: lseibert
"""
from scripts.discretizingAR1 import discretize_AR1
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg
import seaborn as sns
import pandas as pd

#%%
# idea is to simulate the transformed process y_t = rho * y_t-1 + eps_t, with y_t = log(theta_t)
rho = 0.979
mu = 0
sigma = 0.0072
N = 7

process = discretize_AR1(rho, mu, sigma)
#%% Tauchen
tauchen_Z, tauchen_Q = process.Tauchen(N, m=3)
theta_tauchen = np.exp(tauchen_Z)
#%% Rouwenhorst
rouwen_Z, rouwen_Q = process.Rouwenhorst(N)
theta_rouwen = np.exp(rouwen_Z)
#%% Transition Matrices
# Tauchen Q
plt.figure(figsize=(6,5))
sns.heatmap(tauchen_Q, annot=True, fmt=".2f", cmap="viridis")
plt.title("Tauchen Transition Matrix")
plt.xlabel("Next state")
plt.ylabel("Current state")
plt.show()

# Rouwenhorst Q
plt.figure(figsize=(6,5))
sns.heatmap(rouwen_Q, annot=True, fmt=".2f", cmap="viridis")
plt.title("Rouwenhorst Transition Matrix")
plt.xlabel("Next state")
plt.ylabel("Current state")
plt.show()
#%% implied theta values
plt.figure()
plt.plot(theta_tauchen, np.zeros_like(theta_tauchen), 'o', label='Tauchen θ')
plt.plot(theta_rouwen, np.zeros_like(theta_rouwen), 'x', label='Rouwenhorst θ')
plt.yticks([])
plt.legend()
plt.title("Discretized θ grids")
plt.show()
#%% Tauchen simulation
T = 25000
initial_state_idx = np.random.randint(0, N) # makes sure that both have a common starting point
time_seriesT = process.simulate_time_series(T, tauchen_Z, tauchen_Q, initial_state_idx)

plt.plot(time_seriesT)
plt.title("Simulated AR(1) Process (Tauchen)")
plt.xlabel("Time Step")
plt.ylabel("State Value")
plt.show()
#%% Rouwenhorst simulation
time_seriesR = process.simulate_time_series(T, rouwen_Z, rouwen_Q, initial_state_idx)

plt.plot(time_seriesR)
plt.title("Simulated AR(1) Process (Rouwenhorst)")
plt.xlabel("Time Step")
plt.ylabel("State Value")
plt.show()

#%% Estimation check

model_tc = AutoReg(time_seriesR, lags=1).fit()
print(model_tc.summary())

model_rw = AutoReg(time_seriesT, lags=1).fit()
print(model_rw.summary())
       
models = [model_tc, model_rw]
names = ["Tauchen estimation", "Rouwenhorst estimation"]

data = {
    "Constant": [m.params[0] for m in models],
    "AR(1) Coefficient": [m.params[1] for m in models]
}

comparison_table = pd.DataFrame(data, index=names)

comparison_table.loc["True values"] = [mu, rho] #appending the true values

print(comparison_table)
# Answer: Tauchen's method is to sticky and overreflects the persistance. Rouwenhorst correctly produces an AR1 with 3 digit precision!

#%% Results
# Rouwenhorst has smaller OLS error but clearly higher max deviations!
# Tauchen's method generates a weak grid ass we heard in the lecture that it fails with highly persitant processes. 
# Rouwenhorst on the otehr side generates stickiness even at the end of the grid which results in same large deviation

#%% ### Exercise 2: A simple stochastic Ramsey model ###
beta = 0.984
alpha = 0.323
delta = 0.025
rho = 0.979
mu_eps = 0
# process 1
sigma_eps1 = 0.0072
eps_process1 = discretize_AR1(rho, mu_eps, sigma_eps1)
# process 2
sigma_eps2 = 0.2
eps_process2 = discretize_AR1(rho, mu_eps, sigma_eps2)
#setting up the grid
k_ss = (alpha / (1/beta -1 + delta))**(1/(1 - alpha)) # steady state without varying productivity)
k_min = 10e-9
k_max = 3 * k_ss 
print(f"Grid defined on {k_min} to {k_max}")

nk = 2500
kgrid = np.linspace(k_min, k_max, nk)

nk_large = 10000
kgrid_large = np.linspace(k_min, 600, nk_large) # hardcoded to fit policy function of the 0.2 stochastic model
#%% a) setting up the grids
eps_grid_log1, eps_Q1 = eps_process1.Rouwenhorst(7)
eps_grid1 = np.exp(eps_grid_log1)
eps_grid_log2, eps_Q2 = eps_process2.Rouwenhorst(7)
eps_grid2 = np.exp(eps_grid_log2)
print(eps_grid1)
print(eps_grid2)
#%% b)
from scripts.functions import ValueFunctionMethods

state_of_the_worlds = []
for i, theta in enumerate(eps_grid1):
    # closed form steady state for deterministic case
    x_grid = theta * kgrid**alpha + (1-delta)*kgrid
    V_initial = np.log(x_grid) / (1 - beta)
    solver = ValueFunctionMethods(beta=beta, alpha=1.0, state_grid=x_grid, choice_grid=kgrid)
    V, a_opt, time_s, it = solver.PFIhoward_exact(
        Vguess=V_initial,
        max_iter=1000,
        verbose=True
    )
    a_analytical = theta*alpha*beta * kgrid**alpha
    state_of_the_worlds.append({
    'theta': theta,
    'steady_state': k_ss,
    'value_function': V,
    'policy_function': a_opt,
    'policy_analytical': a_analytical,
    'capital_grid': kgrid
})
    
state_of_the_worlds_large_shock = []
for i, theta in enumerate(eps_grid2):
    # closed form steady state for deterministic case
    x_grid = theta * kgrid**alpha + (1-delta)*kgrid
    V_initial = np.log(x_grid) / (1 - beta)
    solver = ValueFunctionMethods(beta=beta, alpha=1.0, state_grid=x_grid, choice_grid=kgrid)
    V, a_opt, time_s, it = solver.PFIhoward_exact(
        Vguess=V_initial,
        max_iter=1000,
        verbose=True
    )
    a_analytical = theta*alpha*beta * kgrid**alpha
    state_of_the_worlds_large_shock.append({
    'theta': theta,
    'steady_state': k_ss,
    'value_function': V,
    'policy_function': a_opt,
    'policy_analytical': a_analytical,
    'capital_grid': kgrid
})

#%% c)
from scripts.discretizingAR1 import agg_uncertainty

model1 = agg_uncertainty(beta, alpha, delta, eps_grid1, eps_Q1)
model2 = agg_uncertainty(beta, alpha, delta, eps_grid2, eps_Q2)
policy_egm1 = model1.solve_EGM(kgrid)
policy_egm2 = model2.solve_EGM(kgrid)

#%% plots of comparing stochastic policy of large and small shock economy
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# --- First subplot: policy_egm1 ---
ax = axes[0]
for z_i, z in enumerate(eps_grid1):
    ax.plot(kgrid, policy_egm1[z_i, :], label=fr'$\theta = {z:.2f}$')
ax.plot(kgrid, kgrid, 'k--', label='45° line')
ax.set_title('Policy Function (EGM1)')
ax.set_xlabel('k')
ax.set_ylabel("Policy Function (k')")
ax.set_xlim(0, 2)
ax.set_ylim(0, 3)
ax.legend(title=r'$\theta$ values', fontsize=9)

# --- Second subplot: policy_egm2 ---
ax = axes[1]
for z_i, z in enumerate(eps_grid2):
    ax.plot(kgrid, policy_egm2[z_i, :], label=fr'$\theta = {z:.2f}$')
ax.plot(kgrid, kgrid, 'k--', label='45° line')
ax.set_title('Policy Function (EGM2)')
ax.set_xlabel('k')
ax.set_xlim(0, 2)
ax.set_ylim(0, 3)
ax.legend(title=r'$\theta$ values', fontsize=9)

plt.suptitle('Policy Functions for Different Productivity States (Small vs. large shock economy)', fontsize=14)
plt.tight_layout()
plt.show()

#%% Difference of the policies small shock economy deterministic vs. stochastic 
for i, state in enumerate(state_of_the_worlds):
    theta = state['theta']  # assuming each dict has the productivity level
    k_det = state['capital_grid']
    pol_det = state['policy_function']
    pol_stoch = policy_egm1[i, :]  # same index corresponds to same θ

    fig, ax = plt.subplots(figsize=(7,5))

    ax.plot(k_det, pol_det, '--', linewidth=2, label='Deterministic policy', color='tab:blue')
    ax.plot(kgrid, pol_stoch, '-', linewidth=2, label='Stochastic policy (EGM)', color='tab:orange')

    ax.set_xlabel('Current Capital (k)')
    ax.set_ylabel("Next Period Capital (k')")
    ax.set_title(f'Policy Comparison for θ = {theta:.3f} small shock economy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 3)

    plt.tight_layout()
    plt.show()
    
# There is no clear difference observable looking at the graphics between the deterministic and stochastic policy
#%% Difference of the policies large shock economy deterministic vs. stochastic 
for i, state in enumerate(state_of_the_worlds_large_shock):
    theta = state['theta']  # assuming each dict has the productivity level
    k_det = state['capital_grid']
    pol_det = state['policy_function']
    pol_stoch = policy_egm2[i, :]  # same index corresponds to same θ

    fig, ax = plt.subplots(figsize=(7,5))

    ax.plot(k_det, pol_det, '--', linewidth=2, label='Deterministic policy', color='tab:blue')
    ax.plot(kgrid, pol_stoch, '-', linewidth=2, label='Stochastic policy (EGM)', color='tab:orange')

    ax.set_xlabel('Current Capital (k)')
    ax.set_ylabel("Next Period Capital (k')")
    ax.set_title(f'Policy Comparison for θ = {theta:.3f} large shock economy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 3)

    plt.tight_layout()
    plt.show()
    
# The difference can be particularly seen looking at the uncertain economy. Here, at beneficial states the optimal policy demands to save more than in the deterministic case.
# Vice versa, in "bad states" the optimal policy demands to save less than optimally in the deterministic case
#%% d) 
T = 5100
burnin = 100
view = 200
#%% full certainty
determ = discretize_AR1(rho, 0, 0)
deter, deterQ = determ.Rouwenhorst(7)
certain = agg_uncertainty(beta, alpha, delta, np.exp(deter), deterQ)
k_sim, c_sim, y_sim, path_sim = certain.simulate_economy(kgrid, T, burnin, plot=True)

#%% simulate the economy for process with sigma 0.0072
k_sim1, c_sim1, y_sim1, path_sim1 = model1.simulate_economy(kgrid, T, burnin, plot=True)
# plot for the last 2000 simlations:
broadcast_range_var = T-2000-burnin
time_grid = np.arange(T-2000, T)

plt.plot(time_grid, k_sim1[broadcast_range_var:],label="capital",linewidth=1)
plt.plot(time_grid, c_sim1[broadcast_range_var:],label="consumption",linewidth=1)
plt.plot(time_grid, y_sim1[broadcast_range_var:],label="output",linewidth=1)
plt.xlabel("time")
plt.legend()
plt.title('capital, consumption, output evolution last 2000 periods')
plt.show()
#%% stationary distribution small shock economy
sns.histplot(k_sim1, kde=True, color='skyblue', bins=40)
mean_k = np.mean(k_sim1)
unc_theta1 = np.exp((sigma_eps1**2) / (2 * (1 - rho**2)))
k_ss_deterministic1 = (unc_theta1 * alpha / (1/beta -1 + delta))**(1/(1 - alpha))

plt.axvline(mean_k, color='red', linestyle='--', linewidth=2, label=f'Mean k = {mean_k:.3f}')
plt.axvline(k_ss_deterministic1, color='green', linestyle='-.', linewidth=2, 
            label=f'Deterministic k* = {k_ss_deterministic1:.3f}')
plt.title('Distribution of Simulated Capital')
plt.xlabel('Capital')
plt.legend()
plt.show()

print(f"Mean of simulated k: {mean_k:.4f}")
print(f"Unconditional mean productivity (unc_theta2): {unc_theta1:.4f}")
print(f"Deterministic steady-state k*: {k_ss_deterministic1:.4f}")
# The mean capital stock holding is almost identical to the deterministic economy with same unconditional producitivity.
# The small discrepance is due the certainty equivalence, that the representative agent stills prefers to have fully determined consumption path!
#%% economy with large risk 0.2
k_sim2, c_sim2, y_sim2, path_sim2 = model2.simulate_economy(kgrid_large, T, burnin, plot=True)

# plot for the last 2000 simlations:
plt.plot(time_grid, k_sim2[broadcast_range_var:],label="capital",linewidth=1)
plt.plot(time_grid, c_sim2[broadcast_range_var:],label="consumption",linewidth=1)
plt.plot(time_grid, y_sim2[broadcast_range_var:],label="output",linewidth=1)
plt.xlabel("time")
plt.legend()
plt.title('capital, consumption, output evolution last 2000 periods')
plt.show()
#%% stationary distribution large shock economy
sns.histplot(k_sim2, kde=True, color='skyblue', bins=40)

mean_k = np.mean(k_sim2)
unc_theta2 = np.exp((sigma_eps2**2) / (2 * (1 - rho**2)))
k_ss_deterministic2 = (unc_theta2 * alpha / (1/beta -1 + delta))**(1/(1 - alpha))


plt.axvline(mean_k, color='red', linestyle='--', linewidth=2, label=f'Mean k = {mean_k:.3f}')
plt.axvline(k_ss_deterministic2, color='green', linestyle='-.', linewidth=2, 
            label=f'Deterministic k* = {k_ss_deterministic2:.3f}')
plt.title('Distribution of Simulated Capital')
plt.xlabel('Capital')
plt.legend()

plt.show()

# Print the values for reference
print(f"Mean of simulated k: {mean_k:.4f}")
print(f"Unconditional mean productivity (unc_theta2): {unc_theta2:.4f}")
print(f"Deterministic steady-state k*: {k_ss_deterministic2:.4f}")
# we observe a clear difference towards the small shock economy. Now, since the unconditional mean is about 1.6 the economy accumulates an extremely large capital stock to insure future uncertainty!
#%% g)
# to answer this question we refer back to the small risk economy and compare the capital distribution for different grid sizes
grid_size = [3, 7, 11, 15, 19]
results = {} 
for N in grid_size:
    eps_grid_log1, eps_Q1 = eps_process1.Rouwenhorst(N)
    eps_grid1 = np.exp(eps_grid_log1)
    current_model = agg_uncertainty(beta, alpha, delta, eps_grid1, eps_Q1)
    k_sim, c_sim, y_sim, path_sim = current_model.simulate_economy(kgrid, T, burnin, plot=False)
    # store results
    results[N] = {
        "k": k_sim,
        "c": c_sim,
        "y": y_sim,
        "path": path_sim
    }
#%% g) results
for N, data in results.items():
    print(f"N={N}, mean={np.mean(data['k']):.4f}, std={np.std(data['k']):.4f}")

plt.figure(figsize=(8,5))
for N, data in results.items():
    sns.kdeplot(data["k"], label=f"N={N}")
plt.title("Stationary Capital Distribution for Different Grid Sizes")
plt.xlabel("Capital $k_t$")
plt.legend()
plt.show()

# Judging based on the graphic, one should at least go with the green (N=11) states.
#%% Euler Equation error!
kfine = np.linspace(k_min, k_max, nk+1000)

#%% Dynamic Euler Equation Error
imp_econ = model1._dynamic_EE(policy_egm1, kgrid, T, burnin)
real_econ = model1.simulate_economy(kgrid, T, burnin, plot=False)
static_EEE = model1.static_EEE(policy_egm1, kgrid, kgrid)
dynamic_EEE = np.abs((real_econ[1] - imp_econ[1]) / imp_econ[1])
#%%
print("Baseline Economy EEE")
print(f" Dynamic EEE: {np.mean(dynamic_EEE)}")
print(f" Static EEE: {np.mean(static_EEE)}")
# Dynamic Euler Equation Errors accumulate over time and thus we observe a higher mean error than in the static case!
#%% h) for large shock economy
imp_econ2 = model2._dynamic_EE(policy_egm1, kgrid_large, T, burnin)
real_econ2 = model2.simulate_economy(kgrid_large, T, burnin, plot=False)
static_EEE2 = model2.static_EEE(policy_egm1, kgrid_large, kgrid_large)
dynamic_EEE2 = np.abs((real_econ[1] - imp_econ[1]) / imp_econ[1])
#%%
print("Large shock Economy EEE")
print(f" Dynamic EEE: {np.mean(dynamic_EEE2)}")
print(f" Static EEE: {np.mean(static_EEE2)}")
# Dynamic Euler Equation Errors accumulate over time and thus we observe a higher mean error than in the static case!
# On top we observe significantly higher errors than in the baseline economy!