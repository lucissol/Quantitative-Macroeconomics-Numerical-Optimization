# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 09:48:50 2025

@author: lseibert
"""
from scripts.discretizingAR1 import discretize_AR1
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
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
#%% 
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
#%%
plt.figure()
plt.plot(theta_tauchen, np.zeros_like(theta_tauchen), 'o', label='Tauchen θ')
plt.plot(theta_rouwen, np.ones_like(theta_rouwen)*0.1, 'x', label='Rouwenhorst θ')
plt.yticks([])
plt.legend()
plt.title("Discretized θ grids")
plt.show()
#%%
print(theta_rouwen, theta_tauchen)
#%% Tauchen
T = 25000
initial_state_idx = np.random.randint(0, N) # makes sure that both have a common starting point

time_seriesT = process.simulate_time_series(T, tauchen_Z, tauchen_Q, initial_state_idx)

plt.plot(time_seriesT)
plt.title("Simulated AR(1) Process (Discretized with Tauchen)")
plt.xlabel("Time Step")
plt.ylabel("State Value")
plt.show()
#%% Rouwenhorst
time_seriesR = process.simulate_time_series(T, rouwen_Z, rouwen_Q, initial_state_idx)

plt.plot(time_seriesR)
plt.title("Simulated AR(1) Process (Discretized with Rouwenhorst)")
plt.xlabel("Time Step")
plt.ylabel("State Value")
plt.show()

#%% Rouwenhorst
y_tR = time_seriesR[1:]
y_t_minus_1R = time_seriesR[:-1]

# Add constant for intercept (though we expect the intercept to be close to 0 for AR(1))
XR = y_t_minus_1R.reshape(-1, 1)

# Perform OLS regression using sklearn
modelR = LinearRegression()
modelR.fit(XR, y_tR)

rho_estimateR = modelR.coef_[0]  # The coefficient of y_{t-1}
intercept_estimateR = modelR.intercept_  
# Compute residuals (difference between observed and predicted)
residuals_R = y_tR - modelR.predict(XR)
mse_R = np.mean(np.sqrt(residuals_R**2))
# Print results
print(f"Estimated rho: {rho_estimateR:.4f}")
print(f"Intercept: {intercept_estimateR:.4f}")
print(f"Residuals: {residuals_R[:10]}...")  # Display first 10 residuals

# Plot residuals
plt.plot(residuals_R)
plt.title("Rouwenhorst Residuals from OLS Fit")
plt.xlabel("Time Step")
plt.ylabel("Residual")
plt.show()
print(modelR.predict(XR))
print(mse_R)
#%%
y_tT = time_seriesT[1:]  # Current state (y_t)
y_t_minus_1T = time_seriesT[:-1]  # Previous state (y_{t-1})

# Add constant for intercept (though we expect the intercept to be close to 0 for AR(1))
XT = y_t_minus_1T.reshape(-1, 1)
# Perform OLS regression using sklearn
modelT = LinearRegression()
modelT.fit(XT, y_tT)
# Get the estimated rho (coefficient) and intercept
rho_estimateT = modelT.coef_[0]  # The coefficient of y_{t-1}
intercept_estimateT = modelT.intercept_  # Intercept (should be close to 0)

# Compute residuals (difference between observed and predicted)
residuals_T= y_tT - modelT.predict(XT)
mse_T = np.mean(np.sqrt(residuals_T**2))

# Print results
print(f"Estimated rho: {rho_estimateT:.4f}")
print(f"Intercept: {intercept_estimateT:.4f}")
print(f"Residuals: {residuals_T[:10]}...")  # Display first 10 residuals

# Plot residuals
plt.plot(residuals_T)
plt.title("Tauchen Residuals from OLS Fit")
plt.xlabel("Time Step")
plt.ylabel("Residual")
plt.show()
print(modelT.predict(XT))
print(mse_T)
#%%
mse_T
#%% Results
# Rouwenhorst has smaller OLS error but clearly higher max deviations!
# Tauchen's method generates a weak grid as we heard in the lecture that it fails with highly persitant processes. 
# Rouwenhorst on the otehr side generates stickiness even at the end of the grid which results in same large deviations