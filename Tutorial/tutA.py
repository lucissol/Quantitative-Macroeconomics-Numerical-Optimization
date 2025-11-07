# Lets do vlaue function iteration
import numpy as np
import matplotlib.pyplot as plt

#%%
def util(x):
    if x > 0:
        return np.log(x)
    else:
        return -np.inf

def ValueFiter(V0, beta, r, y, a_current, a_next, u):
    V = V0.copy()
    maxit = 100000
    tol = 1e-10
    if u.shape != (len(a_current), len(a_next)):
        raise ValueError("Utility grid should be of shape (N, N) for asset grid of length N.")

    #start the iteration
    print("Starting Iteration...")
    for it in range(maxit):
        phi = u + beta*V[np.newaxis, :]
        a_idx = np.argmax(phi, axis=1) # take max out of each row and store index
        a = np.take_along_axis(a_next, a_idx, axis=-1) 
        Vn = np.take_along_axis(phi, a_idx[:,np.newaxis], axis=-1).squeeze() 
        
        if np.linalg.norm(Vn - V) < (1 + np.linalg.norm(V)) * tol:
            print("..Converged!")
            V = Vn
            return V, a
        else:
            V = Vn.copy()
#%%test
a = np.zeros(10).shape
b = np.zeros(10)[np.newaxis, :].shape
c = np.zeros(10)[:,np.newaxis].shape
print(a, b, c)
#%% parameters
beta = 0.99
r = 1/beta - 1
y = 1

# capital grid
na = 550
a_current = np.linspace(0,1000, na)
a_next = np.linspace(0,1000, na)
n = len(a_current)
nprime = len(a_next)
utility = np.full((n, nprime), -1e10)

# setting up the utility matrix
for i in range(n):
    for j in range(nprime):
        c = (1+r)*a_current[i]+y-a_next[j] 
        utility[i, j] = util(c) # function returns -inf for c <= 0

# %% run through
V0 = np.zeros(n)
V, a = ValueFiter(V0, beta, r, y, a_current, a_next, utility)

#%% plotting
plt.plot(a_current, a)


#%%# -*- coding: utf-8 -*-
Given solution
"""
Created on Thu Oct 16 14:39:49 2025

@author: sztacher
"""

""" Import packages """
import numpy as np
import matplotlib.pyplot as plt 

""" Solve the dynamic consumption-savings problem """
opt_matrix  = 1                     # VFI maximization: whether matrix or vector with for loop

beta        = 0.99                  # Problem parametrization
r           = 1/beta - 1
y           = 1 

na          = 500                   # Set up a grid for assets
agrid       = np.linspace(0,1000,na)

V           = np.zeros_like(agrid)  # Initial guess for the value function 

u           = -1e16*np.ones((na,na))# Compute the payoff matrix 

for i in range(na):
    for j in range(na):
        c   = (1+r)*agrid[i]+y-agrid[j] 
        if c>0:
            u[i,j] = np.log(c)
            
if opt_matrix == 0:                 # If Bellman solved in for loop, preallocate objects
    phi     = np.zeros((na,na))
    a       = np.zeros(na)
    Vn      = np.zeros(na)
            
maxit       = 10000
tolv        = 1e-9
            
for it in range(maxit):
                                    # Bellman equation
    if opt_matrix:
        phi     = u + beta*V[np.newaxis,:]
        a_idx   = np.argmax(phi,axis=1) # Take a max along assets tomorrow dimension
        a       = np.take_along_axis(agrid,a_idx,axis=-1) 
        Vn      = np.take_along_axis(phi,a_idx[:,np.newaxis],axis=-1).squeeze() 
    else:
        for i in range(na):
            phi_i   = u[i,:] + beta*V
            a_idx   = np.argmax(phi_i)
            a[i]    = agrid[a_idx]
            Vn[i]   = phi_i[a_idx]
    
    if np.linalg.norm(Vn-V)<(1+np.linalg.norm(V))*tolv:
        break 
    else:
        V   = Vn.copy()             # ".copy()" ensures that V is not retroactively set to Vn
        
c           = (1+r)*agrid + y - a 

""" Compare the numerical and analytical solutions """ 

plt.plot(agrid,a,label="VFI",linewidth=2.5)
plt.plot(agrid,agrid,label="analytical",linewidth=2.5,linestyle="dashed")
plt.xlabel("$a$")
plt.ylabel("$a'$")
plt.legend()
plt.show()

plt.plot(agrid,c,label="VFI",linewidth=2.5)
plt.plot(agrid,r*agrid+y,label="analytical",linewidth=2.5,linestyle="dashed")
plt.xlabel("$a$")
plt.ylabel("$c$")
plt.legend()
plt.show()

plt.plot(agrid,V,label="VFI",linewidth=2.5)
plt.plot(agrid,np.log(r*agrid+y)/(1-beta),label="analytical",linewidth=2.5,linestyle="dashed")
plt.xlabel("$a$")
plt.ylabel("$V$")
plt.legend()
plt.show()