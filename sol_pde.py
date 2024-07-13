#%%
import sys
sys.path.append('/Users/xyguo/Documents/phase_trans/src')
from diffeqpy import de
from utils.util import custom_fig, save_fig, save
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def get_params(job_idx):
    n_list = [2,3,4]
    tau_q_list = range(64,2048+64,64)
    for i,item in enumerate(product(n_list,tau_q_list)):
        if i == job_idx:
            n,tau_q = item
            return n,tau_q

# Define the physical dynamics
def f(du, u, p, t):
    sigma,theta,tau_q,n = p
    eps = t/tau_q
    eta = 1

    du[0] = u[1]
    du[1] = -eta * u[1] - 0.125*(-4 * eps * u[0] + 2*n * u[0] ** (2*n-1))

def g(du, u, p, t):
    sigma,theta,tau_q,n = p

    du[0] = 0
    du[1] = 2 * sigma * theta

# Define the function to run the simulation
def run_sim(sigma, theta,tau_q,n, dt, tmax):
    prob = de.SDEProblem(f, g, [0.0, 0.0], (0.0, tmax), [sigma,theta,tau_q,n])
    sol = de.solve(prob, de.SRIW1(), dt=dt, adaptive=False)
    return sol

if __name__ == "__main__":
    sigma = 1e-4
    dt = 0.01
    tmax = 500.0
    theta = 0.1
    # job_idx = int(sys.argv[1])
    for job_idx in range(150):
        n,tau_q = get_params(job_idx)
        # Run the simulation
        sol = run_sim(sigma,theta,tau_q,n, dt, tmax)
        us = de.stack(sol.u)
        save(us,f'/Users/xyguo/Documents/phase_trans/data/raw/n_{n}_tau_q_{tau_q}.pkl')



