from Wildfire import Wildfire
from Plots import PlotFlow
from Helper import Cost_functional
import numpy as np
import sys
import os
from time import perf_counter
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)

np.set_printoptions(threshold=sys.maxsize, linewidth=300)


import matplotlib.pyplot as plt

# Problem variables
Dimension = "1D"
Nxi = 20
Neta = 1
Nt = 10

# Wildfire solver initialization along with grid initialization
wf = Wildfire(Nxi=Nxi, Neta=Neta if Dimension == "1D" else Nxi, timesteps=Nt)
wf.Grid()

# Optimal control
verbose = True
max_opt_steps = 5000
lamda = 1e-4  # regularization parameter
omega = 1e-1  # initial step size for gradient update

f = jnp.zeros((wf.NumConservedVar * wf.Nxi * wf.Neta, wf.Nt))  # Initial guess for the forcing term
qs_target = jnp.concatenate((jnp.zeros((wf.Nxi * wf.Neta, wf.Nt)),
                            jnp.ones((wf.Nxi * wf.Neta, wf.Nt))), axis=0)  # Target value of the variables
sigma = jnp.ones_like(qs_target)

J_list = []  # Collecting cost functional over the optimization steps
dJ_min = 1e-10

# Initial conditions for primal is defined here as they only need to defined once.
q0 = wf.InitialConditions()
cf = Cost_functional(qs_target, sigma, q0, lamda, wf, ti_method="bdf4")

'''
Forward calculation and objective function
'''
if verbose: print("\n-------------------------------")
qs = wf.TimeIntegration(q0, f, sigma, ti_method="bdf4")

J = cf.C_JAX_cost(qs, f)
if verbose: print("Objective J= %1.3e" % J)
if verbose: print("\n-------------------------------")

for itr in range(max_opt_steps):
    if verbose: print("\n-------------------------------")
    '''
    JAX derivative computation
    '''
    dJ_dx = jax.jit(jax.jacfwd(cf.C_JAX))(f)

    '''
    JAX control update
    '''
    f, J = cf.C_JAX_update(f, omega, dJ_dx, J, max_Armijo_iter=20)
    if verbose: print(f"itr : {itr}, cost J : {J}, || dJ_dx || = {jnp.linalg.norm(dJ_dx)}")
    if itr == max_opt_steps - 1:
        print('warning maximal number of steps reached',
              "Objective J= %1.3e" % J)
    if verbose: print("\n-------------------------------")


qs = wf.TimeIntegration(q0, f, sigma, ti_method="bdf4")

# Plot the results
pf = PlotFlow(wf.X, wf.Y, wf.t)
if Dimension == "1D":
    # Plot the Full Order Model (FOM)
    pf.plot1D(qs)
else:
    # Plot the Full Order Model (FOM)
    pf.plot2D(qs)
