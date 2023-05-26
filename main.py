from Wildfire import Wildfire
from Plots import PlotFlow
from Helper import Calc_Cost, Update_Control
from Wildfire import Adjoint_Matrices, Force_masking
import numpy as np
import sys
import os
from time import perf_counter

import jax
import jax.numpy as jnp

np.set_printoptions(threshold=sys.maxsize, linewidth=300)

import matplotlib.pyplot as plt

# Problem variables
Dimension = "1D"
Nxi = 10
Neta = 1
Nt = 1000

# Wildfire solver initialization along with grid initialization
wf = Wildfire(Nxi=Nxi, Neta=Neta if Dimension == "1D" else Nxi, timesteps=Nt)
wf.Grid()

# Optimal control
max_opt_steps = 1
lamda = 1e-5  # regularization parameter
omega = 1e-1  # initial step size for gradient update

f = jnp.zeros((wf.NumConservedVar * wf.Nxi * wf.Neta, wf.Nt))  # Initial guess for the forcing term
qs_target = jnp.concatenate((jnp.zeros((wf.Nxi * wf.Neta, wf.Nt)),
                            jnp.ones((wf.Nxi * wf.Neta, wf.Nt))), axis=0)  # Target value of the variables

# # Compute the masking array
# qs = wf.TimeIntegration_primal(wf.InitialConditions_primal(),
#                                np.zeros((wf.Nxi * wf.Neta, wf.Nt)))
# sigma = Force_masking(qs, wf.X, wf.Y, wf.t, dim=1)
# sigma = np.tile(sigma, (2, 1))
sigma = jnp.zeros_like(qs_target)

# Initial conditions for both primal and adjoint are defined here as they only need to defined once.
q0 = wf.InitialConditions()
qs = wf.TimeIntegration(q0, f, sigma)

# plt.ion()
# fig, ax = plt.subplots(1, 1)
# for n in range(wf.Nt):
#     ax.plot(wf.X, qs[:wf.Nxi, n])
#     plt.draw()
#     plt.pause(0.002)
#     ax.cla()
# exit()


# Plot the results
pf = PlotFlow(wf.X, wf.Y, wf.t)
if Dimension == "1D":
    # Plot the Full Order Model (FOM)
    pf.plot1D(qs)
else:
    # Plot the Full Order Model (FOM)
    pf.plot2D(qs)
