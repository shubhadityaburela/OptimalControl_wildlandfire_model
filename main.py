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

np.set_printoptions(threshold=sys.maxsize)

import matplotlib.pyplot as plt

# Problem variables
Dimension = "1D"
Nxi = 500
Neta = 1
Nt = 1000

# Wildfire solver initialization along with grid initialization
wf = Wildfire(Nxi=Nxi, Neta=Neta if Dimension == "1D" else Nxi, timesteps=Nt)
wf.Grid()

# Optimal control
max_opt_steps = 100
verbose = True
lamda = 1e-5  # regularization parameter
omega = 1e-1  # initial step size for gradient update

f = np.zeros((wf.NumConservedVar * wf.Nxi * wf.Neta, wf.Nt))  # Initial guess for the forcing term
qs_target = np.concatenate((np.zeros((wf.Nxi * wf.Neta, wf.Nt)),
                            np.ones((wf.Nxi * wf.Neta, wf.Nt))), axis=0)  # Target value of the variables

J_list = []  # Collecting cost functional over the optimization steps
dJ_min = 1e-10

# # Compute the masking array
# qs = wf.TimeIntegration_primal(wf.InitialConditions_primal(),
#                                np.zeros((wf.Nxi * wf.Neta, wf.Nt)))
# sigma = Force_masking(qs, wf.X, wf.Y, wf.t, dim=1)
# sigma = np.tile(sigma, (2, 1))
sigma = np.ones_like(qs_target)
# plt.ion()
# fig, ax = plt.subplots(1, 1)
# for n in range(wf.Nt):
#     ax.plot(wf.X, mask[:, n])
#     ax.plot(wf.X, (qs_target[:wf.Nxi, n] - np.min(qs_target[:wf.Nxi, n])) / (np.max(qs_target[:wf.Nxi, n]) - np.min(qs_target[:wf.Nxi, n])))
#     ax.set_title("mask")
#     plt.draw()
#     plt.pause(0.02)
#     ax.cla()
# exit()

# Initial conditions for both primal and adjoint are defined here as they only need to defined once.
q0 = wf.InitialConditions_primal()
q0_adj = wf.InitialConditions_adjoint()

for opt_step in range(max_opt_steps):
    '''
     Forward calculation 
     '''
    if verbose: print("\n-------------------------------")
    if verbose: print("Optimization step: %d" % opt_step)
    time_odeint = perf_counter()  # save timing
    qs = wf.TimeIntegration_primal(q0, f, sigma)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Forward t_cpu = %1.3f" % time_odeint)

    '''
    Objective and costs for control
    '''
    J = Calc_Cost(qs, qs_target, f, lamda)
    if opt_step > 0:
        dJ = (J - J_list[-1]) / J_list[0]
        if verbose: print("Objective J/J[0] = %1.3f, dJ/J[0] = %1.3e J= %1.3e" % (J / J_list[0], dJ / J_list[0], J))
        if abs(dJ) < dJ_min: break  # stop if we are close to the minimum
    J_list.append(J)

    '''
    Adjoint calculation
    '''
    time_odeint = perf_counter()  # save timing
    qs_adj = wf.TimeIntegration_adjoint(q0_adj, qs, qs_target)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Backward t_cpu = %1.3f" % time_odeint)

    '''
     Update Control
    '''
    time_odeint = perf_counter() - time_odeint
    f = Update_Control(f, omega, lamda, sigma, q0, qs, qs_adj, qs_target, J, max_Armijo_iter=20, base_model_class=wf)
    if verbose: print(
        "Update Control t_cpu = %1.3f" % (perf_counter() - time_odeint))
    if opt_step == max_opt_steps - 1:
        print('warning maximal number of steps reached',
              "Objective J= %1.3e" % J)

if J > J_list[0]:
    print("optimization failed, Objective J/J[0] = %1.3f, J= %1.3e" % (J / J_list[0], J))


# Plot the results
pf = PlotFlow(wf.X, wf.Y, wf.t)
if Dimension == "1D":
    # Plot the Full Order Model (FOM)
    pf.plot1D(qs)
else:
    # Plot the Full Order Model (FOM)
    pf.plot2D(qs)
