from Wildfire import Wildfire
from Plots import PlotFlow
from Helper import Calc_Cost, Update_Control
from Wildfire import Adjoint_Matrices, Force_masking
import numpy as np
import sys
import os
from time import perf_counter
import jax
from jax.config import config
import jax.numpy as jnp
config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)

np.set_printoptions(threshold=sys.maxsize, linewidth=300)

import matplotlib.pyplot as plt

# Problem variables
Dimension = "1D"
Nxi = 250
Neta = 1
Nt = 100

# Wildfire solver initialization along with grid initialization
wf = Wildfire(Nxi=Nxi, Neta=Neta if Dimension == "1D" else Nxi, timesteps=Nt)
wf.Grid()
tm = "bdf4"

# Optimal control
max_opt_steps = 2000
verbose = True
lamda = 10.0  # regularization parameter
omega = 5e-3  # initial step size for gradient update

f = 1.0 * jnp.ones((wf.NumConservedVar * wf.Nxi * wf.Neta, wf.Nt))  # Initial guess for the forcing term
qs_target = jnp.concatenate((jnp.zeros((wf.Nxi * wf.Neta, wf.Nt)),
                            jnp.ones((wf.Nxi * wf.Neta, wf.Nt))), axis=0)  # Target value of the variables
sigma = jnp.ones_like(qs_target)

J_list = []  # Collecting cost functional over the optimization steps
dJ_min = 1e-10

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
    qs = wf.TimeIntegration_primal(q0, f, sigma, ti_method=tm)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Forward t_cpu = %1.3f" % time_odeint)

    '''
    Objective and costs for control
    '''
    J = Calc_Cost(qs, qs_target, f, lamda)
    # if opt_step > 0:
    #     dJ = (J - J_list[-1]) / J_list[0]
    #     # if verbose: print("Objective J= %1.3e" % J)
    #     if abs(dJ) < dJ_min: break  # stop if we are close to the minimum
    J_list.append(J)

    '''
    Adjoint calculation
    '''
    # if tm == "rk4":
    #     print("Use bdf4 instead!!!")
    #     exit()
    # elif tm == "bdf4":
    time_odeint = perf_counter()  # save timing
    qs_adj = wf.TimeIntegration_adjoint(q0_adj, f, qs, qs_target, ti_method=tm)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Backward t_cpu = %1.3f" % time_odeint)


    '''
     Update Control
    '''
    time_odeint = perf_counter() - time_odeint
    f, J_opt, dJ_dx_norm = Update_Control(f, omega, lamda, sigma, q0, qs_adj, qs_target, J,
                                          max_Armijo_iter=100, wf=wf, ti_method=tm)
    if verbose: print(
        "Update Control t_cpu = %1.3f" % (perf_counter() - time_odeint))
    if verbose: print(
        f"J_opt : {J_opt}, ||dJ_dx|| = {dJ_dx_norm}"
    )
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





# # Compute the masking array
# qs = wf.TimeIntegration_primal(wf.InitialConditions_primal(),
#                                np.zeros((wf.Nxi * wf.Neta, wf.Nt)))
# sigma = Force_masking(qs, wf.X, wf.Y, wf.t, dim=1)
# sigma = np.tile(sigma, (2, 1))



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