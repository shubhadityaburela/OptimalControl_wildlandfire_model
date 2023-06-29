from Wildfire import Wildfire
from Plots import PlotFlow
from Helper import Calc_Cost, Update_Control, Force_masking
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
Nt = 200

# Wildfire solver initialization along with grid initialization
wf = Wildfire(Nxi=Nxi, Neta=Neta if Dimension == "1D" else Nxi, timesteps=Nt)
wf.Grid()
tm = "bdf4"
f = jnp.zeros((wf.NumConservedVar * wf.Nxi * wf.Neta, wf.Nt))  # Initial guess for the forcing term
qs_target = jnp.concatenate((jnp.zeros((wf.Nxi * wf.Neta, wf.Nt)),
                            jnp.ones((wf.Nxi * wf.Neta, wf.Nt))), axis=0)  # Target value of the variables

impath = "./data/"
os.makedirs(impath, exist_ok=True)
calc_sigma = False
if calc_sigma:
    f_ = jnp.zeros((wf.NumConservedVar * wf.Nxi * wf.Neta, wf.Nt))
    s_ = jnp.zeros_like(qs_target)
    qs = wf.TimeIntegration_primal(wf.InitialConditions_primal(), f_, s_, ti_method="rk4")
    sigma = Force_masking(qs, wf.X, wf.Y, wf.t, dim=1)
    sigma = jnp.tile(sigma, (2, 1))
    jnp.save(impath + 'sigma.npy', sigma)
    sigma = jnp.load(impath + 'sigma.npy')

    # plt.ion()
    # fig, ax = plt.subplots(1, 1)
    # for n in range(wf.Nt):
    #     ax.plot(wf.X, sigma[wf.Nxi:, n], label="sigma")
    #     ax.plot(wf.X, qs[wf.Nxi:, n], label="S")
    #     ax.set_title("mask")
    #     ax.legend()
    #     plt.draw()
    #     plt.pause(1)
    #     ax.cla()

    # exit()
else:
    sigma = jnp.load(impath + 'sigma.npy')

# Optimal control
max_opt_steps = 2000
verbose = True
lamda = 1e-2  # regularization parameter
omega = 1  # initial step size for gradient update
dL_du_min = 1e-10

# Initial conditions for both primal and adjoint are defined here as they only need to defined once.
q0 = wf.InitialConditions_primal()
q0_adj = wf.InitialConditions_adjoint()
J_list = []  # Collecting cost functional over the optimization steps
dL_du_list = []  # Collecting the gradient over the optimization steps


#%%
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
    if opt_step == 0:
        jnp.save(impath + 'qs_org.npy', qs)
    else:
        dJ = (J - J_list[-1]) / J_list[0]
        if abs(dJ) == 0:
            if verbose: print("WARNING: dJ has turned 0...")
            break
    J_list.append(J)

    '''
    Adjoint calculation
    '''
    time_odeint = perf_counter()  # save timing
    qs_adj = wf.TimeIntegration_adjoint(q0_adj, f, qs, qs_target, ti_method=tm)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Backward t_cpu = %1.3f" % time_odeint)

    '''
     Update Control
    '''
    time_odeint = perf_counter() - time_odeint
    f, J_opt, dL_du = Update_Control(f, omega, lamda, sigma, q0, qs_adj, qs_target, J,
                                     max_Armijo_iter=100, wf=wf, ti_method=tm)
    dL_du_list.append(dL_du)
    if verbose: print(
        "Update Control t_cpu = %1.3f" % (perf_counter() - time_odeint))
    if verbose: print(
        f"J_opt : {J_opt}, ||dL_du|| = {dL_du}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du / dL_du_list[0]}"
    )

    # Convergence criteria
    if opt_step == max_opt_steps - 1:
        if verbose: print("\n\n-------------------------------")
        if verbose: print(
            f"WARNING... maximal number of steps reached, "
            f"J_opt : {J_opt}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du / dL_du_list[0]}"
        )
        break
    elif dL_du / dL_du_list[0] < dL_du_min:
        if verbose: print("\n\n-------------------------------")
        if verbose: print(
            f"Optimization converged with, "
            f"J_opt : {J_opt}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du / dL_du_list[0]}"
        )
        break
# Save the optimized solution
jnp.save(impath + 'qs_opt.npy', qs)

#%%
qs_org = jnp.load(impath + 'qs_org.npy')
qs_opt = jnp.load(impath + 'qs_opt.npy')
plt.ion()
fig, ax = plt.subplots(1, 1)
for n in range(wf.Nt):
    ax.plot(wf.X, qs_org[wf.Nxi:, n], label="qs_org")
    ax.plot(wf.X, qs_opt[wf.Nxi:, n], label="qs_opt")
    ax.set_title("mask")
    ax.legend()
    plt.draw()
    plt.pause(0.05)
    ax.cla()

#%%
# Plot the results
pf = PlotFlow(wf.X, wf.Y, wf.t)
if Dimension == "1D":
    # Plot the Full Order Model (FOM)
    pf.plot1D(qs)
else:
    # Plot the Full Order Model (FOM)
    pf.plot2D(qs)







    # plt.ion()
    # fig, ax = plt.subplots(1, 2)

    # ax[0].pcolormesh(qs[:wf.Nxi, :].T, cmap='YlOrRd')
    # ax[0].axis('auto')
    # ax[0].set_title("T")
    # ax[0].set_yticks([], [])
    # ax[0].set_xticks([], [])
    # ax[1].pcolormesh(qs[wf.Nxi:, :].T, cmap='YlGn')
    # ax[1].axis('auto')
    # ax[1].set_title("S")
    # ax[1].set_yticks([], [])
    # ax[1].set_xticks([], [])
    #
    # plt.draw()
    # plt.pause(0.5)
    # ax[0].cla()
    # ax[1].cla()