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

from mpl_toolkits.axes_grid1 import make_axes_locatable

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
f = 5 * jnp.ones((wf.NumConservedVar * wf.Nxi * wf.Neta, wf.Nt))  # Initial guess for the forcing term
qs_target = jnp.concatenate((jnp.zeros((wf.Nxi * wf.Neta, wf.Nt)),
                            jnp.ones((wf.Nxi * wf.Neta, wf.Nt))), axis=0)  # Target value of the variables

impath = "./data/"
os.makedirs(impath, exist_ok=True)
calc_sigma = False
if calc_sigma:
    qs = wf.TimeIntegration_primal(wf.InitialConditions_primal(), f, jnp.ones_like(qs_target), ti_method=tm)
    sigma = Force_masking(qs, wf.X, wf.Y, wf.t, dim=1)
    sigma = np.tile(sigma, (2, 1))
    jnp.save(impath + 'sigma.npy', sigma)

    plt.ion()
    fig, ax = plt.subplots(1, 1)
    for n in range(wf.Nt):
        ax.plot(wf.X, sigma[wf.Nxi:, n], label="sigma")
        ax.plot(wf.X, qs[wf.Nxi:, n], label="S")
        ax.set_title("mask")
        ax.legend()
        plt.draw()
        plt.pause(1)
        ax.cla()

    exit()
else:
    sigma = jnp.load(impath + 'sigma.npy')

# Optimal control
max_opt_steps = 100
verbose = True
lamda = 1  # regularization parameter
omega = 1e-4  # initial step size for gradient update
dJ_min = 1e-10

# Initial conditions for both primal and adjoint are defined here as they only need to defined once.
q0 = wf.InitialConditions_primal()
q0_adj = wf.InitialConditions_adjoint()
J_list = []  # Collecting cost functional over the optimization steps

plt.ion()
fig, ax = plt.subplots(1, 2)
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
    if opt_step > 0:
        dJ = (J - J_list[-1]) / J_list[0]
        # if verbose: print("Objective J= %1.3e" % J)
        # if abs(dJ) < dJ_min: break  # stop if we are close to the minimum
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



    ax[0].pcolormesh(qs[:wf.Nxi, :].T, cmap='YlOrRd')
    ax[0].axis('auto')
    ax[0].set_title("T")
    ax[0].set_yticks([], [])
    ax[0].set_xticks([], [])
    ax[1].pcolormesh(qs[wf.Nxi:, :].T, cmap='YlGn')
    ax[1].axis('auto')
    ax[1].set_title("S")
    ax[1].set_yticks([], [])
    ax[1].set_xticks([], [])

    plt.draw()
    plt.pause(0.5)
    ax[0].cla()
    ax[1].cla()

if J > J_list[0]:
    print("optimization failed, Objective J/J[0] = %1.3f, J= %1.3e" % (J / J_list[0], J))

# qs, qs_adj = wf.ReDim(qs, qs_adj)
#
# # Plot the results
# # qs, qs_adj = wf.ReDim(qs, qs_adj)
# pf = PlotFlow(wf.X, wf.Y, wf.t)
# if Dimension == "1D":
#     # Plot the Full Order Model (FOM)
#     pf.plot1D(qs)
# else:
#     # Plot the Full Order Model (FOM)
#     pf.plot2D(qs)

















# #%%
# impath1 = "./data/dt_1/"
# os.makedirs(impath1, exist_ok=True)
#
# impath2 = "./data/dt_0.1/"
# os.makedirs(impath2, exist_ok=True)
#
# impath3 = "./data/dt_0.01/"
# os.makedirs(impath3, exist_ok=True)
#
# impath4 = "./data/dt_0.01__/"
# os.makedirs(impath4, exist_ok=True)
#
# # #%%
# # np.save(impath4 + 'bdf4_pr.npy', qs)
# # np.save(impath4 + 'bdf4_ad.npy', qs_adj)
#
#
# #%%
# qs_rk4_1 = np.load(impath1 + 'rk4_pr.npy')
# qs_bdf4_1 = np.load(impath1 + 'bdf4_pr.npy')
# qs_adj_rk4_1 = np.load(impath1 + 'rk4_ad.npy')
# qs_adj_bdf4_1 = np.load(impath1 + 'bdf4_ad.npy')
#
# qs_rk4_01 = np.load(impath2 + 'rk4_pr.npy')
# qs_bdf4_01 = np.load(impath2 + 'bdf4_pr.npy')
# qs_adj_rk4_01 = np.load(impath2 + 'rk4_ad.npy')
# qs_adj_bdf4_01 = np.load(impath2 + 'bdf4_ad.npy')
#
# qs_rk4_001 = np.load(impath3 + 'rk4_pr.npy')
# qs_bdf4_001 = np.load(impath3 + 'bdf4_pr.npy')
# qs_adj_rk4_001 = np.load(impath3 + 'rk4_ad.npy')
# qs_adj_bdf4_001 = np.load(impath3 + 'bdf4_ad.npy')
#
# qs_rk4_001__ = np.load(impath4 + 'rk4_pr.npy')
# qs_bdf4_001__ = np.load(impath4 + 'bdf4_pr.npy')
# qs_adj_rk4_001__ = np.load(impath4 + 'rk4_ad.npy')
# qs_adj_bdf4_001__ = np.load(impath4 + 'bdf4_ad.npy')
#
#
#
#
# #%%
# Nt = 5000
# plot_every = 1
# plt.ion()
# fig, ax = plt.subplots(1, 1)
# for n in range(Nt):
#     if n % plot_every == 0:
#         ax.plot(wf.X, qs_adj_rk4_1[:wf.Nxi, n], label="rk4")
#         ax.plot(wf.X, qs_adj[:wf.Nxi, n], label="bdf4")
#         ax.set_title(f"{n}")
#         ax.legend()
#         plt.draw()
#         plt.pause(0.1)
#         ax.cla()
# #%%





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