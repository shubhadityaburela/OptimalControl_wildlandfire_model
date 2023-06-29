from Wildfire import Wildfire
from Plots import PlotFlow
from Helper import Cost_functional, Force_masking
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
Nxi = 250
Neta = 1
Nt = 200

# Wildfire solver initialization along with grid initialization
wf = Wildfire(Nxi=Nxi, Neta=Neta if Dimension == "1D" else Nxi, timesteps=Nt)
wf.Grid()
tm = "rk4"
f = jnp.zeros((wf.NumConservedVar * wf.Nxi * wf.Neta, wf.Nt))  # Initial guess for the forcing term
qs_target = jnp.concatenate((jnp.zeros((wf.Nxi * wf.Neta, wf.Nt)),
                            jnp.ones((wf.Nxi * wf.Neta, wf.Nt))), axis=0)  # Target value of the variables

impath = "./data/"
os.makedirs(impath, exist_ok=True)
calc_sigma = True
if calc_sigma:
    f_ = jnp.zeros((wf.NumConservedVar * wf.Nxi * wf.Neta, wf.Nt))
    s_ = jnp.zeros_like(qs_target)
    qs = wf.TimeIntegration(wf.InitialConditions(), f_, s_, ti_method="rk4")
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
dJ_dx_min = 1e-10
J_list = []  # Collecting cost functional over the optimization steps
dJ_dx_list = []  # Collecting the gradient over the optimization steps


# Initial conditions for primal is defined here as they only need to defined once.
q0 = wf.InitialConditions()
cf = Cost_functional(qs_target, sigma, q0, lamda, wf, ti_method=tm)

'''
Forward calculation and objective function
'''
if verbose: print("\n-------------------------------")
qs = wf.TimeIntegration(q0, f, sigma, ti_method=tm)
jnp.save(impath + 'qs_org.npy', qs)
J = cf.C_JAX_cost(qs, f)

if verbose: print("Optimization start.......")
for itr in range(max_opt_steps):
    '''
    JAX derivative computation
    '''
    time_odeint = perf_counter()  # save timing
    dJ_dx = jax.jit(jax.jacfwd(cf.C_JAX))(f)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Gradient computation t_cpu = %1.3f" % time_odeint)
    dJ_dx_list.append(jnp.linalg.norm(dJ_dx))

    '''
    JAX control update
    '''
    time_odeint = perf_counter() - time_odeint
    f, J, dJ_dx_ = cf.C_JAX_update(f, omega, dJ_dx, J, max_Armijo_iter=100)
    if verbose: print(
        "Update Control t_cpu = %1.3f" % (perf_counter() - time_odeint))
    if verbose: print(
        f"J_opt : {J}, ||dJ_dx|| = {dJ_dx_}, ||dJ_dx||_{itr} / ||dJ_dx||_0 = {dJ_dx_ / dJ_dx_list[0]}"
    )

    # Convergence criteria
    if itr == max_opt_steps - 1:
        if verbose: print("\n\n-------------------------------")
        if verbose: print(
            f"WARNING... maximal number of steps reached, "
            f"J_opt : {J}, ||dJ_dx||_{itr} / ||dJ_dx||_0 = {dJ_dx_ / dJ_dx_list[0]}"
        )
        break
    elif dJ_dx_ / dJ_dx_list[0] < dJ_dx_min:
        if verbose: print("\n\n-------------------------------")
        if verbose: print(
            f"Optimization converged with, "
            f"J_opt : {J}, ||dJ_dx||_{itr} / ||dJ_dx||_0 = {dJ_dx_ / dJ_dx_list[0]}"
        )
        break


qs = wf.TimeIntegration(q0, f, sigma, ti_method=tm)

# Plot the results
pf = PlotFlow(wf.X, wf.Y, wf.t)
if Dimension == "1D":
    # Plot the Full Order Model (FOM)
    pf.plot1D(qs)
else:
    # Plot the Full Order Model (FOM)
    pf.plot2D(qs)
