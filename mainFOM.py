from Wildfire import Wildfire
from Plots import PlotFlow
from Helper import Calc_Cost, Update_Control, Force_masking, Calc_target_val, ControlSelectionMatrix_advection
import numpy as np
import sys
import os
from time import perf_counter
import jax
from jax.config import config
import jax.numpy as jnp
config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)

jnp.set_printoptions(threshold=sys.maxsize, linewidth=300)

import matplotlib.pyplot as plt

# Problem variables
Dimension = "1D"
Nxi = 200
Neta = 1
Nt = 400

# Wildfire solver initialization along with grid initialization
wf = Wildfire(Nxi=Nxi, Neta=Neta if Dimension == "1D" else Nxi, timesteps=Nt, cfl=0.3)
wf.Grid()
tm = "implicit_midpoint"  # Time stepping method
kwargs = {
    'dx': wf.dx,
    'dy': wf.dy,
    'dt': wf.dt,
    'Nx': wf.Nxi,
    'Ny': wf.Neta,
    'Nt': wf.Nt,
}


#%%
choose_selected_control = True
# Using fewer controls
n_c = 20  # Number of controls
f_tilde = jnp.zeros((n_c, wf.Nt))

# Selection matrix for the control input
wf.psi = ControlSelectionMatrix_advection(wf, n_c, shut_off_the_first_ncontrols=0)


#%% Solve for sigma
impath = "./data/FOM/"
os.makedirs(impath, exist_ok=True)
qs_org = wf.TimeIntegration_primal(wf.InitialConditions_primal(), f0=f_tilde, ti_method=tm)
sigma = Force_masking(qs_org, wf.X, wf.Y, wf.t, dim=Dimension)
jnp.save(impath + 'sigma.npy', sigma)
jnp.save(impath + 'qs_org.npy', qs_org)
sigma = jnp.load(impath + 'sigma.npy')


#%% Optimal control
max_opt_steps = 5
verbose = True
lamda = {'q_reg': 1e-2}  # weights and regularization parameter    # Lower the value of lamda means that we want a stronger forcing term. However higher its value we want weaker control
omega = 1  # initial step size for gradient update
dL_du_min = 1e-6  # Convergence criteria
f = jnp.zeros((wf.Nxi * wf.Neta, wf.Nt))  # Initial guess for the forcing term
qs_target = wf.TimeIntegration_primal_target(wf.InitialConditions_primal(), f0=f_tilde, ti_method=tm)
jnp.save(impath + 'qs_target.npy', qs_target)
J_list = []  # Collecting cost functional over the optimization steps
dL_du_list = []  # Collecting the gradient over the optimization steps

# Initial conditions for both primal and adjoint are defined here as they only need to defined once.
q0 = wf.InitialConditions_primal()
q0_adj = wf.InitialConditions_adjoint()


#%%
# If we choose selected controls then we just switch
if choose_selected_control:
    f = f_tilde
#%%
for opt_step in range(max_opt_steps):
    '''
    Forward calculation
    '''
    if verbose: print("\n-------------------------------")
    if verbose: print("Optimization step: %d" % opt_step)
    time_odeint = perf_counter()  # save timing
    qs = wf.TimeIntegration_primal(q0, f, ti_method=tm)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Forward t_cpu = %1.3f" % time_odeint)

    '''
    Objective and costs for control
    '''
    J = Calc_Cost(qs, qs_target, f, lamda, **kwargs)
    if opt_step == 0:
        pass
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
    qs_adj = wf.TimeIntegration_adjoint(q0_adj, f, qs, qs_target, ti_method=tm, dict_args=lamda)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Backward t_cpu = %1.3f" % time_odeint)


    '''
     Update Control
    '''
    time_odeint = perf_counter() - time_odeint
    f, J_opt, dL_du = Update_Control(f, omega, lamda, q0, qs_adj, qs_target, J,
                                     max_Armijo_iter=100, wf=wf, delta=1e-4, ti_method=tm,
                                     choose_selected_control=choose_selected_control, **kwargs)
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

# Final state corresponding to the optimal control f
qs_opt = wf.TimeIntegration_primal(q0, f, ti_method=tm)
f_opt = wf.psi @ f

#%%
# Save the optimized solution
jnp.save(impath + 'qs_opt.npy', qs_opt)
jnp.save(impath + 'qs_adj_opt.npy', qs_adj)
jnp.save(impath + 'f_opt.npy', f_opt)


#%%
# Load the results
qs_org = jnp.load(impath + 'qs_org.npy')
qs_opt = jnp.load(impath + 'qs_opt.npy')
qs_adj_opt = jnp.load(impath + 'qs_adj_opt.npy')
f_opt = jnp.load(impath + 'f_opt.npy')


# Plot the results
pf = PlotFlow(wf.X, wf.Y, wf.t)
if Dimension == "1D":
    pf.plot1D(qs_org, name="qs_org", immpath="./plots/FOM_1D/")
    pf.plot1D(qs_target, name="qs_target", immpath="./plots/FOM_1D/")
    pf.plot1D(qs_opt, name="qs_opt", immpath="./plots/FOM_1D/")
    pf.plot1D(qs_adj_opt, name="qs_adj_opt", immpath="./plots/FOM_1D/")
    pf.plot1D(f_opt, name="f_opt", immpath="./plots/FOM_1D/")
    pf.plot1D(sigma, name="sigma", immpath="./plots/FOM_1D/")
else:
    pf.plot2D(qs_org, name="qs_org", immpath="./plots/FOM_2D/",
              save_plot=True, plot_every=10, plot_at_all=True)
    pf.plot2D(qs_opt, name="qs_opt", immpath="./plots/FOM_2D/",
              save_plot=True, plot_every=10, plot_at_all=True)
