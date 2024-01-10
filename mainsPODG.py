from advection import advection
from Plots import PlotFlow
from Helper import Calc_Cost_PODG, Update_Control_ROM, Calc_target_val, \
    compute_red_basis, ControlSelectionMatrix, ControlSelectionMatrix_advection, Force_masking, Calc_Cost_sPODG, \
    Update_Control_sPOD_ROM
import numpy as np
from Helper_sPODG import Shifts_1D, srPCA_1D, get_T, get_online_state, findIntervals
import sys
import os
from time import perf_counter
import jax
from jax.config import config
import jax.numpy as jnp
import time

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
wf = advection(Nxi=Nxi, Neta=Neta if Dimension == "1D" else Nxi, timesteps=Nt, cfl=0.3, tilt_from=3*Nt//4)
wf.Grid()
tm = "rk4"  # Time stepping method
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
n_c = 200  # Number of controls
f_tilde = jnp.zeros((n_c, wf.Nt))

# Selection matrix for the control input
wf.psi = ControlSelectionMatrix_advection(wf, n_c, shut_off_the_first_ncontrols=0)

#%% Solve for sigma
impath = "./data/sPODG/"
os.makedirs(impath, exist_ok=True)
qs_org = wf.TimeIntegration_primal(wf.InitialConditions_primal(), f0=f_tilde, ti_method=tm)
sigma = Force_masking(qs_org, wf.X, wf.Y, wf.t, dim=Dimension)
jnp.save(impath + 'sigma.npy', sigma)
jnp.save(impath + 'qs_org.npy', qs_org)
sigma = jnp.load(impath + 'sigma.npy')

#%% Optimal control
max_opt_steps = 1
verbose = True
lamda = {'q_reg': 1e-2}  # weights and regularization parameter    # Lower the value of lamda means that we want a stronger forcing term. However higher its value we want weaker control
omega = 1e-3  # initial step size for gradient update
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

# %% ROM variables
# Compute the shifts from the FOM
delta, _ = Shifts_1D(qs_org, wf.X, wf.t)

# Compute the reduced basis of the uncontrolled system
_, Nm, wf.Vs, _ = srPCA_1D(qs_org, delta, wf.X, wf.t, spod_iter=100)

# Initial condition for dynamical simulation
a_primal = wf.InitialConditions_primal_sPODG(q0)
a_adjoint = wf.InitialConditions_adjoint_sPODG(q0_adj)

# Construct the primal system matrices for the POD-Galerkin approach
lhs, rhs, c, ct = wf.sPOD_Galerkin_mat_primal(samples=1500)

# # Construct the adjoint system matrices for the POD-Galerkin approach
# wf.sPOD_Galerkin_mat_adjoint()

# %%
for opt_step in range(max_opt_steps):

    if verbose: print("\n-------------------------------")
    if verbose: print("Optimization step: %d" % opt_step)

    '''
    Forward calculation
    '''
    time_odeint = perf_counter()  # save timing
    as_ = wf.TimeIntegration_primal_sPODG(lhs, rhs, c, a_primal, f, ti_method=tm)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Forward t_cpu = %1.3f" % time_odeint)

    '''
    Objective and costs for control
    '''
    # Compute the interpolation weight and the interval in which the shift lies corresponding to which we compute the
    # V_delta and W_delta matrices
    intIds, weights = findIntervals(wf.delta_s[2], as_[-1, :])
    J = Calc_Cost_sPODG(wf.V_delta, as_[0, :], qs_target, f, lamda, intIds, weights, **kwargs)
    if opt_step == 0:
        pass
    else:
        dJ = (J - J_list[-1]) / J_list[0]
        if abs(dJ) == 0:
            if verbose: print("WARNING: dJ has turned 0...")
            break
    J_list.append(J)

    # '''
    # Adjoint calculation
    # '''
    # time_odeint = perf_counter()  # save timing
    # as_adj = wf.TimeIntegration_adjoint_PODG(a_adjoint, f, as_, qs_target, ti_method=tm)
    # time_odeint = perf_counter() - time_odeint
    # if verbose: print("Backward t_cpu = %1.3f" % time_odeint)


    as_adj = jnp.ones_like(as_)
    '''
     Update Control
    '''
    time_odeint = perf_counter() - time_odeint
    f, J_opt, dL_du = Update_Control_sPOD_ROM(f, omega, lamda, lhs, rhs, c, a_primal, as_, as_adj, qs_target,
                                              J, ct, intIds, weights, max_Armijo_iter=100, wf=wf,
                                              delta=1e-4, ti_method=tm, red_nl=True, **kwargs)
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


# Compute the final state
as__ = wf.TimeIntegration_primal_sPODG(lhs, rhs, c, a_primal, f, ti_method=tm)
as_online = as__[:Nm]
delta_online = as__[Nm:]
_, T_trafo_1 = get_T(delta_online, wf.X, wf.t)
qs = get_online_state(T_trafo_1, wf.Vs, as_online, wf.X, wf.t)

as_adj_online = as_adj[:Nm]
delta_adj_online = as_adj[Nm:]
_, T_trafo_2 = get_T(delta_adj_online, wf.X, wf.t)
qs_adj = get_online_state(T_trafo_2, wf.Vs, as_adj_online, wf.X, wf.t)

f_opt = wf.psi @ f

# %%
# Save the optimized solution
jnp.save(impath + 'qs_opt.npy', qs)
jnp.save(impath + 'qs_adj_opt.npy', qs_adj)
jnp.save(impath + 'f_opt.npy', f_opt)


# %%
# Load the results
qs_org = jnp.load(impath + 'qs_org.npy')
qs_opt = jnp.load(impath + 'qs_opt.npy')
qs_adj_opt = jnp.load(impath + 'qs_adj_opt.npy')
f_opt = jnp.load(impath + 'f_opt.npy')


# Plot the results
pf = PlotFlow(wf.X, wf.Y, wf.t)
if Dimension == "1D":
    pf.plot1D(qs_org, name="qs_org", immpath="./plots/sPODG_1D/")
    pf.plot1D(qs_target, name="qs_target", immpath="./plots/sPODG_1D/")
    pf.plot1D(qs_opt, name="qs_opt", immpath="./plots/sPODG_1D/")
    pf.plot1D(qs_adj_opt, name="qs_adj_opt", immpath="./plots/sPODG_1D/")
    pf.plot1D(f_opt, name="f_opt", immpath="./plots/sPODG_1D/")
    pf.plot1D(sigma, name="sigma", immpath="./plots/sPODG_1D/")