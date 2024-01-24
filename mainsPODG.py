from Coefficient_Matrix import CoefficientMatrix
from Update import Update_Control_sPODG
from advection import advection
from Plots import PlotFlow
from Helper import ControlSelectionMatrix_advection, Force_masking
from Helper_sPODG import subsample, Shifts_1D, srPCA_1D, findIntervals, get_T, make_target_term_matrices, \
    central_FDMatrix, get_T_one
from Costs import Calc_Cost_sPODG
import sys
import os
from time import perf_counter
import jax
from jax.config import config
import jax.numpy as jnp
from sklearn.utils.extmath import randomized_svd
import numpy as np
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
n_c = Nxi  # Number of controls
f_tilde = jnp.zeros((n_c, wf.Nt))

# Selection matrix for the control input
wf.psi, _, _ = ControlSelectionMatrix_advection(wf, n_c, shut_off_the_first_ncontrols=0, tilt_from=3*Nt//4)
wf.psi = jax.numpy.asarray(wf.psi)

#%% Solve for sigma
impath = "./data/sPODG/"
os.makedirs(impath, exist_ok=True)
qs_org = wf.TimeIntegration_primal(wf.InitialConditions_primal(), f0=f_tilde, ti_method=tm)
sigma = Force_masking(qs_org, wf.X, wf.Y, wf.t, dim=Dimension)
jnp.save(impath + 'sigma.npy', sigma)
jnp.save(impath + 'qs_org.npy', qs_org)
sigma = jnp.load(impath + 'sigma.npy')

#%% Optimal control
max_opt_steps = 30
verbose = True
lamda = {'q_reg': 1e-4}  # weights and regularization parameter    # Lower the value of lamda means that we want a stronger forcing term. However higher its value we want weaker control
omega = 1e-3  # initial step size for gradient update
dL_du_min = 1e-6  # Convergence criteria
f = jnp.zeros((wf.Nxi * wf.Neta, wf.Nt))  # Initial guess for the forcing term
qs_target = wf.TimeIntegration_primal_target(wf.InitialConditions_primal(), f0=f_tilde, ti_method=tm)
jnp.save(impath + 'qs_target.npy', qs_target)
J_list = []  # Collecting cost functional over the optimization steps
dL_du_list = []  # Collecting the gradient over the optimization steps
J_opt_list = []  # Collecting the optimal cost functional for plotting
dL_du_ratio_list = []  # Collecting the ratio of gradients for plotting
basis_refine_itr_list = []  # Collects the optimization step number at which the basis refinement is carried out
trunc_modes_list = []  # Collects the truncated number of modes at each basis refinement step

# Initial conditions for both primal and adjoint are defined here as they only need to defined once.
q0 = wf.InitialConditions_primal()
q0_adj = wf.InitialConditions_adjoint()

#%%
# If we choose selected controls then we just switch
if choose_selected_control:
    f = f_tilde

#%% ROM Variables
Num_sample = 200
# Nm_primal = 50
# Nm_adjoint = 50

D = central_FDMatrix(order=6, Nx=wf.Nxi, dx=wf.dx)
# Construct linear operators
Mat = CoefficientMatrix(orderDerivative=wf.firstderivativeOrder, Nxi=wf.Nxi,
                        Neta=wf.Neta, periodicity='Periodic', dx=wf.dx, dy=wf.dy)
# Convection matrix (Needs to be changed if the velocity is time dependent)
A_p = - (wf.v_x[0] * Mat.Grad_Xi_kron + wf.v_y[0] * Mat.Grad_Eta_kron)
A_a = - A_p.transpose()

# Generate the shift samples
delta_s = subsample(wf.X, num_sample=Num_sample)

# Extract transformation operators based on sub-sampled delta
T_delta, _ = get_T(delta_s, wf.X, wf.t)

# Basis update condition
refine = True
# %%
for opt_step in range(max_opt_steps):

    if verbose: print("\n-------------------------------")
    if verbose: print("Optimization step: %d" % opt_step)

    if refine:
        '''
        Forward calculation with primal at intermediate steps
        '''
        qs = wf.TimeIntegration_primal(q0, f, ti_method=tm)

        # Compute the shifts from the FOM
        delta_primal, _ = Shifts_1D(qs, wf.X, wf.t)

        # Compute the reduced basis of the uncontrolled system
        _, Nm_primal, Vs_p, _ = srPCA_1D(qs, delta_primal, wf.X, wf.t, spod_iter=50)

        # Construct the primal system matrices for the sPOD-Galerkin approach
        Vd_p, Wd_p, lhs_p, rhs_p, c_p = wf.sPOD_Galerkin_mat_primal(T_delta, Vs_p, A_p, samples=Num_sample)

        # Initial condition for dynamical simulation
        a_primal = wf.InitialConditions_primal_sPODG(q0, delta_s, Vd_p)

        basis_refine_itr_list.append(opt_step)
        trunc_modes_list.append(Nm_primal)
    '''
    Forward calculation
    '''
    time_odeint = perf_counter()  # save timing
    as_ = wf.TimeIntegration_primal_sPODG(lhs_p, rhs_p, c_p, a_primal, f, delta_s, ti_method=tm)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Forward t_cpu = %1.3f" % time_odeint)

    '''
    Objective and costs for control
    '''
    # Compute the interpolation weight and the interval in which the shift lies corresponding to which we compute the
    # V_delta and W_delta matrices
    intIds, weights = findIntervals(delta_s, as_[-1, :])
    J = Calc_Cost_sPODG(Vd_p, as_, qs_target, f, lamda, intIds, weights, **kwargs)
    if opt_step == 0:
        pass
    else:
        dJ = (J - J_list[-1]) / J_list[0]
        if abs(dJ) == 0:
            if verbose: print("WARNING: dJ has turned 0...")
            break
    J_list.append(J)

    '''
    Backward calculation with adjoint
    '''
    time_odeint = perf_counter()  # save timing
    qs_adj = wf.TimeIntegration_adjoint(q0_adj, f, qs, qs_target, ti_method=tm, dict_args=lamda)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Backward t_cpu = %1.3f" % time_odeint)

    '''
     Update Control
    '''
    time_odeint = perf_counter() - time_odeint
    f, J_opt, dL_du, refine = Update_Control_sPODG(f, omega, lamda, a_primal, qs_adj, qs_target,
                                                   J, Vd_p, lhs_p, rhs_p, c_p, delta_s, intIds, weights,
                                                   max_Armijo_iter=15, wf=wf,
                                                   delta=1e-4, ti_method=tm, **kwargs)

    # Save for plotting
    J_opt_list.append(J_opt)
    dL_du_list.append(dL_du)
    dL_du_ratio_list.append(dL_du / dL_du_list[0])

    '''
     Basis refinement
    '''
    if J_opt_list[opt_step] >= J_opt_list[opt_step - 1]:
        refine = True


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
            f"J_opt : {J_opt}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du / dL_du_list[0]}, "
            f"Number of basis refinements = {len(basis_refine_itr_list)}"
        )
        break
    elif dL_du / dL_du_list[0] < dL_du_min:
        if verbose: print("\n\n-------------------------------")
        if verbose: print(
            f"Optimization converged with, "
            f"J_opt : {J_opt}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du / dL_du_list[0]}, "
            f"Number of basis refinements = {len(basis_refine_itr_list)}"
        )
        break


# Compute the final state
as__ = wf.TimeIntegration_primal_sPODG(lhs_p, rhs_p, c_p, a_primal, f, delta_s, ti_method=tm)
as_online = as__[:Nm_primal]
delta_online = as__[-1]
qs = jnp.zeros_like(qs_target)
intIds, weights = findIntervals(delta_s, delta_online)
for i in range(f.shape[1]):
    V_delta = weights[i] * Vd_p[intIds[i]] + (1 - weights[i]) * Vd_p[intIds[i] + 1]
    qs = qs.at[:, i].set(V_delta @ as_online[:, i])

f_opt = wf.psi @ f

# %%
# Save the optimized solution
jnp.save(impath + 'qs_opt.npy', qs)
jnp.save(impath + 'qs_adj_opt.npy', qs_adj)
jnp.save(impath + 'f_opt.npy', f_opt)

# Save the convergence lists
np.save(impath + 'J_opt_list.npy', J_opt_list)
np.save(impath + 'dL_du_ratio_list.npy', dL_du_ratio_list)
np.save(impath + 'basis_refine_itr_list.npy', basis_refine_itr_list)
np.save(impath + 'trunc_modes_list.npy', trunc_modes_list)


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

    pf.plot1D_converg(J_opt_list, dL_du_ratio_list,
                      basis_refine_itr_list, trunc_modes_list,
                      immpath="./plots/sPODG_1D/")



















# if opt_step % 10 == 0:
    #     '''
    #     Forward calculation with primal at intermediate steps
    #     '''
    #     qs = wf.TimeIntegration_primal(q0, f, ti_method=tm)
    #
    #     # Compute the shifts from the FOM
    #     delta_primal, _ = Shifts_1D(qs, wf.X, wf.t)
    #
    #     # Compute the reduced basis of the uncontrolled system
    #     _, Nm_primal, wf.Vs_primal, _ = srPCA_1D(qs, delta_primal, wf.X, wf.t, spod_iter=100)
    #
    #     # Initial condition for dynamical simulation
    #     a_primal = wf.InitialConditions_primal_sPODG_tmp(q0)
    #
    #     # Construct the primal system matrices for the POD-Galerkin approach
    #     lhs_p, rhs_p, c_p = wf.sPOD_Galerkin_mat_primal_tmp(samples=1500)
    # '''
    # Forward calculation
    # '''
    # time_odeint = perf_counter()  # save timing
    # as_ = wf.TimeIntegration_primal_sPODG_tmp(lhs_p, rhs_p, c_p, a_primal, f, ti_method=tm)
    # time_odeint = perf_counter() - time_odeint
    # if verbose: print("Forward t_cpu = %1.3f" % time_odeint)
    # # as_ = as_.at[-1, 0].set(as_[-1, 1])
    #
    # '''
    # Objective and costs for control
    # '''
    # # Compute the interpolation weight and the interval in which the shift lies corresponding to which we compute the
    # # V_delta and W_delta matrices
    # intIds, weights = findIntervals(wf.delta_s[2], as_[-2, :])
    # J = Calc_Cost_sPODG_tmp(wf.V_delta_primal, as_, qs_target, f, lamda, intIds, weights, **kwargs)
    # if opt_step == 0:
    #     pass
    # else:
    #     dJ = (J - J_list[-1]) / J_list[0]
    #     if abs(dJ) == 0:
    #         if verbose: print("WARNING: dJ has turned 0...")
    #         break
    # J_list.append(J)
    #
    # if opt_step % 10 == 0:
    #     '''
    #     Backward calculation with adjoint at intermediate steps
    #     '''
    #     qs_adj = wf.TimeIntegration_adjoint(q0_adj, f, qs, qs_target, ti_method=tm, dict_args=lamda)
    #
    #     # Compute the shifts from the FOM
    #     delta_adjoint, _ = Shifts_1D(qs_adj, wf.X, wf.t)
    #
    #     # Compute the reduced basis of the uncontrolled system
    #     _, Nm_adjoint, wf.Vs_adjoint, tmp_low_FOM = srPCA_1D(qs_adj, delta_primal, wf.X, wf.t, spod_iter=10)
    #
    #     # Initial condition for dynamical simulation
    #     a_adjoint = wf.InitialConditions_adjoint_sPODG_tmp(q0_adj)
    #
    #     # Construct the primal system matrices for the POD-Galerkin approach
    #     lhs_a, rhs_a = wf.sPOD_Galerkin_mat_adjoint_tmp(samples=1500)
    # '''
    # Backward calculation with reduced system
    # '''
    # time_odeint = perf_counter()  # save timing
    # as_adj = wf.TimeIntegration_adjoint_sPODG_tmp(lhs_a, rhs_a, a_adjoint, f, as_, qs_target, ti_method=tm)
    # time_odeint = perf_counter() - time_odeint
    # if verbose: print("Backward t_cpu = %1.3f" % time_odeint)
    # as_online = as_adj
    # intIds, weights = findIntervals(wf.delta_s[2], as_[-2, :])
    # tmp = jnp.zeros_like(qs_target)
    # for i in range(f.shape[1]):
    #     V_delta = weights[i] * wf.V_delta_adjoint[intIds[i]] + (1 - weights[i]) * wf.V_delta_adjoint[intIds[i] + 1]
    #     tmp = tmp.at[:, i].set(V_delta @ as_online[:, i])
    # tmp_low = wf.Vs_adjoint @ as_online
