"""
This file is the adaptive version. THIS IS NOT TO BE USED FOR OUR STUDIES
"""

from Coefficient_Matrix import CoefficientMatrix
from Costs import Calc_Cost_PODG, Calc_Cost
from Helper import ControlSelectionMatrix_advection, Force_masking, compute_red_basis
from Update import Update_Control_PODG
from advection import advection
from Plots import PlotFlow
import sys
import numpy as np
import os
from time import perf_counter
import time

import matplotlib.pyplot as plt

# Problem variables
Dimension = "1D"
Nxi = 200
Neta = 1
Nt = 400

# solver initialization along with grid initialization
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
n_c = Nxi // 10  # Number of controls
f_tilde = np.zeros((n_c, wf.Nt))

# Selection matrix for the control input
psi, _, _ = ControlSelectionMatrix_advection(wf, n_c, shut_off_the_first_ncontrols=0, tilt_from=3*Nt//4)


#%% Assemble the linear operators
# Construct linear operators
Mat = CoefficientMatrix(orderDerivative=wf.firstderivativeOrder, Nxi=wf.Nxi,
                        Neta=wf.Neta, periodicity='Periodic', dx=wf.dx, dy=wf.dy)
# Convection matrix (Needs to be changed if the velocity is time dependent)
A_p = - (wf.v_x[0] * Mat.Grad_Xi_kron + wf.v_y[0] * Mat.Grad_Eta_kron)
A_a = A_p.transpose()

#%% Solve for sigma
impath = "./data/PODG/FOTR/adaptive/"
immpath = "./plots/PODG_1D/FOTR/adaptive/"
os.makedirs(impath, exist_ok=True)
qs_org = wf.TimeIntegration_primal(wf.InitialConditions_primal(), f_tilde, A_p, psi, ti_method=tm)
sigma = Force_masking(qs_org, wf.X, wf.Y, wf.t, dim=Dimension)
np.save(impath + 'sigma.npy', sigma)
np.save(impath + 'qs_org.npy', qs_org)
sigma = np.load(impath + 'sigma.npy')


#%% Optimal control
max_opt_steps = 100000
verbose = True
lamda = {'q_reg': 1e-3}  # weights and regularization parameter    # Lower the value of lamda means that we want a stronger forcing term. However higher its value we want weaker control
omega = 1e-3  # initial step size for gradient update
dL_du_min = 1e-4  # Convergence criteria
f = np.zeros((wf.Nxi * wf.Neta, wf.Nt))  # Initial guess for the forcing term
qs_target = wf.TimeIntegration_primal_target(wf.InitialConditions_primal(), f_tilde, A_p, psi, ti_method=tm)
np.save(impath + 'qs_target.npy', qs_target)
J_list = []  # Collecting cost functional over the optimization steps
dL_du_list = []  # Collecting the gradient over the optimization steps
J_opt_list = []  # Collecting the optimal cost functional for plotting
dL_du_ratio_list = []  # Collecting the ratio of gradients for plotting
basis_refine_itr_list = []  # Collects the optimization step number at which the basis refinement is carried out
trunc_modes_primal_list = []  # Collects the truncated number of modes for primal at each basis refinement step
trunc_modes_adjoint_list = []  # Collects the truncated number of modes for adjoint at each basis refinement step


# Initial conditions for both primal and adjoint are defined here as they only need to defined once.
q0 = wf.InitialConditions_primal()
q0_adj = wf.InitialConditions_adjoint()


#%%
# If we choose selected controls then we just switch
if choose_selected_control:
    f = f_tilde

# %% ROM variables
# Modes for the ROM
n_rom_primal = 1
n_rom_adjoint = 1
n_incr = 1
n_stag_stop = 1

# Basis update condition
refine = True
stagnate = 0

start = time.time()
# %%
for opt_step in range(max_opt_steps):

    if verbose: print("\n-------------------------------")
    if verbose: print("Optimization step: %d" % opt_step)

    if refine:
        time_odeint = perf_counter()  # save timing
        '''
        Forward calculation with primal at intermediate steps
        '''
        qs = wf.TimeIntegration_primal(q0, f, A_p, psi, ti_method=tm)

        # Compute the reduced basis
        if n_rom_primal < Nxi:
            n_rom_primal = n_rom_primal + n_stag_stop
        V_p, qs_POD = compute_red_basis(qs, n_rom_primal)
        err = np.linalg.norm(qs - qs_POD) / np.linalg.norm(qs)
        if verbose: print(f"Relative error for primal: {err}, with n_rom_primal: {n_rom_primal}")
        while err > 1e-4:
            n_rom_primal = n_rom_primal + n_incr
            V_p, qs_POD = compute_red_basis(qs, n_rom_primal)
            err = np.linalg.norm(qs - qs_POD) / np.linalg.norm(qs)
            if verbose: print(f"Refine by adding more modes..... Relative error for primal: {err} with n_rom_primal: {n_rom_primal}")

        # Initial condition for dynamical simulation
        a_p = wf.InitialConditions_primal_PODG(V_p, q0)

        # Construct the primal system matrices for the POD-Galerkin approach
        Ar_p, psir_p = wf.POD_Galerkin_mat_primal(A_p, V_p, psi)

        basis_refine_itr_list.append(opt_step)
        trunc_modes_primal_list.append(n_rom_primal)

        time_odeint = perf_counter() - time_odeint
        if verbose: print("Forward basis refinement t_cpu = %1.3f" % time_odeint)

    '''
    Forward calculation with reduced system
    '''
    time_odeint = perf_counter()  # save timing
    as_ = wf.TimeIntegration_primal_PODG(a_p, f, Ar_p, psir_p, ti_method=tm)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Forward t_cpu = %1.3f" % time_odeint)


    '''
    Objective and costs for control
    '''
    time_odeint = perf_counter()  # save timing
    J = Calc_Cost_PODG(V_p, as_, qs_target, f, psi, lamda, **kwargs)
    time_odeint = perf_counter() - time_odeint
    print("Calc_Cost t_cpu = %1.6f" % time_odeint)
    if opt_step == 0:
        pass
    else:
        dJ = (J - J_list[-1]) / J_list[0]
        if abs(dJ) == 0:
            print("WARNING: dJ has turned 0...")
            break
    J_list.append(J)

    if refine:
        time_odeint = perf_counter()  # save timing
        '''
        Backward calculation with adjoint at intermediate steps
        '''
        qs_adj = wf.TimeIntegration_adjoint(q0_adj, f, qs, qs_target, A_a, ti_method=tm, dict_args=lamda)

        # Compute the reduced basis
        if n_rom_adjoint < Nxi:
            n_rom_adjoint = n_rom_adjoint + n_stag_stop
        V_a, qs_adj_POD = compute_red_basis(qs_adj, n_rom_adjoint)
        err = np.linalg.norm(qs_adj - qs_adj_POD) / np.linalg.norm(qs_adj)
        if verbose: print(f"Relative error for adjoint: {err}, with n_rom_adjoint: {n_rom_adjoint}")
        while err > 1e-4:
            n_rom_adjoint = n_rom_adjoint + n_incr
            V_a, qs_adj_POD = compute_red_basis(qs_adj, n_rom_adjoint)
            err = np.linalg.norm(qs_adj - qs_adj_POD) / np.linalg.norm(qs_adj)
            if verbose: print(
                f"Refine by adding more modes..... Relative error for adjoint: {err} with n_rom_adjoint: {n_rom_adjoint}")

        # Initial condition for dynamical simulation
        a_a = wf.InitialConditions_adjoint_PODG(V_a, q0_adj)

        # Construct the adjoint system matrices for the POD-Galerkin approach
        Ar_a, Tr_a, Tarr_a, psir_a = wf.POD_Galerkin_mat_adjoint(V_a, A_p, V_p, qs_target, psi)

        trunc_modes_adjoint_list.append(n_rom_adjoint)

        time_odeint = perf_counter() - time_odeint
        if verbose: print("Backward basis refinement t_cpu = %1.3f" % time_odeint)
    '''
    Backward calculation with reduced system
    '''
    time_odeint = perf_counter()  # save timing
    as_adj = wf.TimeIntegration_adjoint_PODG(a_a, f, as_, Ar_a, Tr_a, Tarr_a, ti_method=tm)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Backward t_cpu = %1.3f" % time_odeint)

    '''
     Update Control
    '''
    time_odeint = perf_counter()
    f, J_opt, dL_du, refine, stag = Update_Control_PODG(f, a_p, as_adj, qs_target, V_p, Ar_p, psir_p, psir_a, psi, J, omega,
                                                        lamda, max_Armijo_iter=18, wf=wf, delta=1e-4, ti_method=tm,
                                                        verbose=verbose, **kwargs)
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
    print(
        f"J_opt : {J_opt}, ||dL_du|| = {dL_du}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du / dL_du_list[0]}"
    )

    # Convergence criteria
    if opt_step == max_opt_steps - 1:
        print("\n\n-------------------------------")
        print(
            f"WARNING... maximal number of steps reached, "
            f"J_opt : {J_opt}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du / dL_du_list[0]}, "
            f"Number of basis refinements = {len(basis_refine_itr_list)}"
        )
        break
    elif dL_du / dL_du_list[0] < dL_du_min:
        print("\n\n-------------------------------")
        print(
            f"Optimization converged with, "
            f"J_opt : {J_opt}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du / dL_du_list[0]}, "
            f"Number of basis refinements = {len(basis_refine_itr_list)}"
        )
        break

    '''
    Checking for stagnation
    '''
    stagnate = stagnate + stag
    if stagnate > 5000:
        print("\n\n-------------------------------")
        print(
            f"WARNING... Armijo starting to stagnate, "
            f"J_opt : {J_opt}, ||dL_du||_{opt_step} / ||dL_du||_0 = {dL_du / dL_du_list[0]}, "
            f"Number of basis refinements = {len(basis_refine_itr_list)}"
        )
        break


# Compute the final state
as__ = wf.TimeIntegration_primal_PODG(a_p, f, Ar_p, psir_p, ti_method=tm)
qs = V_p @ as__
qs_adj = V_a @ as_adj
f_opt = psi @ f

# Compute the cost with the optimal control
qs_opt_full = wf.TimeIntegration_primal(q0, f, A_p, psi, ti_method=tm)
J = Calc_Cost(qs_opt_full, qs_target, f, lamda, **kwargs)
print("\n")
print(f"J with respect to the optimal control for FOM: {J}")

end = time.time()
print("\n")
print("Total time elapsed = %1.3f" % (end - start))
# %%
# Save the optimized solution
np.save(impath + 'qs_opt.npy', qs)
np.save(impath + 'qs_adj_opt.npy', qs_adj)
np.save(impath + 'f_opt.npy', f_opt)


# %%
# Load the results
qs_org = np.load(impath + 'qs_org.npy')
qs_opt = np.load(impath + 'qs_opt.npy')
qs_adj_opt = np.load(impath + 'qs_adj_opt.npy')
f_opt = np.load(impath + 'f_opt.npy')

# Save the convergence lists
np.save(impath + 'J_opt_list.npy', J_opt_list)
np.save(impath + 'dL_du_ratio_list.npy', dL_du_ratio_list)
np.save(impath + 'basis_refine_itr_list.npy', basis_refine_itr_list)
np.save(impath + 'trunc_modes_primal_list.npy', trunc_modes_primal_list)
np.save(impath + 'trunc_modes_adjoint_list.npy', trunc_modes_adjoint_list)



# Plot the results
pf = PlotFlow(wf.X, wf.Y, wf.t)
if Dimension == "1D":
    pf.plot1D(qs_org, name="qs_org", immpath=immpath)
    pf.plot1D(qs_target, name="qs_target", immpath=immpath)
    pf.plot1D(qs_opt, name="qs_opt", immpath=immpath)
    pf.plot1D(qs_adj_opt, name="qs_adj_opt", immpath=immpath)
    pf.plot1D(f_opt, name="f_opt", immpath=immpath)
    pf.plot1D(sigma, name="sigma", immpath=immpath)

    pf.plot1D_ROM_converg(J_opt_list, dL_du_ratio_list,
                          basis_refine_itr_list,
                          trunc_modes_primal_list,
                          trunc_modes_adjoint_list,
                          immpath=immpath)