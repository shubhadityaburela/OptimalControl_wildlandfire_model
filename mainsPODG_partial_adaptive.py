from Coefficient_Matrix import CoefficientMatrix
from Update import Update_Control_sPODG
from advection import advection
from Plots import PlotFlow
from Helper import ControlSelectionMatrix_advection, Force_masking, compute_red_basis
from Helper_sPODG import subsample, Shifts_1D, srPCA_1D, findIntervals, get_T, make_target_term_matrices, \
    central_FDMatrix
from Costs import Calc_Cost_sPODG, Calc_Cost
import sys
import os
from time import perf_counter
from sklearn.utils.extmath import randomized_svd
import numpy as np
import time

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
impath = "./data/sPODG/FOTR/adaptive/refine=nth/Nm=10/"  # For data
immpath = "./plots/sPODG_1D/FOTR/adaptive/refine=nth/Nm=10/"  # For plots
os.makedirs(impath, exist_ok=True)
qs_org = wf.TimeIntegration_primal(wf.InitialConditions_primal(), f_tilde, A_p, psi, ti_method=tm)
sigma = Force_masking(qs_org, wf.X, wf.Y, wf.t, dim=Dimension)
np.save(impath + 'sigma.npy', sigma)
np.save(impath + 'qs_org.npy', qs_org)
sigma = np.load(impath + 'sigma.npy')

#%% Optimal control
max_opt_steps = 75000
verbose = False
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

#%% ROM Variables
Num_sample = 200
Nm_primal = 10
Nm_adjoint = 10


# Basis update condition
stagnate = 0
refine = True
stag_ctr = 10

D = central_FDMatrix(order=6, Nx=wf.Nxi, dx=wf.dx)

# Generate the shift samples
delta_s = subsample(wf.X, num_sample=Num_sample)

# Extract transformation operators based on sub-sampled delta
T_delta, _ = get_T(delta_s, wf.X, wf.t)

start = time.time()
# %%
for opt_step in range(max_opt_steps):

    print("\n-------------------------------")
    print("Optimization step: %d" % opt_step)

    if refine:
        time_odeint = perf_counter()  # save timing
        '''
        Forward calculation with primal at intermediate steps
        '''
        qs = wf.TimeIntegration_primal(q0, f, A_p, psi, ti_method=tm)

        # Compute the shifts from the FOM
        delta_primal, _ = Shifts_1D(qs, wf.X, wf.t)

        # Compute the reduced basis
        # _, Nm_primal, V_p, _ = srPCA_1D(qs, delta_primal, wf.X, wf.t, spod_iter=200)
        _, T = get_T(delta_primal, wf.X, wf.t)
        qs_s = T.reverse(qs)
        V_p, qs_s_POD = compute_red_basis(qs_s, Nm_primal)
        err = np.linalg.norm(qs_s - qs_s_POD) / np.linalg.norm(qs_s)
        if verbose: print(f"Relative error for shifted primal: {err}, with Nm_primal: {Nm_primal}")

        # Construct the primal system matrices for the sPOD-Galerkin approach
        Vd_p, Wd_p, lhs_p, rhs_p, c_p = wf.sPOD_Galerkin_mat_primal(T_delta, V_p, A_p, psi, samples=Num_sample)

        # Initial condition for dynamical simulation
        a_p = wf.InitialConditions_primal_sPODG(q0, delta_s, Vd_p)

        basis_refine_itr_list.append(opt_step)
        trunc_modes_primal_list.append(Nm_primal)

        time_odeint = perf_counter() - time_odeint
        if verbose: print("Forward basis refinement t_cpu = %1.3f" % time_odeint)
    '''
    Forward calculation
    '''
    time_odeint = perf_counter()  # save timing
    as_ = wf.TimeIntegration_primal_sPODG(lhs_p, rhs_p, c_p, a_p, f, delta_s, ti_method=tm)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Forward t_cpu = %1.3f" % time_odeint)

    '''
    Objective and costs for control
    '''
    # Compute the interpolation weight and the interval in which the shift lies corresponding to which we compute the
    # V_delta and W_delta matrices
    time_odeint = perf_counter()  # save timing
    intIds, weights = findIntervals(delta_s, as_[-1, :])
    J = Calc_Cost_sPODG(Vd_p, as_, qs_target, f, lamda, intIds, weights, **kwargs)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Calc_Cost t_cpu = %1.6f" % time_odeint)
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

        # # Compute the reduced basis of the uncontrolled system
        # _, Nm_adjoint, V_a, _ = srPCA_1D(qs_adj, delta_primal, wf.X, wf.t, spod_iter=200)
        qs_adj_s = T.reverse(qs_adj)
        V_a, qs_adj_s_POD = compute_red_basis(qs_adj_s, Nm_adjoint)
        err = np.linalg.norm(qs_adj_s - qs_adj_s_POD) / np.linalg.norm(qs_adj_s)
        if verbose: print(f"Relative error for shifted adjoint: {err}, with Nm_adjoint: {Nm_adjoint}")

        # Construct the primal system matrices for the sPOD-Galerkin approach
        Vd_a, Wd_a, lhs_a, rhs_a = wf.sPOD_Galerkin_mat_adjoint(T_delta, V_a, A_a, samples=Num_sample)

        # Initial condition for dynamical simulation
        a_a = wf.InitialConditions_adjoint_sPODG(Nm_adjoint, as_)

        trunc_modes_adjoint_list.append(Nm_adjoint)

        time_odeint = perf_counter() - time_odeint
        if verbose: print("Backward basis refinement t_cpu = %1.3f" % time_odeint)
    '''
    Backward calculation with reduced system
    '''
    time_odeint = perf_counter()  # save timing
    as_adj = wf.TimeIntegration_adjoint_sPODG(lhs_a, rhs_a, Vd_p, Vd_a, Wd_a, qs_target, a_a, f, as_,
                                              delta_s, ti_method=tm)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Backward t_cpu = %1.3f" % time_odeint)

    '''
     Update Control
    '''
    time_odeint = perf_counter() - time_odeint
    f, J_opt, dL_du, _, stag = Update_Control_sPODG(f, lhs_p, rhs_p, c_p, a_p, as_adj, qs_target, delta_s, Vd_p, Vd_a, psi,
                                                    J, intIds, weights, omega, lamda, max_Armijo_iter=18, wf=wf,
                                                    delta=1e-4, ti_method=tm, **kwargs)

    # Save for plotting
    J_opt_list.append(J_opt)
    dL_du_list.append(dL_du)
    dL_du_ratio_list.append(dL_du / dL_du_list[0])


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
    if stagnate > stag_ctr:
        refine = True
        stagnate = 0

        if verbose: print("\n\n-------------------------------")
        if verbose: print(f"WARNING... Armijo started to stagnate, so we refine ")
    else:
        refine = False


# Compute the final state
as__ = wf.TimeIntegration_primal_sPODG(lhs_p, rhs_p, c_p, a_p, f, delta_s, ti_method=tm)
as_online = as__[:Nm_primal]
delta_online = as__[-1]
qs = np.zeros_like(qs_target)
intIds, weights = findIntervals(delta_s, delta_online)
for i in range(f.shape[1]):
    V_delta = weights[i] * Vd_p[intIds[i]] + (1 - weights[i]) * Vd_p[intIds[i] + 1]
    qs[:, i] = V_delta @ as_online[:, i]

as_adj_online = as_adj[:Nm_adjoint]
qs_adj = np.zeros_like(qs_target)
for i in range(f.shape[1]):
    V_delta = weights[i] * Vd_a[intIds[i]] + (1 - weights[i]) * Vd_a[intIds[i] + 1]
    qs_adj[:, i] = V_delta @ as_adj_online[:, i]

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












# as_online = as_[:-1]
    # delta_online = as_[-1]
    # tmp_low = V_p @ as_online
    # tmp = np.zeros_like(qs_target)
    # intIds, weights = findIntervals(delta_s, delta_online)
    # for i in range(f.shape[1]):
    #     V_delta = weights[i] * Vd_p[intIds[i]] + (1 - weights[i]) * Vd_p[intIds[i] + 1]
    #     tmp[:, i] = V_delta @ as_online[:, i]
    # print(np.linalg.norm(tmp_low - tmp_low_FOM) / np.linalg.norm(tmp_low_FOM))
    # print(np.linalg.norm(tmp - qtilde) / np.linalg.norm(qtilde))
    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # X_1D_grid, t_grid = np.meshgrid(wf.X, wf.t)
    # X_1D_grid = X_1D_grid.T
    # t_grid = t_grid.T
    # fig = plt.figure(figsize=(15, 5))
    # ax1 = fig.add_subplot(131)
    # im1 = ax1.pcolormesh(X_1D_grid, t_grid, tmp_low, cmap='YlOrRd')
    # ax1.axis('off')
    # ax1.set_title(r"$q(x, t)$")
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig.colorbar(im1, cax=cax, orientation='vertical')
    #
    # ax2 = fig.add_subplot(132)
    # im2 = ax2.pcolormesh(X_1D_grid, t_grid, tmp, cmap='YlOrRd')
    # ax2.axis('off')
    # ax2.set_title(r"$q(x, t)$")
    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig.colorbar(im2, cax=cax, orientation='vertical')
    #
    # ax3 = fig.add_subplot(133)
    # im3 = ax3.pcolormesh(X_1D_grid, t_grid, tmp_low_FOM, cmap='YlOrRd')
    # ax3.axis('off')
    # ax3.set_title(r"$q(x, t)$")
    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig.colorbar(im3, cax=cax, orientation='vertical')
    #
    # fig.supylabel(r"time $t$")
    # fig.supxlabel(r"space $x$")
    # plt.show()
    # exit()





   # as_online = as_adj[:Nm_adjoint]
    # delta_online = as_adj[-1]
    # tmp_low = V_a @ as_online
    # tmp = np.zeros_like(qs_target)
    # for i in range(f.shape[1]):
    #     V_delta = weights[i] * Vd_a[intIds[i]] + (1 - weights[i]) * Vd_a[intIds[i] + 1]
    #     tmp[:, i] = V_delta @ as_online[:, i]
    # print(np.linalg.norm(tmp_low - tmp_low_FOM) / np.linalg.norm(tmp_low_FOM))
    # print(np.linalg.norm(tmp - qs_adj) / np.linalg.norm(qs_adj))
    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # X_1D_grid, t_grid = np.meshgrid(wf.X, wf.t)
    # X_1D_grid = X_1D_grid.T
    # t_grid = t_grid.T
    # fig = plt.figure(figsize=(15, 5))
    # ax1 = fig.add_subplot(131)
    # im1 = ax1.pcolormesh(X_1D_grid, t_grid, tmp_low, cmap='YlOrRd')
    # ax1.axis('off')
    # ax1.set_title(r"$q(x, t)$")
    # divider = make_axes_locatable(ax1)
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig.colorbar(im1, cax=cax, orientation='vertical')
    #
    # ax2 = fig.add_subplot(132)
    # im2 = ax2.pcolormesh(X_1D_grid, t_grid, tmp, cmap='YlOrRd')
    # ax2.axis('off')
    # ax2.set_title(r"$q(x, t)$")
    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig.colorbar(im2, cax=cax, orientation='vertical')
    #
    # ax3 = fig.add_subplot(133)
    # im3 = ax3.pcolormesh(X_1D_grid, t_grid, tmp_low_FOM, cmap='YlOrRd')
    # ax3.axis('off')
    # ax3.set_title(r"$q(x, t)$")
    # divider = make_axes_locatable(ax3)
    # cax = divider.append_axes('right', size='10%', pad=0.08)
    # fig.colorbar(im3, cax=cax, orientation='vertical')
    #
    # fig.supylabel(r"time $t$")
    # fig.supxlabel(r"space $x$")
    # plt.show()
    # exit()
