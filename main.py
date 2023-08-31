from Wildfire import Wildfire
from Plots import PlotFlow
from Helper import Calc_Cost, Update_Control, Force_masking, Calc_target_val, compute_red_basis, ControlSelectionMatrix
import numpy as np
import sys
import os
from time import perf_counter
import jax
from jax.config import config
import jax.numpy as jnp
import time

config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)

np.set_printoptions(threshold=sys.maxsize, linewidth=300)

import matplotlib.pyplot as plt

# Problem variables
Dimension = "1D"
Nxi = 500
Neta = 1
Nt = 500

# Wildfire solver initialization along with grid initialization
wf = Wildfire(Nxi=Nxi, Neta=Neta if Dimension == "1D" else Nxi, timesteps=Nt)
wf.Grid()
tm = "rk4"  # Time stepping method
kwargs = {
    'dx': wf.dx,
    'dt': wf.dt,
    'Nx': wf.Nxi,
    'Nt': wf.Nt,
    'Nc': wf.NumConservedVar
}

# Solve for sigma and the target values
impath = "./data/"
os.makedirs(impath, exist_ok=True)
f_ = np.zeros((wf.NumConservedVar * wf.Nxi * wf.Neta, wf.Nt))
s_ = np.zeros_like(f_)
qs = wf.TimeIntegration_primal(wf.InitialConditions_primal(), f_, s_, ti_method="rk4")
np.save(impath + 'qs_org.npy', qs)

# Optimal control
max_opt_steps = 100
verbose = True
lamda = {'T_var': 1, 'S_var': 0, 'T_reg': 1e1, 'S_reg': 0, 'T_sig': 1,
         'S_sig': 0}  # weights and regularization parameter
omega = 1  # initial step size for gradient update
dL_du_min = 1e-6  # Convergence criteria
qs_target = Calc_target_val(qs, wf.t, wf.X, kind='zero', **kwargs)  # Target value for the optimization step
J_list = []  # Collecting cost functional over the optimization steps
dL_du_list = []  # Collecting the gradient over the optimization steps

# Initial conditions for both primal and adjoint FOM are defined here as they only need to be defined once.
q0 = wf.InitialConditions_primal()
q0_adj = wf.InitialConditions_adjoint()

# %% ROM variables

# Modes for the ROM
n_rom_T = 50
n_rom_S = 50
n_deim = 50
kwargs['n_rom_T'] = n_rom_T
kwargs['n_rom_S'] = n_rom_S
kwargs['n_deim'] = n_deim

# Initial guess for the control
n_c = 10  # Number of controls
f_tilde = np.zeros((n_c, wf.Nt))

# Selection matrix for the control input
psi = ControlSelectionMatrix(wf, n_c)

# Compute the reduced basis of the uncontrolled system
V = compute_red_basis(qs, **kwargs)

# Initial condition for dynamical simulation
a_primal = wf.InitialConditions_primal_ROM(V, q0)
a_adjoint = wf.InitialConditions_adjoint_ROM(V, q0_adj)

# Target value for the optimization
as_target = V.transpose() @ qs_target

# Construct the primal system matrices for the POD-DEIM approach
MAT_p = wf.DEIM_Mat_primal(V, qs, psi, **kwargs)

# Construct the adjoint system matrices for the POD-DEIM approach
MAT_a = wf.DEIM_Mat_adjoint(MAT_p, V, lamda, **kwargs)

# %%
for opt_step in range(max_opt_steps):

    if verbose: print("\n-------------------------------")
    if verbose: print("Optimization step: %d" % opt_step)

    '''
    Forward calculation
    '''
    time_odeint = perf_counter()  # save timing
    as_ = wf.TimeIntegration_primal_ROM(a_primal, V, MAT_p, f_tilde, ti_method=tm, red_nl=True, **kwargs)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Forward t_cpu = %1.3f" % time_odeint)

    '''
    Objective and costs for control
    '''
    J = Calc_Cost(as_, as_target, f_tilde, lamda, **kwargs)
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
    as_adj = wf.TimeIntegration_adjoint_ROM(a_adjoint, as_, as_target, V, MAT_a, ti_method=tm,
                                            red_nl=True, dict_args=lamda, **kwargs)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Backward t_cpu = %1.3f" % time_odeint)

    '''
     Update Control
    '''
    time_odeint = perf_counter() - time_odeint
    f_tilde, J_opt, dL_du = Update_Control(f_tilde, omega, lamda, psi, a_primal, V, as_adj, as_target, MAT_p, J,
                                           max_Armijo_iter=100, wf=wf, delta=1e-4, ti_method=tm, red_nl=True,
                                           **kwargs)
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
as__ = wf.TimeIntegration_primal_ROM(a_primal, V, MAT_p, f_tilde, ti_method=tm, red_nl=True, **kwargs)
qs = V @ as__
qs_adj = V @ as_adj
f_opt = psi @ f_tilde

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

# Re-dimensionalize the grid and the solution
wf.ReDim_grid()
qs_org = wf.ReDim_qs(qs_org)
qs_opt = wf.ReDim_qs(qs_opt)

# Plot the results
pf = PlotFlow(wf.X, wf.Y, wf.t)
if Dimension == "1D":
    pf.plot1D(qs_org, name="qs_org", immpath="./plots/ROM_1D/")
    pf.plot1D(qs_opt, name="qs_opt", immpath="./plots/ROM_1D/")
    pf.plot1D(qs_adj_opt, name="qs_adj_opt", immpath="./plots/ROM_1D/")
    pf.plot1D(f_opt, name="f_opt", immpath="./plots/ROM_1D/")
else:
    pf.plot2D(qs)
