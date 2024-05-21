from memory_profiler import profile

from Coefficient_Matrix import CoefficientMatrix
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import numpy as np
import math

from Helper_sPODG import subsample, get_T, make_V_W_delta, make_LHS_mat_offline_primal, make_RHS_mat_offline_primal, \
    make_control_mat_offline_primal, make_control_update_mat, findIntervalAndGiveInterpolationWeight_1D, make_Da, \
    make_LHS_mat_online_primal, make_RHS_mat_online_primal, make_control_mat_online_primal, \
    make_target_mat_online_primal, make_target_term_matrices, make_LHS_mat_offline_primal_red, \
    make_RHS_mat_offline_primal_red, make_LHS_mat_online_primal_red, make_RHS_mat_online_primal_red

from rk4 import rk4
from bdf4 import bdf4
from bdf4_updated import bdf4_updated
from implicit_midpoint import implicit_midpoint


class advection:
    def __init__(self, Nxi: int, Neta: int, timesteps: int, cfl: float, tilt_from: int) -> None:
        # Assertion statements for checking the sanctity of the input variables
        assert Nxi > 0, f"Please input sensible values for the X grid points"
        assert Neta > 0, f"Please input sensible values for the Y grid points"
        assert timesteps >= 0, f"Please input sensible values for time steps"

        # First we define the public variables of the class. All the variables with "__" in front are private variables
        self.X = None
        self.Y = None
        self.X_2D = None
        self.Y_2D = None
        self.dx = None
        self.dy = None
        self.t = None
        self.dt = None

        # Private variables
        self.Lxi = 100
        self.Leta = 1
        self.Nxi = Nxi
        self.Neta = Neta
        self.NN = self.Nxi * self.Neta
        self.Nt = timesteps
        self.cfl = cfl

        self.M = self.Nxi * self.Neta

        # Order of accuracy for the derivative matrices of the first and second order
        self.firstderivativeOrder = "6thOrder"

        self.v_x = 0.5 * np.ones(self.Nt)
        self.v_y = np.zeros(self.Nt)
        self.C = 1.0

        self.v_x_target = self.v_x
        self.v_y_target = self.v_y
        self.v_x_target[tilt_from:] = 1.0

    def Grid(self):
        self.X = np.arange(1, self.Nxi + 1) * self.Lxi / self.Nxi
        self.dx = self.X[1] - self.X[0]

        if self.Neta == 1:
            self.Y = 0
            self.dy = 0
        else:
            self.Y = np.arange(1, self.Neta + 1) * self.Leta / self.Neta
            self.dy = self.Y[1] - self.Y[0]

        dt = (np.sqrt(self.dx ** 2 + self.dy ** 2)) * self.cfl / self.C
        self.t = dt * np.arange(self.Nt)
        self.dt = self.t[1] - self.t[0]

        print('dt = ', dt)
        print('Final time : ', self.t[-1])

    def InitialConditions_primal(self):
        if self.Neta == 1:
            q = np.exp(-((self.X - self.Lxi / 12) ** 2) / 10)

        q = np.reshape(q, newshape=self.NN, order="F")

        return q

    def RHS_primal(self, q, f, A, psi):

        qdot = A.dot(q) + psi @ f

        return qdot

    def TimeIntegration_primal(self, q, f0=None, A=None, psi=None, ti_method="rk4"):
        # Creating the system matrices. The class for the creation of Coefficient matrix is created separately
        # as they are of more general use for a wide variety of problems

        # Time loop
        if ti_method == "rk4":
            # Time loop
            qs = np.zeros((self.Nxi * self.Neta, self.Nt))
            qs[:, 0] = q

            for n in range(1, self.Nt):
                qs[:, n] = rk4(self.RHS_primal, qs[:, n - 1], f0[:, n - 1], self.dt, A, psi)

        return qs

    def InitialConditions_adjoint(self):
        if self.Neta == 1:
            q_adj = np.zeros_like(self.X)

        q_adj = np.reshape(q_adj, newshape=self.NN, order="F")

        return q_adj

    def RHS_adjoint(self, q_adj, f, q, q_tar, A):

        q_adj_dot = - A.dot(q_adj) - (q - q_tar)

        return q_adj_dot

    def TimeIntegration_adjoint(self, q0_adj, f0, qs, qs_target, A, ti_method="rk4", dict_args=None):
        # Time loop
        if ti_method == "rk4":
            # Time loop
            qs_adj = np.zeros((self.Nxi * self.Neta, self.Nt))
            qs_adj[:, -1] = q0_adj

            for n in range(1, self.Nt):
                qs_adj[:, -(n + 1)] = rk4(self.RHS_adjoint, qs_adj[:, -n], f0[:, -n], -self.dt, qs[:, -n],
                                          qs_target[:, -n], A)

        return qs_adj


    def RHS_primal_target(self, q, f, Mat, v_x, v_y):

        DT = v_x * Mat.Grad_Xi_kron + v_y * Mat.Grad_Eta_kron
        qdot = - DT.dot(q)

        return qdot

    def TimeIntegration_primal_target(self, q, f0=None, sigma=None, psi=None, ti_method="rk4"):
        # Creating the system matrices. The class for the creation of Coefficient matrix is created separately
        # as they are of more general use for a wide variety of problems
        Mat = CoefficientMatrix(orderDerivative=self.firstderivativeOrder, Nxi=self.Nxi,
                                Neta=self.Neta, periodicity='Periodic', dx=self.dx, dy=self.dy)

        # Time loop
        qs = np.zeros((self.Nxi * self.Neta, self.Nt))
        qs[:, 0] = q

        for n in range(1, self.Nt):
            qs[:, n] = rk4(self.RHS_primal_target, qs[:, n - 1], f0[:, n - 1], self.dt, Mat, self.v_x_target[n - 1],
                           self.v_y_target[n - 1])

        return qs

    ######################################### FOTR POD #############################################
    def InitialConditions_primal_PODG(self, V_p, q0):

        return V_p.transpose() @ q0

    def RHS_primal_PODG(self, a, f, Ar_p, psir_p):

        return Ar_p @ a + psir_p @ f

    def TimeIntegration_primal_PODG(self, a, f0, Ar_p, psir_p, ti_method="rk4"):
        # Time loop
        if ti_method == "rk4":
            # Time loop
            as_ = np.zeros((a.shape[0], self.Nt))
            as_[:, 0] = a

            for n in range(1, self.Nt):
                as_[:, n] = rk4(self.RHS_primal_PODG, as_[:, n - 1], f0[:, n - 1], self.dt, Ar_p, psir_p)

        return as_

    def POD_Galerkin_mat_primal(self, A_p, V_p, psi):

        V_pT = V_p.transpose()

        return (V_pT @ A_p) @ V_p, V_pT @ psi

    def InitialConditions_adjoint_PODG(self, V_a, q0_adj):

        return V_a.transpose() @ q0_adj

    def RHS_adjoint_PODG(self, a_adj, f, a, Tarr_a, Ar_a, Tr_a):

        return - (Ar_a @ a_adj + (Tr_a @ a - Tarr_a))

    def TimeIntegration_adjoint_PODG(self, at_adj, f0, as_, Ar_a, Tr_a, Tarr_a, ti_method="rk4"):
        # Time loop
        if ti_method == "rk4":
            # Time loop
            as_adj = np.zeros((at_adj.shape[0], self.Nt))
            as_adj[:, -1] = at_adj

            for n in range(1, self.Nt):
                as_adj[:, -(n + 1)] = rk4(self.RHS_adjoint_PODG, as_adj[:, -n], f0[:, -n], -self.dt,
                                          as_[:, -n],
                                          Tarr_a[:, -n],
                                          Ar_a, Tr_a)

            return as_adj


    def POD_Galerkin_mat_adjoint(self, V_a, A_p, V_p, q_target, psi):
        V_aT = V_a.transpose()
        return (V_aT @ A_p.transpose()) @ V_a, V_aT @ V_p, V_aT @ q_target, psi.transpose() @ V_a

    ######################################### FOTR sPOD #############################################
    def InitialConditions_primal_sPODG(self, q0, ds, Vd):
        z = 0
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], z)
        V = weight * Vd[intervalIdx] + (1 - weight) * Vd[intervalIdx + 1]
        a = V.transpose() @ q0
        # Initialize the shifts with zero for online phase
        a = np.concatenate((a, np.asarray([z])))

        return a

    def sPOD_Galerkin_mat_primal(self, T_delta, V_p, A_p, psi, D, samples):

        # Construct V_delta and W_delta matrix
        V_delta_primal, W_delta_primal = make_V_W_delta(V_p, T_delta, D, samples)

        # Construct LHS matrix
        LHS_matrix = make_LHS_mat_offline_primal(V_delta_primal, W_delta_primal)

        # Construct RHS matrix
        RHS_matrix = make_RHS_mat_offline_primal(V_delta_primal, W_delta_primal, A_p)

        # Construct the control matrix
        C_matrix = make_control_mat_offline_primal(V_delta_primal, W_delta_primal, psi)

        return V_delta_primal, W_delta_primal, LHS_matrix, RHS_matrix, C_matrix

    def RHS_primal_sPODG(self, a, f, lhs, rhs, c, ds):

        # Compute the interpolation weight and the interval in which the shift lies
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], -a[-1])

        # Assemble the dynamic matrix D(a)
        Da = make_Da(a)

        # Prepare the LHS side of the matrix using D(a)
        M = make_LHS_mat_online_primal(lhs, Da, intervalIdx, weight)

        # Prepare the RHS side of the matrix using D(a)
        A = make_RHS_mat_online_primal(rhs, Da, intervalIdx, weight)

        # Prepare the online control matrix
        C = make_control_mat_online_primal(f, c, Da, intervalIdx, weight)

        # print(np.linalg.cond(M, p='fro'))
        if np.linalg.cond(M, p='fro') == np.inf:
            res = np.linalg.solve(M.T.dot(M) + 1e-14 * np.identity(M.shape[1]), M.T.dot(A @ a + C))
        else:
            res = np.linalg.solve(M, A @ a + C)

        return res    # np.linalg.solve(M.T.dot(M) + 1e-14 * np.identity(M.shape[1]), M.T.dot(A @ a + C))

    def TimeIntegration_primal_sPODG(self, lhs, rhs, c, a, f0, delta_s, ti_method="rk4"):
        if ti_method == "rk4":
            # Time loop
            as_ = np.zeros((a.shape[0], self.Nt))
            as_[:, 0] = a

            for n in range(1, self.Nt):
                as_[:, n] = rk4(self.RHS_primal_sPODG, as_[:, n - 1], f0[:, n - 1], self.dt, lhs, rhs, c, delta_s)

            return as_


    def InitialConditions_adjoint_sPODG(self, Nm_adjoint, a_):
        # Initialize the shift adjoint with zero for online phase
        a = np.concatenate((np.zeros(Nm_adjoint), np.asarray([a_[-1, -1]])))

        return a

    def sPOD_Galerkin_mat_adjoint(self, T_delta, V_delta_primal, V_a, A_a, D, samples):

        # Construct V_delta and W_delta matrix
        V_delta_adjoint, W_delta_adjoint = make_V_W_delta(V_a, T_delta, D, samples)

        # Construct LHS matrix
        LHS_matrix = make_LHS_mat_offline_primal(V_delta_adjoint, W_delta_adjoint)

        # Construct RHS matrix
        RHS_matrix = make_RHS_mat_offline_primal(V_delta_adjoint, W_delta_adjoint, A_a)

        # Construct the target precomputed terms
        Tar_matrix = make_target_term_matrices(V_delta_primal, V_delta_adjoint, W_delta_adjoint)


        return V_delta_adjoint, W_delta_adjoint, LHS_matrix, RHS_matrix, Tar_matrix

    def RHS_adjoint_sPODG(self, a, f, a_, lhs, rhs, T_a, Vda, Wda, qs_target, ds):

        # Compute the interpolation weight and the interval in which the shift lies
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], -a_[-1])

        # Assemble the dynamic matrix D(a)
        Da = make_Da(a)

        # Prepare the LHS side of the matrix using D(a)
        M = make_LHS_mat_online_primal(lhs, Da, intervalIdx, weight)

        # Prepare the RHS side of the matrix using D(a)
        A = make_RHS_mat_online_primal(rhs, Da, intervalIdx, weight)

        # Prepare the online control matrix
        C = make_target_mat_online_primal(T_a, Vda, Wda, qs_target, a_[:-1], Da, intervalIdx, weight)

        if np.linalg.cond(M, p='fro') == np.inf:
            res = np.linalg.solve(M.T.dot(M) + 1e-14 * np.identity(M.shape[1]), M.T.dot(-A @ a - C))
        else:
            res = np.linalg.solve(M, -A @ a - C)

        return res

    def TimeIntegration_adjoint_sPODG(self, lhs, rhs, T_a, Vda, Wda, qs_target, at_adj, f0, as_, delta_s, ti_method="rk4"):
        # Time loop
        if ti_method == "rk4":
            # Time loop
            as_adj = np.zeros((at_adj.shape[0], self.Nt))
            as_adj[:, -1] = at_adj

            for n in range(1, self.Nt):
                as_adj[:, -(n + 1)] = rk4(self.RHS_adjoint_sPODG, as_adj[:, -n], f0[:, -n],
                                          -self.dt, as_[:, -n], lhs, rhs, T_a, Vda, Wda,
                                          qs_target[:, -n], delta_s)

            return as_adj

    ######################################### FOTR sPOD (REDUCED) #############################################
    def InitialConditions_primal_sPODG_red(self, q0, ds, Vd):
        z = 0
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], z)
        V = weight * Vd[intervalIdx] + (1 - weight) * Vd[intervalIdx + 1]
        a = V.transpose() @ q0
        # Initialize the shifts with zero for online phase
        a = np.concatenate((a, np.asarray([z])))

        return a

    def sPOD_Galerkin_mat_primal_red(self, T_delta, V_p, A_p, psi, D, samples):

        # Construct V_delta and W_delta matrix
        V_delta_primal, W_delta_primal = make_V_W_delta(V_p, T_delta, D, samples)

        # Construct LHS matrix
        LHS_matrix = make_LHS_mat_offline_primal_red(V_delta_primal, W_delta_primal)

        # Construct RHS matrix
        RHS_matrix = make_RHS_mat_offline_primal_red(V_delta_primal, W_delta_primal, A_p)

        # Construct the control matrix
        C_matrix = make_control_mat_offline_primal(V_delta_primal, W_delta_primal, psi)

        return V_delta_primal, W_delta_primal, LHS_matrix, RHS_matrix, C_matrix

    def RHS_primal_sPODG_red(self, a, f, lhs, rhs, c, ds):

        # Compute the interpolation weight and the interval in which the shift lies
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], -a[-1])

        # Assemble the dynamic matrix D(a)
        Da = make_Da(a)

        # Prepare the LHS side of the matrix using D(a)
        M = make_LHS_mat_online_primal_red(lhs, Da)

        # Prepare the RHS side of the matrix using D(a)
        A = make_RHS_mat_online_primal_red(rhs, Da)

        # Prepare the online control matrix
        C = make_control_mat_online_primal(f, c, Da, intervalIdx, weight)

        # # print(np.linalg.cond(M, p='fro'))
        # if np.linalg.cond(M, p='fro') == np.inf:
        #     res = np.linalg.solve(M.T.dot(M) + 1e-14 * np.identity(M.shape[1]), M.T.dot(A @ a + C))
        # else:
        #     res = np.linalg.solve(M, A @ a + C)

        return np.linalg.solve(M, A @ a + C)

    def TimeIntegration_primal_sPODG_red(self, lhs, rhs, c, a, f0, delta_s, ti_method="rk4"):
        if ti_method == "rk4":
            # Time loop
            as_ = np.zeros((a.shape[0], self.Nt))
            as_[:, 0] = a

            for n in range(1, self.Nt):
                as_[:, n] = rk4(self.RHS_primal_sPODG_red, as_[:, n - 1], f0[:, n - 1], self.dt, lhs, rhs, c, delta_s)

            return as_

    def InitialConditions_adjoint_sPODG_red(self, Nm_adjoint, a_):
        # Initialize the shift adjoint with zero for online phase
        a = np.concatenate((np.zeros(Nm_adjoint), np.asarray([a_[-1, -1]])))

        return a

    def sPOD_Galerkin_mat_adjoint_red(self, T_delta, V_delta_primal, V_a, A_a, D, samples):

        # Construct V_delta and W_delta matrix
        V_delta_adjoint, W_delta_adjoint = make_V_W_delta(V_a, T_delta, D, samples)

        # Construct LHS matrix
        LHS_matrix = make_LHS_mat_offline_primal_red(V_delta_adjoint, W_delta_adjoint)

        # Construct RHS matrix
        RHS_matrix = make_RHS_mat_offline_primal_red(V_delta_adjoint, W_delta_adjoint, A_a)

        # Construct the target precomputed terms
        Tar_matrix = make_target_term_matrices(V_delta_primal, V_delta_adjoint, W_delta_adjoint)

        return V_delta_adjoint, W_delta_adjoint, LHS_matrix, RHS_matrix, Tar_matrix

    def RHS_adjoint_sPODG_red(self, a, f, a_, lhs, rhs, T_a, Vda, Wda, qs_target, ds):

        # Compute the interpolation weight and the interval in which the shift lies
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(ds[2], -a_[-1])

        # Assemble the dynamic matrix D(a)
        Da = make_Da(a)

        # Prepare the LHS side of the matrix using D(a)
        M = make_LHS_mat_online_primal_red(lhs, Da)

        # Prepare the RHS side of the matrix using D(a)
        A = make_RHS_mat_online_primal_red(rhs, Da)

        # Prepare the online control matrix
        C = make_target_mat_online_primal(T_a, Vda, Wda, qs_target, a_[:-1], Da, intervalIdx, weight)

        if np.linalg.cond(M, p='fro') == np.inf:
            res = np.linalg.solve(M.T.dot(M) + 1e-14 * np.identity(M.shape[1]), M.T.dot(-A @ a - C))
        else:
            res = np.linalg.solve(M, -A @ a - C)

        return res

    def TimeIntegration_adjoint_sPODG_red(self, lhs, rhs, T_a, Vda, Wda, qs_target, at_adj, f0, as_, delta_s,
                                          ti_method="rk4"):
        # Time loop
        if ti_method == "rk4":
            # Time loop
            as_adj = np.zeros((at_adj.shape[0], self.Nt))
            as_adj[:, -1] = at_adj

            for n in range(1, self.Nt):
                as_adj[:, -(n + 1)] = rk4(self.RHS_adjoint_sPODG_red, as_adj[:, -n], f0[:, -n],
                                          -self.dt, as_[:, -n], lhs, rhs, T_a, Vda, Wda,
                                          qs_target[:, -n], delta_s)

            return as_adj
