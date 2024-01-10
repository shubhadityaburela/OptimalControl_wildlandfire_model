import jax.lax
import jax.numpy as jnp
from jax.config import config

config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)

from Coefficient_Matrix import CoefficientMatrix
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import numpy as np
import math

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
        self.firstderivativeOrder = "5thOrder"

        self.v_x = 0.4 * jnp.ones(self.Nt)
        self.v_y = jnp.zeros(self.Nt)
        self.C = 0.4

        self.v_x_target = self.v_x
        self.v_y_target = self.v_y
        self.v_x_target = self.v_x_target.at[tilt_from:].set(0.9)

        # Sparse matrices of the first and second order
        self.Mat = None
        self.Mat_adjoint = None
        self.Mat_primal = None

        # Offline matrices that are pre-computed for PODG
        self.C00 = None
        self.V_primal = None
        self.V_adjoint = None
        self.psi = None
        self.VT_primal_psi = None
        self.VT_adjoint_psi = None
        self.A_conv = None
        self.A_conv_adj = None

    def Grid(self):
        self.X = jnp.arange(1, self.Nxi + 1) * self.Lxi / self.Nxi
        self.dx = self.X[1] - self.X[0]

        if self.Neta == 1:
            self.Y = 0
            self.dy = 0
        else:
            self.Y = jnp.arange(1, self.Neta + 1) * self.Leta / self.Neta
            self.dy = self.Y[1] - self.Y[0]

        dt = (jnp.sqrt(self.dx ** 2 + self.dy ** 2)) * self.cfl / self.C
        self.t = dt * jnp.arange(self.Nt)
        self.dt = self.t[1] - self.t[0]

        print('dt = ', dt)
        print('Final time : ', self.t[-1])

    def InitialConditions_primal(self):
        if self.Neta == 1:
            q = jnp.exp(-((self.X - self.Lxi / 8) ** 2) / 10)

        q = jnp.reshape(q, newshape=self.NN, order="F")

        return q

    def RHS_primal(self, q, f):

        Coeff_conv_x = self.v_x[0]
        Coeff_conv_y = self.v_y[0]

        DT = Coeff_conv_x * self.Mat_primal.Grad_Xi_kron + Coeff_conv_y * self.Mat_primal.Grad_Eta_kron
        qdot = - DT.dot(q) + self.psi @ f

        return qdot

    def TimeIntegration_primal(self, q, f0=None, sigma=None, psi=None, ti_method="rk4"):
        # Creating the system matrices. The class for the creation of Coefficient matrix is created separately
        # as they are of more general use for a wide variety of problems
        self.Mat_primal = CoefficientMatrix(orderDerivative=self.firstderivativeOrder, Nxi=self.Nxi,
                                            Neta=self.Neta, periodicity='Periodic', dx=self.dx, dy=self.dy)

        # Time loop
        if ti_method == "rk4":
            # Time loop
            qs = jnp.zeros((self.Nxi * self.Neta, self.Nt))
            qs = qs.at[:, 0].set(q)

            @jax.jit
            def body(n, qs_):
                # Main Runge-Kutta 4 solver step
                h = rk4(self.RHS_primal, qs_[:, n - 1], f0[:, n], self.dt)
                return qs_.at[:, n].set(h)

            return jax.lax.fori_loop(1, self.Nt, body, qs)

        elif ti_method == "bdf4":
            @jax.jit
            def body(x, u):
                return self.RHS_primal(x, u)

            return bdf4(f=body, tt=self.t, x0=q, uu=f0.T).T

        elif ti_method == "bdf4_updated":
            @jax.jit
            def body(x, u):
                return self.RHS_primal(x, u)

            return bdf4_updated(f=body, tt=self.t, x0=q, uu=f0.T).T

        elif ti_method == "implicit_midpoint":
            @jax.jit
            def body(x, u):
                return self.RHS_primal(x, u)

            return implicit_midpoint(f=body, tt=self.t, x0=q, uu=f0.T).T

    def InitialConditions_adjoint(self):
        if self.Neta == 1:
            q_adj = jnp.zeros_like(self.X)

        # Arrange the values of T_adj and S_adj in 'q_adj'
        q_adj = jnp.reshape(q_adj, newshape=self.NN, order="F")

        return q_adj

    def RHS_adjoint(self, q_adj, f, q, q_tar):

        Coeff_conv_x = self.v_x[0]
        Coeff_conv_y = self.v_y[0]

        DT = Coeff_conv_x * self.Mat_adjoint.Grad_Xi_kron + Coeff_conv_y * self.Mat_adjoint.Grad_Eta_kron
        q_adj_dot = DT.transpose().dot(q_adj) - (q - q_tar)

        return q_adj_dot

    def TimeIntegration_adjoint(self, q0_adj, f0, qs, qs_target, ti_method="rk4", dict_args=None):
        # Creating the system matrices. The class for the creation of Coefficient matrix is created separately
        # as they are of more general use for a wide variety of problems
        self.Mat_adjoint = CoefficientMatrix(orderDerivative=self.firstderivativeOrder, Nxi=self.Nxi,
                                             Neta=self.Neta, periodicity='Periodic', dx=self.dx, dy=self.dy)

        # Time loop
        if ti_method == "rk4":
            # Time loop
            qs_adj = jnp.zeros((self.Nxi * self.Neta, self.Nt))
            qs_adj = qs_adj.at[:, -1].set(q0_adj)

            @jax.jit
            def body(n, qs_adj_):
                # Main Runge-Kutta 4 solver step
                h = rk4(self.RHS_adjoint, qs_adj_[:, -n], f0[:, -(n + 1)], -self.dt,
                        qs[:, -(n + 1)], qs_target[:, -(n + 1)])
                return qs_adj_.at[:, -(n + 1)].set(h)

            return jax.lax.fori_loop(1, self.Nt, body, qs_adj)

        elif ti_method == "bdf4":
            @jax.jit
            def body(x_ad, u, *args):
                return self.RHS_adjoint(x_ad, u, *args)

            return bdf4(f=body, tt=self.t, x0=q0_adj, uu=f0.T, func_args=(qs.T, qs_target.T,), type='backward').T

        elif ti_method == "bdf4_updated":
            @jax.jit
            def body(x_ad, u, *args):
                return self.RHS_adjoint(x_ad, u, *args)

            return bdf4_updated(f=body, tt=self.t, x0=q0_adj, uu=f0.T, func_args=(qs.T, qs_target.T,),
                                type='backward').T

        elif ti_method == "implicit_midpoint":
            @jax.jit
            def body(x_ad, u, *args):
                return self.RHS_adjoint(x_ad, u, *args)

            return implicit_midpoint(f=body, tt=self.t, x0=q0_adj, uu=f0.T, func_args=(qs.T, qs_target.T,),
                                     type='backward').T


    def RHS_primal_target(self, q, f, v_x, v_y):

        DT = v_x * self.Mat_primal.Grad_Xi_kron + v_y * self.Mat_primal.Grad_Eta_kron
        qdot = - DT.dot(q)

        return qdot

    def TimeIntegration_primal_target(self, q, f0=None, sigma=None, psi=None, ti_method="rk4"):
        # Creating the system matrices. The class for the creation of Coefficient matrix is created separately
        # as they are of more general use for a wide variety of problems
        self.Mat_primal = CoefficientMatrix(orderDerivative=self.firstderivativeOrder, Nxi=self.Nxi,
                                            Neta=self.Neta, periodicity='Periodic', dx=self.dx, dy=self.dy)

        # Time loop
        qs = jnp.zeros((self.Nxi * self.Neta, self.Nt))
        qs = qs.at[:, 0].set(q)

        @jax.jit
        def body(n, qs_):
            # Main Runge-Kutta 4 solver step
            h = rk4(self.RHS_primal_target, qs_[:, n - 1], f0[:, n], self.dt, self.v_x_target[n],
                    self.v_y_target[n - 1])
            return qs_.at[:, n].set(h)

        return jax.lax.fori_loop(1, self.Nt, body, qs)

    ######################################### FOTR POD #############################################
    def InitialConditions_primal_PODG(self, q0):

        return self.V_primal.transpose() @ q0

    def RHS_primal_PODG(self, a, f):

        return self.A_conv @ a + self.VT_primal_psi @ f

    def TimeIntegration_primal_PODG(self, a, f0, ti_method="rk4"):
        # Time loop
        if ti_method == "rk4":
            # Time loop
            as_ = jnp.zeros((a.shape[0], self.Nt))
            as_ = as_.at[:, 0].set(a)

            @jax.jit
            def body(n, as__):
                # Main Runge-Kutta 4 solver step
                h = rk4(self.RHS_primal_PODG, as__[:, n - 1], f0[:, n], self.dt)
                return as__.at[:, n].set(h)

            return jax.lax.fori_loop(1, self.Nt, body, as_)

        elif ti_method == "bdf4":
            @jax.jit
            def body(x, u):
                return self.RHS_primal_PODG(x, u)

            return bdf4(f=body, tt=self.t, x0=a, uu=f0.T).T

        elif ti_method == "bdf4_updated":
            @jax.jit
            def body(x, u):
                return self.RHS_primal_PODG(x, u)

            return bdf4_updated(f=body, tt=self.t, x0=a, uu=f0.T).T

        elif ti_method == "implicit_midpoint":
            @jax.jit
            def body(x, u):
                return self.RHS_primal_PODG(x, u)

            return implicit_midpoint(f=body, tt=self.t, x0=a, uu=f0.T).T

    def POD_Galerkin_mat_primal(self):
        # Construct linear operators
        Mat = CoefficientMatrix(orderDerivative=self.firstderivativeOrder, Nxi=self.Nxi,
                                Neta=self.Neta, periodicity='Periodic', dx=self.dx, dy=self.dy)

        # Convection matrix (Needs to be changed if the velocity is time dependent)
        self.C00 = - (self.v_x[0] * Mat.Grad_Xi_kron + self.v_y[0] * Mat.Grad_Eta_kron)
        self.A_conv = (self.V_primal.transpose() @ self.C00) @ self.V_primal

        # Compute the pre factor for the control
        self.VT_primal_psi = self.V_primal.transpose() @ self.psi


    def InitialConditions_adjoint_PODG(self, q0_adj):

        return self.V_adjoint.transpose() @ q0_adj

    def RHS_adjoint_PODG(self, a_adj, f, a, q_target):

        return - (self.A_conv_adj @ a_adj + self.V_adjoint.transpose() @ (self.V_primal @ a - q_target))

    def TimeIntegration_adjoint_PODG(self, at_adj, f0, as_, qs_target, ti_method="rk4"):
        # Time loop
        if ti_method == "rk4":
            # Time loop
            as_adj = jnp.zeros((at_adj.shape[0], self.Nt))
            as_adj = as_adj.at[:, -1].set(at_adj)

            @jax.jit
            def body(n, as_adj_):
                # Main Runge-Kutta 4 solver step
                h = rk4(self.RHS_adjoint_PODG, as_adj_[:, -n], f0[:, -(n + 1)], -self.dt,
                        as_[:, -(n + 1)], qs_target[:, -(n + 1)])
                return as_adj_.at[:, -(n + 1)].set(h)

            return jax.lax.fori_loop(1, self.Nt, body, as_adj)

        elif ti_method == "bdf4":
            @jax.jit
            def body(x_ad, u, *args):
                return self.RHS_adjoint_PODG(x_ad, u, *args)

            return bdf4(f=body, tt=self.t, x0=at_adj, uu=f0.T, func_args=(as_.T, qs_target.T,), type='backward').T

        elif ti_method == "bdf4_updated":
            @jax.jit
            def body(x_ad, u, *args):
                return self.RHS_adjoint_PODG(x_ad, u, *args)

            return bdf4_updated(f=body, tt=self.t, x0=at_adj, uu=f0.T, func_args=(as_.T, qs_target.T,),
                                type='backward').T

        elif ti_method == "implicit_midpoint":
            @jax.jit
            def body(x_ad, u, *args):
                return self.RHS_adjoint_PODG(x_ad, u, *args)

            return implicit_midpoint(f=body, tt=self.t, x0=at_adj, uu=f0.T, func_args=(as_.T, qs_target.T,),
                                     type='backward').T

    def POD_Galerkin_mat_adjoint(self):
        self.A_conv_adj = (self.V_adjoint.transpose() @ self.C00.transpose()) @ self.V_adjoint
