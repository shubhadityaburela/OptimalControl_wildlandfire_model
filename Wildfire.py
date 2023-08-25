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


class Wildfire:
    def __init__(self, Nxi: int, Neta: int, timesteps: int) -> None:
        # Assertion statements for checking the sanctity of the input variables
        assert Nxi > 0, f"Please input sensible values for the X grid points"
        assert Neta > 0, f"Please input sensible values for the Y grid points"
        assert timesteps >= 0, f"Please input sensible values for time steps"

        # First we define the public variables of the class. All the variables with "__" in front are private variables
        self.X = None
        self.Y = None
        self.dx = None
        self.dy = None
        self.t = None
        self.dt = None

        self.NumConservedVar = 2

        # Private variables
        self.Lxi = 500
        self.Leta = 500
        self.Nxi = Nxi
        self.Neta = Neta
        self.NN = self.Nxi * self.Neta
        self.Nt = timesteps
        self.cfl = 1.0

        self.M = self.NumConservedVar * self.Nxi * self.Neta

        # Order of accuracy for the derivative matrices of the first and second order
        self.__firstderivativeOrder = "5thOrder"

        # Dimensional constants used in the model
        self.__k = 0.2136
        self.__gamma_s = 0.1625
        self.__v_x = jnp.zeros(self.Nt)
        self.__v_y = jnp.zeros(self.Nt)
        self.__alpha = 187.93
        self.__gamma = 4.8372e-5
        self.__mu = 558.49
        self.__T_a = 300
        self.__speedofsoundsquare = 1

        # Sparse matrices of the first and second order
        self.Mat_primal = None
        self.Mat_adjoint = None

        # Reference variables
        self.T_ref = self.__mu
        self.S_ref = 1
        self.x_ref = jnp.sqrt(self.__k * self.__mu) / jnp.sqrt(self.__alpha)
        self.t_ref = self.__mu / self.__alpha

    def Grid(self):
        self.X = jnp.arange(1, self.Nxi + 1) * self.Lxi / self.Nxi / self.x_ref
        self.dx = self.X[1] - self.X[0]

        if self.Neta == 1:
            self.Y = 0
            self.dy = 0
        else:
            self.Y = jnp.arange(1, self.Neta + 1) * self.Leta / self.Neta
            self.dy = self.Y[1] - self.Y[0]

        dt = (jnp.sqrt(self.dx ** 2 + self.dy ** 2)) * self.cfl / jnp.sqrt(self.__speedofsoundsquare)
        t = dt * jnp.arange(self.Nt)

        self.t = t / self.t_ref
        self.dt = self.t[1] - self.t[0]

    def InitialConditions_primal(self):
        if self.Neta == 1:
            T = 1200 * jnp.exp(-((self.X - self.Lxi / (2 * self.x_ref)) ** 2) / 200) / self.T_ref
            S = jnp.ones_like(T) / self.S_ref
        else:
            T = 1200 * jnp.exp(-(((self.X - self.Lxi / 2) ** 2) / 200 + ((self.Y - self.Leta / 2) ** 2) / 200))
            S = jnp.ones_like(T)

            # Arrange the values of T and S in 'q'
        T = jnp.reshape(T, newshape=self.NN, order="F")
        S = jnp.reshape(S, newshape=self.NN, order="F")
        q = jnp.array(jnp.concatenate((T, S)))

        return q

    def RHS_primal(self, q, f, sigma):

        # Forcing term
        f_T = f[:self.NN]
        f_S = f[self.NN:]

        # Masking
        m_T = sigma[:self.NN]
        m_S = sigma[self.NN:]

        T = q[:self.NN]
        S = q[self.NN:]

        # This array is a masking array that becomes 1 if the T is greater than 0 and 0 if not. It activates
        # the arrhenius term
        arrhenius_activate = jnp.where(T > 0, 1, 0)

        # This parameter is for preventing division by 0
        epsilon = 1e-8

        # Coefficients for the terms in the equation
        Coeff_diff = 1.0

        Coeff_conv_x = self.__v_x.at[0].get()
        Coeff_conv_y = self.__v_y.at[0].get()

        Coeff_react_1 = 1.0
        Coeff_react_2 = self.__gamma * self.__mu
        Coeff_react_3 = (self.__mu * self.__gamma_s / self.__alpha)

        DT = Coeff_conv_x * self.Mat_primal.Grad_Xi_kron + Coeff_conv_y * self.Mat_primal.Grad_Eta_kron
        Tdot = Coeff_diff * self.Mat_primal.Laplace.dot(T) - DT.dot(T) - Coeff_react_2 * T + \
               Coeff_react_1 * arrhenius_activate * S * jnp.exp(-1.0 / (jnp.maximum(T, epsilon))) + m_T * f_T
        Sdot = - Coeff_react_3 * arrhenius_activate * S * jnp.exp(-1.0 / (jnp.maximum(T, epsilon))) + m_S * f_S

        qdot = jnp.array(jnp.concatenate((Tdot, Sdot)))

        return qdot

    def TimeIntegration_primal(self, q0, f0=None, sigma=None, ti_method="bdf4"):
        # Creating the system matrices. The class for the creation of Coefficient matrix is created separately
        # as they are of more general use for a wide variety of problems
        self.Mat_primal = CoefficientMatrix(orderDerivative=self.__firstderivativeOrder, Nxi=self.Nxi,
                                            Neta=self.Neta, periodicity='Periodic', dx=self.dx, dy=self.dy)

        # Time loop
        if ti_method == "rk4":
            # Time loop
            qs = jnp.zeros((self.NumConservedVar * self.Nxi * self.Neta, self.Nt))
            qs = qs.at[:, 0].set(q0)

            @jax.jit
            def body(n, qs_):
                # Main Runge-Kutta 4 solver step
                h = self.rk4(self.RHS_primal, qs_[:, n - 1], self.dt, f0[:, n], sigma[:, n])
                return qs_.at[:, n].set(h)

            return jax.lax.fori_loop(1, self.Nt, body, qs)

        elif ti_method == "bdf4":

            @jax.jit
            def body(x, u, *args):
                return self.RHS_primal(x, u, *args)

            return self.bdf4(f=body, tt=self.t, x0=q0, uu=f0.T,
                             func_args=(sigma.T,)).T

    def InitialConditions_adjoint(self):
        T_adj = jnp.zeros_like(self.X) / self.T_ref
        S_adj = jnp.zeros_like(T_adj) / self.S_ref

        # Arrange the values of T_adj and S_adj in 'q_adj'
        T_adj = jnp.reshape(T_adj, newshape=self.NN, order="F")
        S_adj = jnp.reshape(S_adj, newshape=self.NN, order="F")
        q_adj = jnp.array(jnp.concatenate((T_adj, S_adj)))

        return q_adj

    def RHS_adjoint(self, q_adj, f, TS, TS_tar, T_var, S_var):

        T_adj = q_adj[:self.NN]
        S_adj = q_adj[self.NN:]

        T = TS[:self.NN]
        S = TS[self.NN:]

        T_tar = TS_tar[:self.NN]
        S_tar = TS_tar[self.NN:]

        # This array is a masking array that becomes 1 if the T is greater than 0 and 0 if not. It activates
        # the arrhenius term
        arrhenius_activate = jnp.where(T > 0, 1, 0)
        # This parameter is for preventing division by 0
        epsilon = 1e-8

        # Control parameters
        cp_1 = self.__gamma * self.__mu
        cp_2 = self.__mu * self.__gamma_s / self.__alpha

        DT = self.__v_x.at[0].get() * self.Mat_adjoint.Grad_Xi_kron + self.__v_y.at[
            0].get() * self.Mat_adjoint.Grad_Eta_kron
        T_adj_dot = - self.Mat_adjoint.Laplace.dot(T_adj) - DT.dot(T_adj) - arrhenius_activate * \
                    (S * jnp.exp(-1 / (jnp.maximum(T, epsilon))) * (1 / (jnp.maximum(T, epsilon)) ** 2) *
                     T_adj + jnp.exp(-1 / (jnp.maximum(T, epsilon))) * S_adj) + cp_1 * T_adj - T_var * (T - T_tar)
        S_adj_dot = arrhenius_activate * (cp_2 * S * jnp.exp(-1 / (jnp.maximum(T, epsilon))) *
                                          (1 / (jnp.maximum(T, epsilon)) ** 2) * T_adj +
                                          cp_2 * jnp.exp(-1 / (jnp.maximum(T, epsilon))) * S_adj) - S_var * (S - S_tar)

        q_adj_dot = jnp.array(jnp.concatenate((T_adj_dot, S_adj_dot)))

        return q_adj_dot

    def TimeIntegration_adjoint(self, qt_adj, f0, qs, qs_target, ti_method="bdf4", dict_args=None):
        # Creating the system matrices. The class for the creation of Coefficient matrix is created separately
        # as they are of more general use for a wide variety of problems
        self.Mat_adjoint = CoefficientMatrix(orderDerivative=self.__firstderivativeOrder, Nxi=self.Nxi,
                                             Neta=self.Neta, periodicity='Periodic', dx=self.dx, dy=self.dy)

        # Time loop
        if ti_method == "rk4":
            # Time loop
            qs_adj = jnp.zeros((self.NumConservedVar * self.Nxi * self.Neta, self.Nt))
            qs_adj = qs_adj.at[:, -1].set(qt_adj)

            @jax.jit
            def body(n, qs_adj_):
                # Main Runge-Kutta 4 solver step
                h = self.rk4(self.RHS_adjoint, qs_adj_[:, -n], -self.dt, f0[:, -(n + 1)],
                             qs[:, -(n + 1)], qs_target[:, -(n + 1)], dict_args['T_var'], dict_args['S_var'])
                return qs_adj_.at[:, -(n + 1)].set(h)

            return jax.lax.fori_loop(1, self.Nt, body, qs_adj)

        elif ti_method == "bdf4":
            @jax.jit
            def body(x_ad, u, *args):
                return self.RHS_adjoint(x_ad, u, *args, dict_args['T_var'], dict_args['S_var'])

            return self.bdf4(f=body, tt=self.t, x0=qt_adj, uu=f0.T,
                             func_args=(qs.T, qs_target.T,),
                             type='backward').T

    def ReDim_grid(self):
        self.X = self.X.at[:].set(self.X.at[:].get() * self.x_ref)
        self.t = self.t.at[:].set(self.t.at[:].get() * self.t_ref)

    def ReDim_qs(self, qs):
        qs = qs.at[:self.NN, :].set(qs.at[:self.NN, :].get() * self.T_ref)
        qs = qs.at[self.NN:, :].set(qs.at[self.NN:, :].get() * self.S_ref)

        return qs

    @staticmethod
    def rk4(RHS: callable,
            u0: jnp.ndarray,
            dt,
            *args) -> jnp.ndarray:

        k1 = RHS(u0, *args)
        k2 = RHS(u0 + dt / 2 * k1, *args)
        k3 = RHS(u0 + dt / 2 * k2, *args)
        k4 = RHS(u0 + dt * k3, *args)

        u1 = u0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return u1

    @staticmethod
    def bdf4(f: callable,
             tt: jnp.ndarray,
             x0: jnp.ndarray,
             uu: jnp.ndarray,
             func_args=(),
             dict_args=None,
             type='forward',
             debug=False,
             ) -> jnp.ndarray:

        if dict_args is None:
            dict_args = {}

        from Helper import newton

        """
        uses bdf4 method to solve the initial value problem

        x' = f(x,u), x(tt[0]) = x0    (if type == 'forward')

        or

        p' = -f(p,u), p(tt[-1]) = p0   (if type == 'backward')

        :param f: right hand side of ode, f = f(x,u)
        :param tt: timepoints
        :param x0: initial or final value
        :param uu: control input at timepoints
        :param type: 'forward' or 'backward'
        :param debug: if True, print debug information
        :return: solution of the problem in the form
            x[i,:] = x(tt[i])
            p[i,:] = p(tt[i])
        """

        N = len(x0)  # system dimension
        nt = len(tt)  # number of timepoints
        dt = tt[1] - tt[0]  # timestep

        def m_bdf(xj, xj1, xj2, xj3, xj4, uj, *args, **kwargs):
            return \
                    25 * xj \
                    - 48 * xj1 \
                    + 36 * xj2 \
                    - 16 * xj3 \
                    + 3 * xj4 \
                    - 12 * dt * f(xj, uj, *args, **kwargs)

        def m_bdf_rev(xj, xj1, xj2, xj3, xj4, uj, *args, **kwargs):
            return \
                    25 * xj \
                    - 48 * xj1 \
                    + 36 * xj2 \
                    - 16 * xj3 \
                    + 3 * xj4 \
                    + 12 * dt * f(xj, uj, *args, **kwargs)

        def m_mid(xj, xjm1, ujm1, *args, **kwargs):
            # return xj - xjm1 - dt * f( 1/2 *  (xjm1 + xj), ujm1 )
            return xj - xjm1 - dt / 2 * f(1 / 2 * (xjm1 + xj), ujm1, *args, **kwargs)  # for finer implicit midpoint

        def m_mid_rev(xj, xjm1, ujm1, *args, **kwargs):
            # return xj - xjm1 - dt * f( 1/2 *  (xjm1 + xj), ujm1 )
            return xj - xjm1 + dt / 2 * f(1 / 2 * (xjm1 + xj), ujm1, *args, **kwargs)  # for finer implicit midpoint

        solver_mid = newton(m_mid)

        solver_mid_rev = newton(m_mid_rev)

        solver_bdf = newton(m_bdf)

        solver_bdf_rev = newton(m_bdf_rev)

        if type == 'forward':

            x = jnp.zeros((nt, N))
            x = x.at[0, :].set(x0)
            xaux = jnp.zeros((7, N))
            xaux = xaux.at[0, :].set(x0)

            # first few steps with finer implicit midpoint
            for jsmall in range(1, 7):

                xauxm1 = xaux[jsmall - 1, :]
                if jsmall == 1 or jsmall == 2:
                    uauxm1 = uu[0, :]
                elif jsmall == 3 or jsmall == 4:
                    uauxm1 = uu[1, :]
                else:  # jsmall == 5 or jsmall == 6:
                    uauxm1 = uu[2, :]
                xaux = xaux.at[jsmall, :].set(
                    solver_mid(xauxm1, xauxm1, uauxm1, *tuple(ai[jsmall] for ai in func_args),
                               **{key: value[jsmall] for key, value in dict_args.items()}))

            # put values in x
            for j in range(1, 4):

                if j == 1:
                    x = x.at[j, :].set(xaux[2, :])
                elif j == 2:
                    x = x.at[j, :].set(xaux[4, :])
                else:  # j == 3
                    x = x.at[j, :].set(xaux[6, :])

                # xjm1 = x[j-1,:]
                # tjm1 = tt[j-1]
                # ujm1 = uu[j-1,:] # here u is assumed to be constant in [tjm1, tj]
                # # ujm1 = 1/2 * (uu[:,j-1] + uu[:,j]) # here u is approximated piecewise linearly
                #
                # x = x.at[j,:].set( solver_mid( xjm1, xjm1, ujm1) )
                #
                # # jax.debug.print('\n forward midpoint: j = {x}', x = j)
                #
                # # if j == 1:
                # #     jax.debug.print('\nat j = 1, midpoint, forward: ||residual|| = {x}', x = jnp.linalg.norm(m_mid(x[j,:],xjm1,ujm1)) )

            # after that bdf method
            def body(j, var):
                x, uu, args, dict_vals = var

                xjm4 = x[j - 4, :]
                xjm3 = x[j - 3, :]
                xjm2 = x[j - 2, :]
                xjm1 = x[j - 1, :]
                uj = uu[j, :]
                aij = tuple(ai[j] for ai in args)
                y = solver_bdf(xjm1, xjm1, xjm2, xjm3, xjm4, uj, *aij,
                               **{key: val[j] for key, val in zip(dict_args.keys(), dict_vals)})
                x = x.at[j, :].set(y)

                # jax.debug.print('\n forward bdf: j = {x}', x = j)

                # jax.debug.print('||residual|| = {x}', x = jnp.linalg.norm(m_bdf(y,xjm1,xjm2,xjm3,xjm4,uj)) )

                return x, uu, args, dict_vals

            x, _, _, _ = jax.lax.fori_loop(4, nt, body, (x, uu, func_args, tuple(dict_args.values())), )

            # jax.debug.print('\n forward solution: j = {x}', x=x)
            if jnp.isnan(x).any():
                jax.debug.print('forward solution went NAN')
                exit()

            return x

        else:  # type == 'backward'

            # print(dict_args)

            p = jnp.zeros((nt, N))
            p = p.at[-1, :].set(x0)

            # first few steps with finer implicit midpoint
            paux = jnp.zeros((7, N))
            paux = paux.at[-1, :].set(x0)

            for jsmall in range(1, 7):

                pauxp1 = paux[-jsmall, :]
                if jsmall == 1 or jsmall == 2:
                    uauxp1 = uu[-1, :]
                elif jsmall == 3 or jsmall == 4:
                    uauxp1 = uu[-2, :]
                else:  # jsmall == 5 or jsmall == 6:
                    uauxp1 = uu[-3, :]

                # jax.debug.print('jsmall = {x}', x = jsmall)
                # jax.debug.print('writing at {x}', x = -jsmall-1)

                paux = paux.at[-jsmall - 1, :].set(
                    solver_mid_rev(
                        pauxp1, pauxp1, 0,
                        *tuple(ai[-jsmall - 1] for ai in func_args),
                        **{key: value[-jsmall - 1] for key, value in dict_args.items()},
                    )
                )

            # jax.debug.print('paux = {x}', x = paux)

            # put values in p
            for j in reversed(range(nt - 4, nt - 1)):

                if j == nt - 2:
                    p = p.at[j, :].set(paux[4, :])
                elif j == nt - 3:
                    p = p.at[j, :].set(paux[2, :])
                else:  # j == nt-4:
                    p = p.at[j, :].set(paux[0, :])

                # jax.debug.print('j = {x}', x = j)

                # pjp1 = p[j+1,:]
                # tjp1 = tt[j+1]
                # ujp1 = uu[j+1,:] # here u is assumed to be constant in [tj, tjp1]
                # # ujp1 = 1/2 * (uu[:,j] + uu[:,j+1]) # here u is approximated piecewise linearly
                #
                # p = p.at[j,:].set(solver_mid( pjp1,pjp1, ujp1 ))
                #
                # jax.debug.print('\n backward midpoint: j = {x}', x = j)
                # # if j == nt-1:
                # #     jax.debug.print('\nat j = 1, midpoint, backward: ||residual|| = {x}', x = jnp.linalg.norm(m_mid(p[j,:],pjp1,ujp1)) )

            # jax.debug.print('\np = {x}\n', x = p)

            # after that bdf method

            def body(tup):
                j, p, uu, args, dict_vals = tup

                pjp4 = p[j + 4, :]
                pjp3 = p[j + 3, :]
                pjp2 = p[j + 2, :]
                pjp1 = p[j + 1, :]
                uj = uu[j + 1, :]
                aij = tuple(ai[j + 1] for ai in args)
                tj = tt[j + 1]

                y = solver_bdf_rev(pjp1, pjp1, pjp2, pjp3, pjp4, uj, *aij,
                                   **{key: val[j] for key, val in zip(dict_args.keys(), dict_vals)})
                p = p.at[j, :].set(y)

                # jax.debug.print('\n backward bdf: j = {x}', x = j)

                # jax.debug.print('j = {x}', x = j)
                # jax.debug.print('||residual|| = {x}', x = jnp.linalg.norm(m_bdf(y,pjp1,pjp2,pjp3,pjp4,uj)))

                return j - 1, p, uu, args, dict_vals

            def cond(tup):
                j = tup[0]
                return jnp.greater(j, -1)

            _, p, _, _, _ = jax.lax.while_loop(cond, body, (nt - 5, p, uu, func_args, tuple(dict_args.values())))

            # jax.debug.print('\np = {x}\n', x = p)

            return p
