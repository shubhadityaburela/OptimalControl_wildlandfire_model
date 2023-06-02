import jax.lax
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import numpy as np
import math
import jax.numpy as jnp
from jax.config import config
import matplotlib
import jax.lax
from Coefficient_Matrix import CoefficientMatrix

config.update("jax_enable_x64", True)  # double precision

matplotlib.use("TkAgg")


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
        self.Lxi = 100
        self.Leta = 100
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
        self.Mat = None

    def Grid(self):
        self.X = jnp.arange(1, self.Nxi + 1) * self.Lxi / self.Nxi
        self.dx = self.X[1] - self.X[0]

        if self.Neta == 1:
            self.Y = 0
            self.dy = 0
        else:
            self.Y = jnp.arange(1, self.Neta + 1) * self.Leta / self.Neta
            self.dy = self.Y[1] - self.Y[0]

        self.dt = (jnp.sqrt(self.dx ** 2 + self.dy ** 2)) * self.cfl / jnp.sqrt(self.__speedofsoundsquare)
        self.t = self.dt * jnp.arange(self.Nt)

    def InitialConditions(self):
        if self.Neta == 1:
            T = 1200 * jnp.exp(-((self.X - self.Lxi / 2) ** 2) / 200)
            S = jnp.ones_like(T)
        else:
            T = 1200 * jnp.exp(-(((self.X - self.Lxi / 2) ** 2) / 200 + ((self.Y - self.Leta / 2) ** 2) / 200))
            S = jnp.ones_like(T)

        # Arrange the values of T and S in 'q'
        T = jnp.reshape(T, newshape=self.NN, order="F")
        S = jnp.reshape(S, newshape=self.NN, order="F")
        q = jnp.array(jnp.concatenate((T, S)))

        return q

    def RHS(self, q, f=None, sigma=None):
        # Forcing term
        if f is not None:
            f_T = f[:self.NN]
            f_S = f[self.NN:]
        else:
            f_T = 0
            f_S = 0

        # Masking
        if sigma is not None:
            m_T = sigma[:self.NN]
            m_S = sigma[self.NN:]
        else:
            m_T = 0
            m_S = 0

        T = q[:self.NN]
        S = q[self.NN:]

        # This array is a masking array that becomes 1 if the T is greater than 0 and 0 if not. It activates
        # the arrhenius term
        arrhenius_activate = (T > 0).astype(int)
        # This parameter is for preventing division by 0
        epsilon = 0.00001

        # Coefficients for the terms in the equation
        Coeff_diff = self.__k
        Coeff_conv_x = self.__v_x.at[0].get()
        Coeff_conv_y = self.__v_y.at[0].get()
        Coeff_source = self.__alpha * self.__gamma
        Coeff_arrhenius = self.__alpha
        Coeff_massfrac = self.__gamma_s

        DT = Coeff_conv_x * self.Mat.Grad_Xi_kron + Coeff_conv_y * self.Mat.Grad_Eta_kron
        Tdot = Coeff_diff * self.Mat.Laplace.dot(T) - DT.dot(T) - Coeff_source * T + Coeff_arrhenius * \
               arrhenius_activate * S * jnp.exp(-self.__mu / (T + epsilon)) + m_T * f_T
        Sdot = - Coeff_massfrac * arrhenius_activate * S * jnp.exp(-self.__mu / (T + epsilon)) + m_S * f_S

        qdot = jnp.array(jnp.concatenate((Tdot, Sdot)))

        return qdot

    def TimeIntegration(self, q0, f0=None, sigma=None, ti_method="rk4"):
        # Creating the system matrices. The class for the creation of Coefficient matrix is created separately
        # as they are of more general use for a wide variety of problems
        self.Mat = CoefficientMatrix(orderDerivative=self.__firstderivativeOrder, Nxi=self.Nxi,
                                     Neta=self.Neta, periodicity='Periodic', dx=self.dx, dy=self.dy)

        if ti_method == "rk4":
            # Time loop
            qs = jnp.zeros((self.NumConservedVar * self.Nxi * self.Neta, self.Nt))
            qs = qs.at[:, 0].set(q0)

            def body(n, qs_):
                # Main Runge-Kutta 4 solver step
                h = self.rk4(self.RHS, qs_[:, n - 1], dt=self.dt, f=f0[:, n], sigma=sigma[:, n])
                return qs_.at[:, n].set(h)

            return jax.lax.fori_loop(1, self.Nt, body, qs)

        elif ti_method == "bdf4":
            # @jax.jit
            def body(q, f, sigma):
                return self.RHS(q, f=f, sigma=sigma)

            return self.bdf4(f=body, tt=self.t, x0=q0, uu=f0.T, func_args=(sigma.T,)).T


    @staticmethod
    def rk4(RHS, u0, dt, f=None, sigma=None):
        k1 = RHS(u0, f, sigma)
        k2 = RHS(u0 + dt / 2 * k1, f, sigma)
        k3 = RHS(u0 + dt / 2 * k2, f, sigma)
        k4 = RHS(u0 + dt * k3, f, sigma)

        u1 = u0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return u1

    @staticmethod
    def bdf4(f: callable,
             tt: jnp.ndarray,
             x0: jnp.ndarray,
             uu: jnp.ndarray,
             func_args: jnp.ndarray,
             type='forward',
             debug=False,
             ) -> jnp.ndarray:

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

        def m_bdf(xj, xj1, xj2, xj3, xj4, uj, *args):
            return \
                    25 * xj \
                    - 48 * xj1 \
                    + 36 * xj2 \
                    - 16 * xj3 \
                    + 3 * xj4 \
                    - 12 * dt * f(xj, uj, *args)

        def m_mid(xj, xjm1, ujm1, *args):
            # return xj - xjm1 - dt * f( 1/2 *  (xjm1 + xj), ujm1 )
            return xj - xjm1 - dt / 2 * f(1 / 2 * (xjm1 + xj), ujm1, *args)  # for finer implicit midpoint

        # solver_mid = jax_minimize( m_mid )
        solver_mid = newton(m_mid)
        # solver_mid = scipy_root( m_mid )

        # solver_bdf = jax_minimize( m_bdf )
        solver_bdf = newton(m_bdf)
        # solver_bdf = scipy_root( m_bdf )

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
                    solver_mid(xauxm1, xauxm1, uauxm1, *tuple(ai[jsmall] for ai in func_args)))

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
                x, uu, args = var

                xjm4 = x[j - 4, :]
                xjm3 = x[j - 3, :]
                xjm2 = x[j - 2, :]
                xjm1 = x[j - 1, :]
                uj = uu[j, :]
                aij = tuple(ai[j] for ai in args)
                y = solver_bdf(xjm1, xjm1, xjm2, xjm3, xjm4, uj, *aij)
                x = x.at[j, :].set(y)

                # jax.debug.print('\n forward bdf: j = {x}', x = j)

                # jax.debug.print('||residual|| = {x}', x = jnp.linalg.norm(m_bdf(y,xjm1,xjm2,xjm3,xjm4,uj)) )

                return x, uu, args

            x, _, _ = jax.lax.fori_loop(4, nt, body, (x, uu, func_args))

            return x

        else:  # type == 'backward'

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

                paux = paux.at[-jsmall - 1, :].set(solver_mid(pauxp1, pauxp1, uauxp1))

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
                j, p, uu = tup

                pjp4 = p[j + 4, :]
                pjp3 = p[j + 3, :]
                pjp2 = p[j + 2, :]
                pjp1 = p[j + 1, :]
                uj = uu[j + 1, :]
                tj = tt[j + 1]

                y = solver_bdf(pjp1, pjp1, pjp2, pjp3, pjp4, uj)
                p = p.at[j, :].set(y)

                # jax.debug.print('\n backward bdf: j = {x}', x = j)

                # jax.debug.print('j = {x}', x = j)
                # jax.debug.print('||residual|| = {x}', x = jnp.linalg.norm(m_bdf(y,pjp1,pjp2,pjp3,pjp4,uj)))

                return j - 1, p, uu

            def cond(tup):
                j = tup[0]
                return jnp.greater(j, -1)

            _, p, _ = jax.lax.while_loop(cond, body, (nt - 5, p, uu))

            # jax.debug.print('\np = {x}\n', x = p)

            return p


def Force_masking(qs, X, Y, t, dim=1):
    from scipy.signal import savgol_filter
    from scipy.ndimage import uniform_filter1d

    Nx = len(X)
    Nt = len(t)

    T = qs[:Nx, :]
    S = qs[Nx:, :]

    if dim == 1:
        mask = np.zeros((Nx, Nt))
        for j in reversed(range(Nt)):
            if j > Nt // 4:
                mask[:, j] = 1
            else:
                mask[:, j] = uniform_filter1d(S[:, j + Nt // 4], size=100, mode="nearest")
    elif dim == 2:
        pass
    else:
        print('Implement masking first!!!!!!!')

    return mask

