import jax.lax
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import jax.numpy as jnp
import numpy as np
import math

from Coefficient_Matrix import CoefficientMatrix
import matplotlib

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
        self.cfl = 0.7

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

    def RHS(self, q, t, t_step=0, forcing=None, qs=None, qs_target=None, sigma=None):
        # Forcing term
        if forcing is not None:
            f_T = forcing[:self.NN, t_step]
            f_S = forcing[self.NN:, t_step]
        else:
            f_T = 0
            f_S = 0

        # Masking
        if sigma is not None:
            m_T = sigma[:self.NN, t_step]
            m_S = sigma[self.NN:, t_step]
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

        Coeff_source = self.__alpha * self.__gamma
        Coeff_arrhenius = self.__alpha
        Coeff_massfrac = self.__gamma_s

        Tdot = Coeff_diff * self.Mat.Laplace.dot(T) - Coeff_source * T + Coeff_arrhenius * \
               arrhenius_activate * S * jnp.exp(-self.__mu / (T + epsilon)) + m_T * f_T
        Sdot = - Coeff_massfrac * arrhenius_activate * S * jnp.exp(-self.__mu / (T + epsilon)) + m_S * f_S

        qdot = jnp.array(jnp.concatenate((Tdot, Sdot)))

        return qdot

    def TimeIntegration(self, q, f0=None, sigma=None):
        # Creating the system matrices. The class for the creation of Coefficient matrix is created separately
        # as they are of more general use for a wide variety of problems
        self.Mat = CoefficientMatrix(orderDerivative=self.__firstderivativeOrder, Nxi=self.Nxi,
                                     Neta=self.Neta, periodicity='Periodic', dx=self.dx, dy=self.dy)

        # Time loop
        qs = jnp.zeros((self.NumConservedVar * self.Nxi, self.Nt))
        qs = qs.at[:, 0].set(q)

        def body(n, qs):
            # Main Runge-Kutta 4 solver step
            h = self.RK4(self.RHS, qs[:, n - 1], self.dt, 0, t_step=n, forcing=f0, sigma=sigma)
            return qs.at[n].set(
                jnp.concatenate((h[:self.NN], h[self.NN:]), axis=0),
            )

        return jax.lax.fori_loop(1, self.Nt, body, qs)

    @staticmethod
    def RK4(RHS, u0, dt, t, t_step=0, forcing=None, qs=None, qs_target=None, sigma=None):
        k1 = RHS(u0, t, t_step, forcing, qs, qs_target, sigma)
        k2 = RHS(u0 + dt / 2 * k1, t + dt / 2, t_step, forcing, qs, qs_target, sigma)
        k3 = RHS(u0 + dt / 2 * k2, t + dt / 2, t_step, forcing, qs, qs_target, sigma)
        k4 = RHS(u0 + dt * k3, t + dt, t_step, forcing, qs, qs_target, sigma)

        u1 = u0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return u1


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


def Adjoint_Matrices():
    import sympy as sy
    from sympy.physics.units.quantities import Quantity

    k = Quantity('k')
    al = Quantity('alpha')
    g = Quantity('gamma')
    gs = Quantity('gamma_s')
    m = Quantity('mu')

    T, S = sy.symbols('T S')

    a = sy.Matrix([k * T, 0])
    b = sy.Matrix([T, 0])
    c_0 = sy.Matrix([(al * S * sy.exp(-m / T) - al * g * T), -S * sy.exp(-m / T) * gs])
    c_1 = sy.Matrix([- al * g * T, 0])

    da_dq = a.jacobian([T, S])
    db_dq = b.jacobian([T, S])
    dc_dq = [c_0.jacobian([T, S]), c_1.jacobian([T, S])]

    return da_dq, db_dq, dc_dq

