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
        self.X_2D = None
        self.Y_2D = None
        self.dx = None
        self.dy = None
        self.t = None
        self.dt = None

        self.NumConservedVar = 2

        # Private variables
        self.Lxi = 1000
        self.Leta = 1000
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
        self.__v_x = np.zeros(self.Nt)
        self.__v_y = np.zeros(self.Nt)
        self.__alpha = 187.93
        self.__gamma = 4.8372e-5
        self.__mu = 558.49
        self.__T_a = 300
        self.__speedofsoundsquare = 1

        # Sparse matrices of the first and second order
        self.Mat_primal = None
        self.Mat_adjoint = None

    def Grid(self):
        self.X = np.arange(1, self.Nxi + 1) * self.Lxi / self.Nxi
        self.dx = self.X[1] - self.X[0]

        if self.Neta == 1:
            self.Y = 0
            self.dy = 0
        else:
            self.Y = np.arange(1, self.Neta + 1) * self.Leta / self.Neta
            self.dy = self.Y[1] - self.Y[0]

        self.dt = (np.sqrt(self.dx ** 2 + self.dy ** 2)) * self.cfl / math.sqrt(self.__speedofsoundsquare)
        self.t = self.dt * np.arange(self.Nt)

        self.X_2D, self.Y_2D = np.meshgrid(self.X, self.Y)
        self.X_2D = np.transpose(self.X_2D)
        self.Y_2D = np.transpose(self.Y_2D)

    def InitialConditions_primal(self):
        if self.Neta == 1:
            T = 1200 * np.exp(-((self.X_2D - self.Lxi / 2) ** 2) / 200)
            S = np.ones_like(T)
        else:
            T = 1200 * np.exp(-(((self.X_2D - self.Lxi / 2) ** 2) / 200 + ((self.Y_2D - self.Leta / 2) ** 2) / 200))
            S = np.ones_like(T)

        # Arrange the values of T and S in 'q'
        T = np.reshape(T, newshape=self.NN, order="F")
        S = np.reshape(S, newshape=self.NN, order="F")
        q = np.array([T, S]).T

        return q

    def RHS_primal(self, q, t, t_step=0, forcing=None, qs=None, qs_target=None, mask=None):
        # Forcing term
        if forcing is not None:
            f_T = forcing[:self.NN, t_step]
            f_S = forcing[self.NN:, t_step]
        else:
            f_T = 0
            f_S = 0

        # Masking
        if mask is not None:
            m = mask[:, t_step]
        else:
            m = 0

        T = q[:, 0]
        S = q[:, 1]

        # This array is a masking array that becomes 1 if the T is greater than 0 and 0 if not. It activates
        # the arrhenius term
        arrhenius_activate = (T > 0).astype(int)
        # This parameter is for preventing division by 0
        epsilon = 0.00001

        # Coefficients for the terms in the equation
        Coeff_diff = self.__k
        Coeff_conv_x = self.__v_x[t_step]
        Coeff_conv_y = self.__v_y[t_step]

        Coeff_source = self.__alpha * self.__gamma
        Coeff_arrhenius = self.__alpha
        Coeff_massfrac = self.__gamma_s

        DT = Coeff_conv_x * self.Mat_primal.Grad_Xi_kron + Coeff_conv_y * self.Mat_primal.Grad_Eta_kron
        Tdot = Coeff_diff * self.Mat_primal.Laplace.dot(T) - DT.dot(T) - Coeff_source * T + Coeff_arrhenius * \
               arrhenius_activate * S * np.exp(-self.__mu / (T + epsilon)) + m * f_T
        Sdot = - Coeff_massfrac * arrhenius_activate * S * np.exp(-self.__mu / (T + epsilon)) + m * f_S

        qdot = np.array([Tdot, Sdot]).T

        return qdot

    def TimeIntegration_primal(self, q, f0=None, mask=None):
        # Creating the system matrices. The class for the creation of Coefficient matrix is created separately
        # as they are of more general use for a wide variety of problems
        self.Mat_primal = CoefficientMatrix(orderDerivative=self.__firstderivativeOrder, Nxi=self.Nxi,
                                            Neta=self.Neta, periodicity='Periodic', dx=self.dx, dy=self.dy)

        # Time loop
        qs = []
        for n in range(self.Nt):
            # Main Runge-Kutta 4 solver step
            q = self.RK4(self.RHS_primal, q, self.dt, 0, t_step=n, forcing=f0, mask=mask)
            qs.append(np.concatenate((q[:, 0], q[:, 1]), axis=0))

            # print('Time step: ', n)

        qs = np.asarray(qs).transpose()

        return qs

    def InitialConditions_adjoint(self):
        T_adj = np.zeros_like(self.X_2D)
        S_adj = np.zeros_like(T_adj)

        # Arrange the values of T_adj and S_adj in 'q_adj'
        T_adj = np.reshape(T_adj, newshape=self.NN, order="F")
        S_adj = np.reshape(S_adj, newshape=self.NN, order="F")
        q_adj = np.array([T_adj, S_adj]).T

        return q_adj

    def RHS_adjoint(self, q_adj, t, t_step=0, forcing=None, qs=None, qs_target=None, mask=None):
        T_adj = q_adj[:, 0]
        S_adj = q_adj[:, 1]

        T = qs[:self.NN]
        S = qs[self.NN:]

        T_tar = qs_target[:self.NN]
        S_tar = qs_target[self.NN:]

        # This array is a masking array that becomes 1 if the T is greater than 0 and 0 if not. It activates
        # the arrhenius term
        arrhenius_activate = (T > 0).astype(int)
        # This parameter is for preventing division by 0
        epsilon = 0.00001

        DT = self.__v_x[t_step] * self.Mat_adjoint.Grad_Xi_kron + self.__v_y[t_step] * self.Mat_adjoint.Grad_Eta_kron
        T_adj_dot = - self.__k * self.Mat_adjoint.Laplace.dot(T_adj) - DT.dot(T_adj) - arrhenius_activate * \
                    (self.__alpha * self.__mu * S * np.exp(
                        -self.__mu / (T + epsilon)) * T_adj / (T + epsilon)**2 + self.__alpha * np.exp(
                        -self.__mu / (T + epsilon)) * S_adj) + self.__alpha * self.__gamma * T_adj - (T - T_tar)
        S_adj_dot = arrhenius_activate * (
                self.__gamma_s * self.__mu * S * np.exp(-self.__mu / (T + epsilon)) * T_adj / (T + epsilon)**2 +
                self.__gamma_s * np.exp(-self.__mu / (T + epsilon)) * S_adj) - (S - S_tar)

        q_adj_dot = np.array([T_adj_dot, S_adj_dot]).T

        return q_adj_dot

    def TimeIntegration_adjoint(self, q_adj, qs, qs_target):
        # Creating the system matrices. The class for the creation of Coefficient matrix is created separately
        # as they are of more general use for a wide variety of problems
        self.Mat_adjoint = CoefficientMatrix(orderDerivative=self.__firstderivativeOrder, Nxi=self.Nxi,
                                             Neta=self.Neta, periodicity='Periodic', dx=self.dx, dy=self.dy)

        # Time loop
        qs_adj = []
        t = self.t[-1]
        for n in reversed(range(self.Nt)):
            qs_adj.append(np.concatenate((q_adj[:, 0], q_adj[:, 1]), axis=0))
            t = t - self.dt
            # Main Runge-Kutta 4 solver step
            q_adj = self.RK4(self.RHS_adjoint, q_adj, -self.dt, t, t_step=n, qs=qs[:, n],
                             qs_target=qs_target[:, n])

            # print('Time step: ', n)

        qs_adj.reverse()
        qs_adj = np.asarray(qs_adj).transpose()

        return qs_adj

    @staticmethod
    def RK4(RHS, u0, dt, t, t_step=0, forcing=None, qs=None, qs_target=None, mask=None):
        k1 = RHS(u0, t, t_step, forcing, qs, qs_target, mask)
        k2 = RHS(u0 + dt / 2 * k1, t + dt / 2, t_step, forcing, qs, qs_target, mask)
        k3 = RHS(u0 + dt / 2 * k2, t + dt / 2, t_step, forcing, qs, qs_target, mask)
        k4 = RHS(u0 + dt * k3, t + dt, t_step, forcing, qs, qs_target, mask)

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
                mask[:, j] = S[:, j + Nt // 4]  # uniform_filter1d(S[:, j + Nt // 4], size=100, mode="nearest")
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

