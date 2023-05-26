import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse


class CoefficientMatrix:
    def __init__(self, orderDerivative: str, Nxi: int, Neta: int, periodicity: str, dx, dy) -> None:

        # Private variables
        self.__Nxi = Nxi
        self.__Neta = Neta
        self.__GradCoef = None  # Coefficient array for the matrix creation. More details in self.stencil_selection
        self.__DivCoef = None
        self.__GradCoefUL = None
        self.__GradCoefBR = None
        self.__DivCoefUL = None
        self.__DivCoefBR = None

        # Coefficient Matrices to be used in Solver
        self.Grad_Xi = None
        self.Div_Xi = None
        self.Grad_Eta = None
        self.Div_Eta = None
        self.Grad_Xi_kron = None  # Coefficient Matrices in the Kronecker form
        self.Div_Xi_kron = None
        self.Grad_Eta_kron = None
        self.Div_Eta_kron = None
        self.Laplace = None

        ########################################################
        # Matrix generation steps
        # 1. Stencil selection
        self.__StencilSelection(orderDerivative)

        # 2. Matrix assembly
        self.Grad_Xi = self.D1_periodic(self.__GradCoef, self.__Nxi, dx)
        self.Div_Xi = self.D1_periodic(self.__DivCoef, self.__Nxi, dx)
        self.Grad_Eta = 0
        self.Div_Eta = 0

        # Create the matrices in Kronecker form
        self.Grad_Xi_kron = self.Grad_Xi
        self.Div_Xi_kron = self.Div_Xi
        self.Grad_Eta_kron = self.Grad_Eta
        self.Div_Eta_kron = self.Div_Eta

        # Laplace matrices
        P_Xi = 1
        P_Eta = 1

        P = P_Xi * P_Eta

        self.Laplace = self.Div_Xi_kron * P * self.Grad_Xi_kron
        ########################################################

    def __StencilSelection(self, orderDerivative):
        # First derivative accuracy order stencil
        if orderDerivative == '1stOrder':
            self.__GradCoef = jnp.array([0, -1, 1, 0, 0])
            self.__DivCoef = jnp.array([0, 0, -1, 1, 0])
        elif orderDerivative == '2ndOrder':
            self.__GradCoef = jnp.array([1 / 2, -2, 3 / 2, 0, 0])
            self.__DivCoef = jnp.array([0, 0, -3 / 2, 2, -1 / 2])
        elif orderDerivative == '3rdOrder':
            self.__GradCoefUL = jnp.array([[-11 / 6, 3, -3 / 2, 1 / 3],
                                          [-1 / 3, -1 / 2, 1, -1 / 6]])  # Non periodic (left)
            self.__GradCoef = jnp.array([1 / 6, -1, 1 / 2, 1 / 3, 0])  # inner
            self.__GradCoefBR = jnp.array([-1 / 3, 3 / 2, -3, 11 / 6])

            self.__DivCoefUL = jnp.array([-11 / 6, 3, -3 / 2, 1 / 3])  # Non periodic (left)
            self.__DivCoef = jnp.array([0, -1 / 3, -1 / 2, 1, -1 / 6])  # inner
            self.__DivCoefBR = jnp.array([[1 / 6, -1, 1 / 2, 1 / 3],
                                         [-1 / 3, 3 / 2, -3, 11 / 6]])  # right
        elif orderDerivative == '5thOrder':
            self.__GradCoefUL = jnp.array([[-137 / 60, 5, -5, 10 / 3, -5 / 4, 1 / 5],
                                          [-1 / 5, -13 / 12, 2, -1, 1 / 3, -1 / 20],
                                          [1 / 20, -1 / 2, -1 / 3, 1, -1 / 4, 1 / 30]])
            self.__GradCoef = jnp.array([-2, 15, -60, 20, 30, -3, 0]) / 60
            self.__GradCoefBR = jnp.array([[1 / 20, -1 / 3, 1, -2, 13 / 12, 1 / 5],
                                          [-1 / 5, 5 / 4, -10 / 3, 5, -5, 137 / 60]])

            self.__DivCoefUL = jnp.array([[-137 / 60, 5, -5, 10 / 3, -5 / 4, 1 / 5],
                                         [-1 / 5, -13 / 12, 2, -1, 1 / 3, -1 / 20]])
            self.__DivCoef = jnp.array([0, 3, -30, -20, 60, -15, 2]) / 60
            self.__DivCoefBR = jnp.array([[-1 / 30, 1 / 4, -1, 1 / 3, 1 / 2, -1 / 20],
                                         [1 / 20, -1 / 3, 1, -2, 13 / 12, 1 / 5],
                                         [-1 / 5, 5 / 4, -10 / 3, 5, -5, 137 / 60]])
        elif orderDerivative == '7thOrder':
            self.__GradCoef = jnp.array([3, -28, 126, -420, 105, 252, -42, 4, 0]) / (140 * 3)
            self.__DivCoef = jnp.array([0, -4, 42, -252, -105, 420, -126, 28, -3]) / (140 * 3)
        else:
            raise NotImplemented("Please select the derivative order from the list already implemented")

        pass

    @staticmethod
    def D1_periodic(Coeffs, N, h):
        diagonalLow = int(-(len(Coeffs) - 1) / 2)
        diagonalUp = int(-diagonalLow)

        D_1 = jnp.zeros((N, N))

        for k in range(diagonalLow, diagonalUp + 1):
            D_1 = D_1 + Coeffs[k - diagonalLow] * jnp.diag(jnp.ones(N - abs(k)), k)
            if k < 0:
                D_1 = D_1 + Coeffs[k - diagonalLow] * jnp.diag(jnp.ones(abs(k)), N + k)
            if k > 0:
                D_1 = D_1 + Coeffs[k - diagonalLow] * jnp.diag(jnp.ones(abs(k)), -N + k)

        return D_1 / h

    # In non-periodic cases we need to adjust for the boundary nodes. Therefore, we apply BlockUL and BlockBR matrices
    # for that purpose
    @staticmethod
    def D1_nonperiodic(Coeffs, N, h, BlockUL, BlockBR):
        diagonalLow = int(-(len(Coeffs) - 1) / 2)
        diagonalUp = int(-diagonalLow)

        D_1 = sparse.BCOO.fromdense(jnp.zeros((N, N)))

        for k in range(diagonalLow, diagonalUp + 1):
            D_1 = D_1 + Coeffs[k - diagonalLow] * sparse.BCOO.fromdense(jnp.diag(jnp.ones(N - abs(k)), k))

        a = BlockUL.shape[0]
        b = BlockUL.shape[1]
        D_1[0:a, 0:b] = BlockUL

        a = BlockBR.shape[0]
        b = BlockBR.shape[1]
        D_1[D_1.shape[0] - a:D_1.shape[0], D_1.shape[1] - b:D_1.shape[1]] = BlockBR

        return D_1 / h







