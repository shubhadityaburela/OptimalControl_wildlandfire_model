import numpy as np
from sklearn.utils.extmath import randomized_svd
from scipy import interpolate
from jax import jit, jacobian
from jax.scipy.optimize import minimize
from scipy.optimize import root
from scipy import integrate
import jax.numpy as jnp
import jax

import sys
import os


def tensor_mat_prod(T, M):
    prod = jnp.zeros((T.shape[1], T.shape[0]))
    for i in range(T.shape[0]):
        prod = prod.at[:, i].set(T.at[i, ...].get() @ M.at[:, i].get())

    return prod


def trapezoidal_integration(q, **kwargs):
    return integrate.trapezoid(integrate.trapezoid(np.square(q), axis=0, dx=kwargs['dx']), axis=0, dx=kwargs['dt'])


def integrate_cost(q, **kwargs):
    q[:, 0] = q[:, 0] / 2
    q[:, -1] = q[:, -1] / 2
    q = q.reshape((-1))
    return np.sum(np.square(q)) * kwargs.get('dx') * kwargs.get('dt')


def L2norm_ROM(q, **kwargs):
    q[:, 0] = q[:, 0] / 2
    q[:, -1] = q[:, -1] / 2
    q = q.reshape((-1))
    return np.sum(np.square(q)) * kwargs.get('dt')


# BDF4 helper functions
def newton(f, Df=None, maxIter=10, tol=1e-14):
    if Df is None:
        Df = jacobian(f, argnums=0)

    @jit
    def solver(x0, *args, **kwargs):
        def body(tup):
            i, x = tup
            update = jnp.linalg.solve(Df(x, *args, **kwargs), f(x, *args, **kwargs))
            return i + 1, x - update

        def cond(tup):
            i, x = tup

            # return jnp.less( i, maxIter )  # only check for maxIter

            return jnp.logical_and(  # check maxIter and tol
                jnp.less(i, maxIter),  # i < maxIter
                jnp.greater(jnp.linalg.norm(f(x, *args, **kwargs)), tol)  # norm( f(x) ) > tol
            )

        i, x = jax.lax.while_loop(cond, body, (0, x0))

        # jax.debug.print( '||f(x)|| = {x}', x = jnp.linalg.norm(f(x, * args, ** kwargs )))
        # jax.debug.print( 'iter = {x}', x = i)
        return x

    return solver


def jax_minimize(f):
    @jit
    def solver(x0, *args):
        g = lambda x: jnp.linalg.norm(
            f(x, *args)
        ) ** 2

        return minimize(g, x0, method='BFGS').x

    return solver


def scipy_root(f, Df=None):
    if Df is None:
        Df = jit(jacobian(f, argnums=0))

    # @jit
    def solver(x0, *args):
        return root(f, x0, jac=Df, args=args).x

    return solver


# Other Helper functions
def Calc_target_val(qs, wf, kind='exp_decay', **kwargs):
    NN = kwargs.get('Nx') * kwargs.get('Ny')

    return np.zeros_like(qs)  # Target value of the variable


def Force_masking(qs, X, Y, t, dim):
    from scipy.signal import savgol_filter
    from scipy.ndimage import uniform_filter1d

    Nx = len(X)
    Nt = len(t)

    if dim == '1D':
        mask = np.zeros((Nx, Nt))
        for j in reversed(range(Nt)):
            if j > 3 * Nt // 4:
                mask[:, j] = 1
            else:
                mask[:, j] = 0  # uniform_filter1d(S[:, j + Nt // 4], size=10, mode="nearest")
    elif dim == '2D':
        pass
    else:
        print('Implement masking first!!!!!!!')

    return mask


def ControlSelectionMatrix(wf, n_c):
    psi_v = np.zeros((wf.Nxi, n_c))

    ignition_zone = wf.Nxi // 5
    non_ignition_zone = wf.Nxi - ignition_zone

    left_ignition_index = np.split(np.arange(0, non_ignition_zone // 2), n_c // 2)
    ignition_zone_index = np.arange(non_ignition_zone // 2, non_ignition_zone // 2 + ignition_zone)
    right_ignition_index = np.split(np.arange(non_ignition_zone // 2 + ignition_zone, wf.Nxi), n_c // 2)

    for i in range(len(left_ignition_index)):
        left_ignition_index[i][:] = i
    for i in range(len(right_ignition_index)):
        right_ignition_index[i][:] = i + n_c // 2
    ignition_index = np.ones(ignition_zone) * left_ignition_index[-1][0]

    fill_column = np.concatenate((np.concatenate(left_ignition_index, axis=0),
                                  ignition_index,
                                  (np.concatenate(right_ignition_index, axis=0))))

    replace_values = np.ones_like(fill_column)
    replace_values[ignition_zone_index] = 0

    np.put_along_axis(psi_v, fill_column[:, None].astype(int), replace_values[:, None], axis=1)
    psi = psi_v

    return psi


def ControlSelectionMatrix_advection(wf, n_c, shut_off_the_first_ncontrols=2, tilt_from=None):
    psi = np.zeros((wf.Nxi, n_c))
    psi_tensor = np.zeros((wf.Nt, wf.Nxi, n_c))
    psiT_tensor = np.zeros((wf.Nt, n_c, wf.Nxi))

    control_index = np.split(np.arange(0, wf.Nxi), n_c)
    for i in range(len(control_index)):
        control_index[i][:] = i

    fill_column = np.concatenate(control_index, axis=0)
    replace_values = np.ones_like(fill_column)
    np.put_along_axis(psi, fill_column[:, None].astype(int), replace_values[:, None], axis=1)

    for i in range(shut_off_the_first_ncontrols):
        psi[:, i] = 0

    for i in range(wf.Nt):
        if i < tilt_from:
            psi_tensor[i, ...] = np.zeros_like(psi)
            psiT_tensor[i, ...] = np.zeros_like(psi.transpose())
        else:
            psi_tensor[i, ...] = psi
            psiT_tensor[i, ...] = psi.transpose()

    return psi, psi_tensor, psiT_tensor


def compute_red_basis(qs, nm):
    U, S, VT = randomized_svd(qs, n_components=nm, random_state=None)

    return U, U.dot(np.diag(S).dot(VT))


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



