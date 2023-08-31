import numpy as np
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import diags, bmat


def L2norm(q, **kwargs):
    q = q.reshape((-1))
    return np.sqrt(np.sum(np.square(q)) * kwargs.get('dt'))


def Calc_Cost(as_, as_target, f, lamda, **kwargs):
    n_rom_T = kwargs.get('n_rom_T')
    N = kwargs.get('Nx')
    aT_res = as_[:n_rom_T, :] - as_target[:n_rom_T, :]
    aS_res = as_[n_rom_T:, :] - as_target[n_rom_T:, :]
    f_T = f
    f_S = np.zeros_like(f)

    cost_T = (lamda['T_var'] / 2) * (L2norm(aT_res, **kwargs)) ** 2 + \
             (lamda['T_reg'] / 2) * (L2norm(f_T, **kwargs)) ** 2
    cost_S = (lamda['S_var'] / 2) * (L2norm(aS_res, **kwargs)) ** 2 + \
             (lamda['S_reg'] / 2) * (L2norm(f_S, **kwargs)) ** 2

    return cost_T + cost_S


def Calc_Grad(lamda, psi, f, V, as_adj, **kwargs):
    N = kwargs.get('Nx')
    qs_adj = V @ as_adj
    dL_du = lamda['T_reg'] * f + psi.transpose() @ qs_adj
    return dL_du


def Update_Control(f, omega, lamda, psi, a0_primal, V, as_adj, as_target, MAT_p, J_prev, max_Armijo_iter,
                   wf, delta, ti_method, red_nl, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5
    dL_du = Calc_Grad(lamda, psi, f, V, as_adj, **kwargs)
    for k in range(max_Armijo_iter):
        f_new = f - omega * dL_du

        # Solve the primal equation
        as_ = wf.TimeIntegration_primal_ROM(a0_primal, V, MAT_p, f_new, ti_method, red_nl, **kwargs)

        if np.isnan(as_).any() and k < max_Armijo_iter - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / 2
        elif np.isnan(as_).any() and k == max_Armijo_iter - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost(as_, as_target, f_new, lamda, **kwargs)
            dJ = J_prev - delta * omega * L2norm(dL_du, **kwargs) ** 2
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, L2norm(dL_du, **kwargs)
            elif J >= dJ or np.isnan(J):
                if k == max_Armijo_iter - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, L2norm(dL_du, **kwargs)
                else:
                    if J == dJ:
                        print(f"J has started to saturate now so we reduce the omega = {omega}!",
                              f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 2
                        count = count + 1
                        if count > itr:
                            J_opt = J
                            f_opt = f_new
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, L2norm(dL_du, **kwargs)
                    else:
                        print(f"No NANs found but step size omega = {omega} too large!",
                              f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 2


def Calc_target_val(qs, *args, kind='exp_decay', **kwargs):
    N = kwargs.get('Nx')
    t = args[0]
    X = args[1]
    T = qs[:N]
    S = qs[N:]

    if kind == 'exp_decay':
        exp_T = np.tile(np.exp(-0.1 * t), (N, 1))
        T = np.multiply(T, exp_T)
        S = np.multiply(S, (T > 0).astype(int))

        # import matplotlib.pyplot as plt
        # X_grid, t_grid = np.meshgrid(X, t)
        # X_grid = X_grid.T
        # t_grid = t_grid.T
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(X_grid, t_grid, T, rstride=1, cstride=1,
        #                 cmap='viridis', edgecolor='none')
        # ax.set_title('surface')
        # plt.show()
        # exit()

        return np.concatenate((T, S), axis=0)  # Target value of the variable

    elif kind == 'zero':
        return np.concatenate((np.zeros_like(T), np.ones_like(S)), axis=0)  # Target value of the variable


from jax import jit, jacobian
from jax.scipy.optimize import minimize
from scipy.optimize import root
import jax.numpy as jnp
import jax


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
                mask[:, j] = S[:, j + Nt // 2]  # uniform_filter1d(S[:, j + Nt // 4], size=10, mode="nearest")
    elif dim == 2:
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
    psi = np.concatenate((psi_v, np.zeros_like(psi_v)), axis=0)

    return psi


def NL(T, S):
    arr_act = np.where(T > 0, 1, 0)
    epsilon = 1e-8

    return arr_act * S * np.exp(-1 / (np.maximum(T, epsilon)))


def NL_Jac(T, S):
    arr_act = np.where(T > 0, 1, 0)
    epsilon = 1e-8

    q00 = np.squeeze(np.asarray(- arr_act * S * np.exp(-1 / (np.maximum(T, epsilon))) / (np.maximum(T, epsilon)) ** 2))
    q01 = np.squeeze(np.asarray(arr_act * np.exp(-1 / (np.maximum(T, epsilon)))))

    Jac = bmat([[diags(q00), diags(q01)]])

    return Jac


def compute_red_basis(qs, **kwargs):
    U_T, S_T, VT_T = randomized_svd(qs[:kwargs.get('Nx')], n_components=kwargs.get('n_rom_T'), random_state=None)
    U_S, S_S, VT_S = randomized_svd(qs[kwargs.get('Nx'):], n_components=kwargs.get('n_rom_S'), random_state=None)
    V = jnp.block([
        [U_T, np.zeros_like(U_S)],
        [np.zeros_like(U_T), U_S]
    ])

    return V


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
