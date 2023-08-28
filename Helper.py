import numpy as np


def L2norm(q, **kwargs):
    q = q.reshape((-1))
    return np.sqrt(np.sum(np.square(q)) * kwargs.get('dx') * kwargs.get('dt'))


def Calc_Cost(q, q_target, f, lamda, **kwargs):
    NN = kwargs.get('Nx') * kwargs.get('Ny')
    T_res = q[:NN, :] - q_target[:NN, :]
    S_res = q[NN:, :] - q_target[NN:, :]
    f_T = f[:NN, :]
    f_S = f[NN:, :]

    cost_T = (lamda['T_var'] / 2) * (L2norm(T_res, **kwargs)) ** 2 + \
             (lamda['T_reg'] / 2) * (L2norm(f_T, **kwargs)) ** 2
    cost_S = (lamda['S_var'] / 2) * (L2norm(S_res, **kwargs)) ** 2 + \
             (lamda['S_reg'] / 2) * (L2norm(f_S, **kwargs)) ** 2

    return cost_T + cost_S


def Calc_Grad(lamda, sigma, f, qs_adj, **kwargs):
    NN = kwargs.get('Nx') * kwargs.get('Ny')
    T_adj = qs_adj[:NN, :]
    S_adj = qs_adj[NN:, :]
    f_T = f[:NN, :]
    f_S = f[NN:, :]
    sigma_T = lamda['T_sig'] * sigma[:NN, :]
    sigma_S = lamda['S_sig'] * sigma[NN:, :]

    dL_du = np.zeros_like(f)
    dL_du[:NN, :] = lamda['T_reg'] * f_T + np.multiply(sigma_T, T_adj)
    dL_du[NN:, :] = lamda['S_reg'] * f_S + np.multiply(sigma_S, S_adj)

    return dL_du


def Update_Control(f, omega, lamda, sigma, q0, qs_adj, qs_target, J_prev, max_Armijo_iter,
                   wf, delta, ti_method, **kwargs):

    print("Armijo iterations.........")
    count = 0
    itr = 5
    for k in range(max_Armijo_iter):
        dL_du = Calc_Grad(lamda, sigma, f, qs_adj, **kwargs)
        f_new = f - omega * dL_du

        # Solve the primal equation
        qs = wf.TimeIntegration_primal(q0, f_new, sigma, ti_method=ti_method)

        if np.isnan(qs).any() and k < max_Armijo_iter - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / 2
        elif np.isnan(qs).any() and k == max_Armijo_iter - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost(qs, qs_target, f_new, lamda, **kwargs)
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
                            print(f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, L2norm(dL_du, **kwargs)
                    else:
                        print(f"No NANs found but step size omega = {omega} too large!", f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 2


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


def Calc_target_val(qs, wf, kind='exp_decay', **kwargs):
    NN = kwargs.get('Nx') * kwargs.get('Ny')
    T = qs[:NN]
    S = qs[NN:]

    if kind == 'exp_decay':
        exp_T = np.tile(np.exp(-0.1 * wf.t), (NN, 1))
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


def Force_masking(qs, X, Y, t, dim):
    from scipy.signal import savgol_filter
    from scipy.ndimage import uniform_filter1d

    Nx = len(X)
    Ny = len(Y)
    Nt = len(t)
    NN = Nx * Ny

    S = qs[NN:, :]

    if dim == '1D':
        mask = np.zeros((Nx, Nt))
        for j in reversed(range(Nt)):
            if j > Nt // 4:
                mask[:, j] = 1
            else:
                mask[:, j] = S[:, j + Nt // 2]  # uniform_filter1d(S[:, j + Nt // 4], size=10, mode="nearest")
    elif dim == '2D':
        mask = np.zeros((NN, Nt))
        for j in reversed(range(Nt)):
            if j > Nt // 4:
                mask[:, j] = 1
            else:
                mask[:, j] = 1  # S[:, j + Nt // 2]  # uniform_filter1d(S[:, j + Nt // 4], size=10, mode="nearest")
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