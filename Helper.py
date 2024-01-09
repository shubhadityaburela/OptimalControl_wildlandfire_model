import numpy as np
from sklearn.utils.extmath import randomized_svd



def tensor_mat_prod(T, M):
    prod = jnp.zeros((T.shape[1], T.shape[0]))
    for i in range(T.shape[0]):
        prod = prod.at[:, i].set(T.at[i, ...].get() @ M.at[:, i].get())

    return prod


def L2norm(q, **kwargs):
    q = q.reshape((-1))
    return jnp.sqrt(jnp.sum(jnp.square(q)) * kwargs.get('dx') * kwargs.get('dt'))


def L2norm_ROM(q, **kwargs):
    q = q.reshape((-1))
    return jnp.sqrt(jnp.sum(jnp.square(q)) * kwargs.get('dt'))


def Calc_Cost(q, q_target, f, lamda, **kwargs):
    q_res = q - q_target
    cost = 1 / 2 * (L2norm(q_res, **kwargs)) ** 2 + (lamda['q_reg'] / 2) * (L2norm(f, **kwargs)) ** 2

    return cost


def Calc_Cost_PODG(V, as_, qs_target, f, lamda, **kwargs):
    a_res = V @ as_ - qs_target

    cost = 1 / 2 * (L2norm_ROM(a_res, **kwargs)) ** 2 + (lamda['q_reg'] / 2) * (L2norm_ROM(f, **kwargs)) ** 2

    return cost


def Calc_Grad(lamda, mask, f, qs_adj, **kwargs):

    dL_du = lamda['q_reg'] * f + mask.transpose() @ qs_adj

    return dL_du


def Calc_Grad_PODG(lamda, mask, f, V, as_adj, **kwargs):
    qs_adj = V @ as_adj
    dL_du = lamda['q_reg'] * f + mask @ qs_adj
    return dL_du



def Update_Control(f, omega, lamda, q0, qs_adj, qs_target, J_prev, max_Armijo_iter,
                   wf, delta, ti_method, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5
    mask = wf.psi

    for k in range(max_Armijo_iter):
        dL_du = Calc_Grad(lamda, mask, f, qs_adj, **kwargs)
        f_new = f - omega * dL_du

        # Solve the primal equation
        qs = wf.TimeIntegration_primal(q0, f_new, ti_method=ti_method)

        if jnp.isnan(qs).any() and k < max_Armijo_iter - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / 2
        elif jnp.isnan(qs).any() and k == max_Armijo_iter - 1:
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
            elif J >= dJ or jnp.isnan(J):
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



def Update_Control_ROM(f, omega, lamda, a0_primal, as_adj, qs_target, J_prev, max_Armijo_iter,
                   wf, delta, ti_method, red_nl, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5
    mask = wf.psi.transpose()

    dL_du = Calc_Grad_PODG(lamda, mask, f, wf.V, as_adj, **kwargs)
    for k in range(max_Armijo_iter):
        f_new = f - omega * dL_du

        # Solve the primal equation
        as_ = wf.TimeIntegration_primal_PODG(a0_primal, f_new, ti_method)

        if jnp.isnan(as_).any() and k < max_Armijo_iter - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / 2
        elif jnp.isnan(as_).any() and k == max_Armijo_iter - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost_PODG(wf.V, as_, qs_target, f_new, lamda, **kwargs)
            dJ = J_prev - delta * omega * L2norm_ROM(dL_du, **kwargs) ** 2
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, L2norm_ROM(dL_du, **kwargs)
            elif J >= dJ or jnp.isnan(J):
                if k == max_Armijo_iter - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, L2norm_ROM(dL_du, **kwargs)
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
                            return f_opt, J_opt, L2norm_ROM(dL_du, **kwargs)
                    else:
                        print(f"No NANs found but step size omega = {omega} too large!",
                              f"Reducing omega at iter={k + 1}, with J={J}")
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


def compute_red_basis(qs, **kwargs):
    U, S, VT = randomized_svd(qs, n_components=kwargs.get('n_rom'), random_state=None)

    return U


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
