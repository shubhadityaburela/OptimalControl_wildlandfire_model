import numpy as np
import jax.numpy as jnp

import jax


class Cost_functional:
    def __init__(self, qs_target, sigma, qs_0, lamda, wf, ti_method="rk4"):

        self.qs_target = qs_target.reshape((-1))
        self.sigma = sigma
        self.qs_0 = qs_0
        self.lamda = lamda
        self.wf = wf
        self.ti_method = ti_method

    def C_JAX(self, f):

        qs = self.wf.TimeIntegration(self.qs_0, f, self.sigma, self.ti_method)

        qs = qs.reshape((-1))
        f = f.reshape((-1))

        return (qs - self.qs_target).T @ (qs - self.qs_target) / 2 + (self.lamda / 2) * f.T @ f

    def C_JAX_cost(self, qs, f):
        qs = qs.reshape((-1))
        f = f.reshape((-1))
        return (qs - self.qs_target).T @ (qs - self.qs_target) / 2 + (self.lamda / 2) * f.T @ f

    def C_JAX_update(self, f, omega, dJ_dx, J_prev, max_Armijo_iter=5, delta=0.5):

        print("Armijo iterations.........")
        for k in range(max_Armijo_iter):
            f_new = f - omega * dJ_dx

            # Solve the primal equation
            qs = self.wf.TimeIntegration(self.qs_0, f_new, self.sigma, self.ti_method)
            if jnp.isnan(qs).any() and k < max_Armijo_iter - 1:
                print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
                omega = omega / 4
            elif jnp.isnan(qs).any() and k == max_Armijo_iter - 1:
                print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
                exit()
            else:
                J = self.C_JAX_cost(qs, f_new)
                if J < J_prev - delta * omega * jnp.linalg.norm(f_new):
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration converged after {k + 1} steps")
                    return f_opt, J_opt
                elif J >= J_prev or np.isnan(J):
                    omega = omega / 4




from jax import jit, jacobian
from jax.scipy.optimize import minimize
from scipy.optimize import root


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