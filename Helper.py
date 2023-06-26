import numpy as np


def Calc_Cost(q, q_target, f, lamda):
    return (jnp.linalg.norm(q - q_target)) ** 2 / 2 + (lamda / 2) * (jnp.linalg.norm(f)) ** 2


def Calc_Grad(lamda, sigma, f, qs_adj):
    return lamda * f + jnp.multiply(sigma, qs_adj)


def Update_Control(f, omega, lamda, sigma, q0, qs_adj, qs_target, J_prev, max_Armijo_iter=5,
                   wf=None, delta=0.5, ti_method="rk4"):

    print("Armijo iterations.........")
    for k in range(max_Armijo_iter):
        dJ_dx = Calc_Grad(lamda, sigma, f, qs_adj)
        f_new = f - omega * dJ_dx

        # Solve the primal equation
        qs = wf.TimeIntegration_primal(q0, f_new, sigma, ti_method=ti_method)
        if jnp.isnan(qs).any() and k < max_Armijo_iter - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / 2
        elif jnp.isnan(qs).any() and k == max_Armijo_iter - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost(qs, qs_target, f_new, lamda)
            dJ = J_prev - delta * omega * jnp.linalg.norm(f_new) ** 2
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, jnp.linalg.norm(dJ_dx)
            elif J >= dJ or jnp.isnan(J):
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