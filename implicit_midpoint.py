from Helper import *


def implicit_midpoint(
        f: callable,
        tt: jnp.ndarray,
        x0: jnp.ndarray,
        uu: jnp.ndarray,
        func_args=(),
        dict_args=None,
        type='forward',
        debug=False,
) -> jnp.ndarray:
    if dict_args is None:
        dict_args = {}
    """
    uses implicit midpoint method to solve the initial value problem

    x' = f(x,u), x(tt[0]) = x0

    :param f: right hand side of ode, f = f(x,u)
    :param tt: timepoints, assumed to be evenly spaced
    :param x0: initial or final value
    :param uu: control input at timepoints, shape = (len(tt), N)
    :param debug: if True, print debug information
    :return: solution of the problem in the form
        x[i,:] = x(tt[i])
    """

    N = len(x0)  # system dimension
    nt = len(tt)  # number of timepoints
    dt = tt[1] - tt[0]  # timestep -> assumed to be constant

    # identity matrix
    eye = jnp.eye(N)

    # jacobian of f
    Df = jacobian(f, argnums=0)

    def F_mid(xj, xjm1, uj, ujm1, *args, **kwargs):
        return \
                xj \
                - dt * f(1 / 2 * (xjm1 + xj), 1 / 2 * (ujm1 + uj), *args, **kwargs) \
                - xjm1

    def F_mid_rev(xj, xjm1, uj, ujm1, *args, **kwargs):
        return \
                xj \
                + dt * f(1 / 2 * (xjm1 + xj), 1 / 2 * (ujm1 + uj), *args, **kwargs) \
                - xjm1

    def DF_mid(xj, xjm1, uj, ujm1, *args, **kwargs):
        return eye - dt / 2 * Df(1 / 2 * (xjm1 + xj), 1 / 2 * (ujm1 + uj), *args, **kwargs)

    def DF_mid_rev(xj, xjm1, uj, ujm1, *args, **kwargs):
        return eye + dt / 2 * Df(1 / 2 * (xjm1 + xj), 1 / 2 * (ujm1 + uj), *args, **kwargs)

    solver_mid = newton(F_mid, Df=DF_mid)
    solver_mid_rev = newton(F_mid_rev, Df=DF_mid_rev)


    if type == 'forward':
        # set initial condition
        x = jnp.zeros((nt, N))
        x = x.at[0, :].set(x0)

        # loop
        def body(j, var):
            x, uu, args, dict_vals = var

            xjm1 = x[j - 1, :]
            xj = x[j, :]
            ujm1 = uu[j - 1, :]
            uj = uu[j, :]
            aij = tuple(ai[j] for ai in args)

            y = solver_mid(xjm1, xjm1, uj, ujm1, *aij,
                           **{key: val[j] for key, val in zip(dict_args.keys(), dict_vals)})
            x = x.at[j, :].set(y)

            # jax.debug.print('\n forward bdf: j = {x}', x = j)

            # jax.debug.print('log10(||residual||) = {x}', x = jnp.log10(jnp.linalg.norm(m_bdf(y,xjm1,xjm2,xjm3,xjm4,uj))) )

            return x, uu, args, dict_vals

        x, _, _, _ = jax.lax.fori_loop(1, nt, body, (x, uu, func_args, tuple(dict_args.values())), )

        return x

    else:

        p = jnp.zeros((nt, N))
        p = p.at[-1, :].set(x0)

        def body(tup):
            j, p, uu, args, dict_vals = tup

            pauxp1 = p[j + 1, :]
            aij = tuple(ai[j] for ai in args)

            y = solver_mid_rev(pauxp1, pauxp1, 0, 0, *aij, **{key: val[j] for key, val in zip(dict_args.keys(), dict_vals)})
            p = p.at[j, :].set(y)

            # jax.debug.print('\n backward midpoint: j = {x}', x = j)

            return j - 1, p, uu, args, dict_vals

        def cond(tup):
            j = tup[0]
            return jnp.greater(j, -1)

        _, p, _, _, _ = jax.lax.while_loop(cond, body, (nt - 2, p, uu, func_args, tuple(dict_args.values())))

        # jax.debug.print('\np = {x}\n', x = p)

        return p



if __name__ == '__main__':

    import matplotlib.pyplot as plt

    n = 2
    A = -jnp.eye(n)

    def rhs(x, u):
        return A @ x

    x0 = jnp.ones((n,))


    nt = 10
    T = 4
    tt = jnp.linspace(0,T,nt)

    xx = implicit_midpoint(f=rhs, tt=tt, x0=x0, uu=jnp.ones((nt,1)), type='backward')

    plt.plot(tt, xx[:,0], label=f'solution to $\dot{{x}} = Ax$ for $nt={nt}$')
    plt.legend()
    plt.show()
