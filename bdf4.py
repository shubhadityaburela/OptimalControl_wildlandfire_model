from Helper import *


def bdf4(f: callable,
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
    uses bdf4 method to solve the initial value problem

    x' = f(x,u), x(tt[0]) = x0    (if type == 'forward')

    or

    p' = -f(p,u), p(tt[-1]) = p0   (if type == 'backward')

    :param f: right hand side of ode, f = f(x,u)
    :param tt: timepoints
    :param x0: initial or final value
    :param uu: control input at timepoints
    :param type: 'forward' or 'backward'
    :param debug: if True, print debug information
    :return: solution of the problem in the form
        x[i,:] = x(tt[i])
        p[i,:] = p(tt[i])
    """

    N = len(x0)  # system dimension
    nt = len(tt)  # number of timepoints
    dt = tt[1] - tt[0]  # timestep

    def m_bdf(xj, xj1, xj2, xj3, xj4, uj, *args, **kwargs):
        return \
                25 * xj \
                - 48 * xj1 \
                + 36 * xj2 \
                - 16 * xj3 \
                + 3 * xj4 \
                - 12 * dt * f(xj, uj, *args, **kwargs)

    def m_bdf_rev(xj, xj1, xj2, xj3, xj4, uj, *args, **kwargs):
        return \
                25 * xj \
                - 48 * xj1 \
                + 36 * xj2 \
                - 16 * xj3 \
                + 3 * xj4 \
                + 12 * dt * f(xj, uj, *args, **kwargs)

    def m_mid(xj, xjm1, ujm1, *args, **kwargs):
        return xj - xjm1 - dt / 2 * f(1 / 2 * (xjm1 + xj), ujm1, *args, **kwargs)  # for finer implicit midpoint

    def m_mid_rev(xj, xjm1, ujm1, *args, **kwargs):
        return xj - xjm1 + dt / 2 * f(1 / 2 * (xjm1 + xj), ujm1, *args, **kwargs)  # for finer implicit midpoint

    solver_mid = newton(m_mid)

    solver_mid_rev = newton(m_mid_rev)

    solver_bdf = newton(m_bdf)

    solver_bdf_rev = newton(m_bdf_rev)

    if type == 'forward':

        x = jnp.zeros((nt, N))
        x = x.at[0, :].set(x0)
        xaux = jnp.zeros((7, N))
        xaux = xaux.at[0, :].set(x0)

        # first few steps with finer implicit midpoint
        for jsmall in range(1, 7):

            xauxm1 = xaux[jsmall - 1, :]
            if jsmall == 1 or jsmall == 2:
                uauxm1 = uu[0, :]
            elif jsmall == 3 or jsmall == 4:
                uauxm1 = uu[1, :]
            else:  # jsmall == 5 or jsmall == 6:
                uauxm1 = uu[2, :]
            xaux = xaux.at[jsmall, :].set(
                solver_mid(xauxm1, xauxm1, uauxm1, *tuple(ai[jsmall] for ai in func_args),
                           **{key: value[jsmall] for key, value in dict_args.items()}))

        # put values in x
        for j in range(1, 4):

            if j == 1:
                x = x.at[j, :].set(xaux[2, :])
            elif j == 2:
                x = x.at[j, :].set(xaux[4, :])
            else:  # j == 3
                x = x.at[j, :].set(xaux[6, :])

            # xjm1 = x[j-1,:]
            # tjm1 = tt[j-1]
            # ujm1 = uu[j-1,:] # here u is assumed to be constant in [tjm1, tj]
            # # ujm1 = 1/2 * (uu[:,j-1] + uu[:,j]) # here u is approximated piecewise linearly
            #
            # x = x.at[j,:].set( solver_mid( xjm1, xjm1, ujm1) )
            #
            # # jax.debug.print('\n forward midpoint: j = {x}', x = j)
            #
            # # if j == 1:
            # #     jax.debug.print('\nat j = 1, midpoint, forward: ||residual|| = {x}', x = jnp.linalg.norm(m_mid(x[j,:],xjm1,ujm1)) )

        # after that bdf method
        def body(j, var):
            x, uu, args, dict_vals = var

            xjm4 = x[j - 4, :]
            xjm3 = x[j - 3, :]
            xjm2 = x[j - 2, :]
            xjm1 = x[j - 1, :]
            uj = uu[j, :]
            aij = tuple(ai[j] for ai in args)
            y = solver_bdf(xjm1, xjm1, xjm2, xjm3, xjm4, uj, *aij,
                           **{key: val[j] for key, val in zip(dict_args.keys(), dict_vals)})
            x = x.at[j, :].set(y)

            # jax.debug.print('\n forward bdf: j = {x}', x = j)

            # jax.debug.print('||residual|| = {x}', x = jnp.linalg.norm(m_bdf(y,xjm1,xjm2,xjm3,xjm4,uj)) )

            return x, uu, args, dict_vals

        x, _, _, _ = jax.lax.fori_loop(4, nt, body, (x, uu, func_args, tuple(dict_args.values())), )

        # jax.debug.print('\n forward solution: j = {x}', x=x)
        if jnp.isnan(x).any():
            jax.debug.print('forward solution went NAN')
            exit()

        return x

    else:  # type == 'backward'

        # print(dict_args)

        p = jnp.zeros((nt, N))
        p = p.at[-1, :].set(x0)

        # first few steps with finer implicit midpoint
        paux = jnp.zeros((7, N))
        paux = paux.at[-1, :].set(x0)

        for jsmall in range(1, 7):

            pauxp1 = paux[-jsmall, :]
            if jsmall == 1 or jsmall == 2:
                uauxp1 = uu[-1, :]
            elif jsmall == 3 or jsmall == 4:
                uauxp1 = uu[-2, :]
            else:  # jsmall == 5 or jsmall == 6:
                uauxp1 = uu[-3, :]

            # jax.debug.print('jsmall = {x}', x = jsmall)
            # jax.debug.print('writing at {x}', x = -jsmall-1)

            paux = paux.at[-jsmall - 1, :].set(
                solver_mid_rev(
                    pauxp1, pauxp1, 0,
                    *tuple(ai[-jsmall - 1] for ai in func_args),
                    **{key: value[-jsmall - 1] for key, value in dict_args.items()},
                )
            )

        # jax.debug.print('paux = {x}', x = paux)

        # put values in p
        for j in reversed(range(nt - 4, nt - 1)):

            if j == nt - 2:
                p = p.at[j, :].set(paux[4, :])
            elif j == nt - 3:
                p = p.at[j, :].set(paux[2, :])
            else:  # j == nt-4:
                p = p.at[j, :].set(paux[0, :])

            # jax.debug.print('j = {x}', x = j)

            # pjp1 = p[j+1,:]
            # tjp1 = tt[j+1]
            # ujp1 = uu[j+1,:] # here u is assumed to be constant in [tj, tjp1]
            # # ujp1 = 1/2 * (uu[:,j] + uu[:,j+1]) # here u is approximated piecewise linearly
            #
            # p = p.at[j,:].set(solver_mid( pjp1,pjp1, ujp1 ))
            #
            # jax.debug.print('\n backward midpoint: j = {x}', x = j)
            # # if j == nt-1:
            # #     jax.debug.print('\nat j = 1, midpoint, backward: ||residual|| = {x}', x = jnp.linalg.norm(m_mid(p[j,:],pjp1,ujp1)) )

        # jax.debug.print('\np = {x}\n', x = p)

        # after that bdf method

        def body(tup):
            j, p, uu, args, dict_vals = tup

            pjp4 = p[j + 4, :]
            pjp3 = p[j + 3, :]
            pjp2 = p[j + 2, :]
            pjp1 = p[j + 1, :]
            uj = uu[j + 1, :]
            aij = tuple(ai[j + 1] for ai in args)
            tj = tt[j + 1]

            y = solver_bdf_rev(pjp1, pjp1, pjp2, pjp3, pjp4, uj, *aij,
                               **{key: val[j+1] for key, val in zip(dict_args.keys(), dict_vals)})
            p = p.at[j, :].set(y)

            # jax.debug.print('\n backward bdf: j = {x}', x = j)

            # jax.debug.print('j = {x}', x = j)
            # jax.debug.print('||residual|| = {x}', x = jnp.linalg.norm(m_bdf(y,pjp1,pjp2,pjp3,pjp4,uj)))

            return j - 1, p, uu, args, dict_vals

        def cond(tup):
            j = tup[0]
            return jnp.greater(j, -1)

        _, p, _, _, _ = jax.lax.while_loop(cond, body, (nt - 5, p, uu, func_args, tuple(dict_args.values())))

        # jax.debug.print('\np = {x}\n', x = p)

        return p