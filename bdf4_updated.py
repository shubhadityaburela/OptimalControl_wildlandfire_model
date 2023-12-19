from Helper import *


@jax.profiler.annotate_function
def bdf4_updated(f: callable,
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
    :param tt: timepoints, assumed to be evenly spaced
    :param x0: initial or final value
    :param uu: control input at timepoints, shape = (len(tt), N)
    :param type: 'forward' or 'backward'
    :param debug: if True, print debug information
    :return: solution of the problem in the form
        x[i,:] = x(tt[i])
        p[i,:] = p(tt[i])
    """

    N = len(x0)  # system dimension
    nt = len(tt)  # number of timepoints
    dt = tt[1] - tt[0]  # timestep

    # identity matrix
    eye = jnp.eye(N)

    # jacobian of f
    Df = jacobian(f, argnums=0)

    @jax.profiler.annotate_function
    def F_bdf(xj, xj1, xj2, xj3, xj4, uj, *args, **kwargs):
        return \
                25 * xj \
                - 48 * xj1 \
                + 36 * xj2 \
                - 16 * xj3 \
                + 3 * xj4 \
                - 12 * dt * f(xj, uj, *args, **kwargs)

    @jax.profiler.annotate_function
    def F_bdf_rev(xj, xj1, xj2, xj3, xj4, uj, *args, **kwargs):
        return \
                25 * xj \
                - 48 * xj1 \
                + 36 * xj2 \
                - 16 * xj3 \
                + 3 * xj4 \
                + 12 * dt * f(xj, uj, *args, **kwargs)

    @jax.profiler.annotate_function
    def DF_bdf(xj, xj1, xj2, xj3, xj4, uj, *args, **kwargs):
        return 25 * eye - 12 * dt * Df(xj, uj, *args, **kwargs)

    @jax.profiler.annotate_function
    def DF_bdf_rev(xj, xj1, xj2, xj3, xj4, uj, *args, **kwargs):
        return 25 * eye + 12 * dt * Df(xj, uj, *args, **kwargs)

    # for first four values
    @jax.profiler.annotate_function
    def F_start(x1234):
        # the magic coefficients in this function come from a polynomial approach
        # the approach calculates 4 timesteps at once and is of order 4.
        # for details, see here: https://colab.research.google.com/drive/1DHtzD3U1PsMQbn-nsBF3crh_Tj3cgmMd?usp=sharing

        x1 = x1234[:N]
        x2 = x1234[N:2 * N]
        x3 = x1234[2 * N:3 * N]
        x4 = x1234[3 * N:]

        # entries of F
        pprime_t1 = -3.0 * x0 - 10.0 * x1 + 18.0 * x2 - 6.0 * x3 + x4
        pprime_t2 = x0 - 8.0 * x1 + 8.0 * x3 - 1.0 * x4
        pprime_t3 = -1.0 * x0 + 6.0 * x1 - 18.0 * x2 + 10.0 * x3 + 3.0 * x4
        pprime_t4 = 3.0 * x0 - 16.0 * x1 + 36.0 * x2 - 48.0 * x3 + 25.0 * x4

        return jnp.hstack((
            pprime_t1 - 12 * dt * f(x1, uu[1, :], *tuple(ai[1] for ai in func_args),
                                    **{key: value[1] for key, value in dict_args.items()}),
            pprime_t2 - 12 * dt * f(x2, uu[2, :], *tuple(ai[2] for ai in func_args),
                                    **{key: value[2] for key, value in dict_args.items()}),
            pprime_t3 - 12 * dt * f(x3, uu[3, :], *tuple(ai[3] for ai in func_args),
                                    **{key: value[3] for key, value in dict_args.items()}),
            pprime_t4 - 12 * dt * f(x4, uu[4, :], *tuple(ai[4] for ai in func_args),
                                    **{key: value[4] for key, value in dict_args.items()})
        ))

    @jax.profiler.annotate_function
    def F_start_rev(x1234):
        # the magic coefficients in this function come from a polynomial approach
        # the approach calculates 4 timesteps at once and is of order 4.
        # for details, see here: https://colab.research.google.com/drive/1DHtzD3U1PsMQbn-nsBF3crh_Tj3cgmMd?usp=sharing

        x1 = x1234[:N]
        x2 = x1234[N:2 * N]
        x3 = x1234[2 * N:3 * N]
        x4 = x1234[3 * N:]

        # entries of F
        pprime_t1 = -3.0 * x0 - 10.0 * x1 + 18.0 * x2 - 6.0 * x3 + x4
        pprime_t2 = x0 - 8.0 * x1 + 8.0 * x3 - 1.0 * x4
        pprime_t3 = -1.0 * x0 + 6.0 * x1 - 18.0 * x2 + 10.0 * x3 + 3.0 * x4
        pprime_t4 = 3.0 * x0 - 16.0 * x1 + 36.0 * x2 - 48.0 * x3 + 25.0 * x4

        return jnp.hstack((
            pprime_t1 + 12 * dt * f(x1, uu[1, :], *tuple(ai[1] for ai in func_args),
                                    **{key: value[1] for key, value in dict_args.items()}),
            pprime_t2 + 12 * dt * f(x2, uu[2, :], *tuple(ai[2] for ai in func_args),
                                    **{key: value[2] for key, value in dict_args.items()}),
            pprime_t3 + 12 * dt * f(x3, uu[3, :], *tuple(ai[3] for ai in func_args),
                                    **{key: value[3] for key, value in dict_args.items()}),
            pprime_t4 + 12 * dt * f(x4, uu[4, :], *tuple(ai[4] for ai in func_args),
                                    **{key: value[4] for key, value in dict_args.items()})
        ))

    @jax.profiler.annotate_function
    def DF_start(x1234):
        # the magic coefficients in this function come from a polynomial approach
        # the approach calculates 4 timesteps at once and is of order 4.
        # for details, see here: https://colab.research.google.com/drive/1DHtzD3U1PsMQbn-nsBF3crh_Tj3cgmMd?usp=sharing

        x1 = x1234[:N]
        x2 = x1234[N:2 * N]
        x3 = x1234[2 * N:3 * N]
        x4 = x1234[3 * N:]

        # first row
        DF_11 = -10.0 * eye - 12 * dt * Df(x1, uu[1, :], *tuple(ai[1] for ai in func_args),
                                           **{key: value[1] for key, value in dict_args.items()})
        DF_12 = 18.0 * eye
        DF_13 = -6.0 * eye
        DF_14 = 1.0 * eye
        DF_1 = jnp.hstack((DF_11, DF_12, DF_13, DF_14))

        # second row
        DF_21 = -8.0 * eye
        DF_22 = 0.0 * eye - 12 * dt * Df(x2, uu[2, :], *tuple(ai[2] for ai in func_args),
                                         **{key: value[2] for key, value in dict_args.items()})
        DF_23 = 8.0 * eye
        DF_24 = -1.0 * eye
        DF_2 = jnp.hstack((DF_21, DF_22, DF_23, DF_24))

        # third row
        DF_31 = 6.0 * eye
        DF_32 = -18.0 * eye
        DF_33 = 10.0 * eye - 12 * dt * Df(x3, uu[3, :], *tuple(ai[3] for ai in func_args),
                                          **{key: value[3] for key, value in dict_args.items()})
        DF_34 = 3.0 * eye
        DF_3 = jnp.hstack((DF_31, DF_32, DF_33, DF_34))

        # fourth row
        DF_41 = -16.0 * eye
        DF_42 = 36.0 * eye
        DF_43 = -48.0 * eye
        DF_44 = 25.0 * eye - 12 * dt * Df(x4, uu[4, :], *tuple(ai[4] for ai in func_args),
                                          **{key: value[4] for key, value in dict_args.items()})
        DF_4 = jnp.hstack((DF_41, DF_42, DF_43, DF_44))

        # return all rows together
        return jnp.vstack((DF_1, DF_2, DF_3, DF_4))

    @jax.profiler.annotate_function
    def DF_start_rev(x1234):
        # the magic coefficients in this function come from a polynomial approach
        # the approach calculates 4 timesteps at once and is of order 4.
        # for details, see here: https://colab.research.google.com/drive/1DHtzD3U1PsMQbn-nsBF3crh_Tj3cgmMd?usp=sharing

        x1 = x1234[:N]
        x2 = x1234[N:2 * N]
        x3 = x1234[2 * N:3 * N]
        x4 = x1234[3 * N:]

        # first row
        DF_11 = -10.0 * eye + 12 * dt * Df(x1, uu[1, :], *tuple(ai[1] for ai in func_args),
                                           **{key: value[1] for key, value in dict_args.items()})
        DF_12 = 18.0 * eye
        DF_13 = -6.0 * eye
        DF_14 = 1.0 * eye
        DF_1 = jnp.hstack((DF_11, DF_12, DF_13, DF_14))

        # second row
        DF_21 = -8.0 * eye
        DF_22 = 0.0 * eye + 12 * dt * Df(x2, uu[2, :], *tuple(ai[2] for ai in func_args),
                                         **{key: value[2] for key, value in dict_args.items()})
        DF_23 = 8.0 * eye
        DF_24 = -1.0 * eye
        DF_2 = jnp.hstack((DF_21, DF_22, DF_23, DF_24))

        # third row
        DF_31 = 6.0 * eye
        DF_32 = -18.0 * eye
        DF_33 = 10.0 * eye + 12 * dt * Df(x3, uu[3, :], *tuple(ai[3] for ai in func_args),
                                          **{key: value[3] for key, value in dict_args.items()})
        DF_34 = 3.0 * eye
        DF_3 = jnp.hstack((DF_31, DF_32, DF_33, DF_34))

        # fourth row
        DF_41 = -16.0 * eye
        DF_42 = 36.0 * eye
        DF_43 = -48.0 * eye
        DF_44 = 25.0 * eye + 12 * dt * Df(x4, uu[4, :], *tuple(ai[4] for ai in func_args),
                                          **{key: value[4] for key, value in dict_args.items()})
        DF_4 = jnp.hstack((DF_41, DF_42, DF_43, DF_44))

        # return all rows together
        return jnp.vstack((DF_1, DF_2, DF_3, DF_4))

    solver_start = newton(F_start, Df=DF_start)
    solver_bdf = newton(F_bdf, Df=DF_bdf)

    solver_start_rev = newton(F_start_rev, Df=DF_start_rev)
    solver_bdf_rev = newton(F_bdf_rev, Df=DF_bdf_rev)



    if type == 'forward':

        x = jnp.zeros((nt, N))
        x = x.at[0, :].set(x0)

        # first few steps with polynomial interpolation technique
        x1234 = solver_start(jnp.hstack((x0, x0, x0, x0)))
        x = x.at[1, :].set(x1234[:N])
        x = x.at[2, :].set(x1234[N:2 * N])
        x = x.at[3, :].set(x1234[2 * N:3 * N])

        @jax.profiler.annotate_function
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

            if debug:
                jax.debug.print('j = {x}', x=j)
            # jax.debug.print( 'iter = {x}', x = i)

            # jax.debug.print('\n forward bdf: j = {x}', x = j)

            # jax.debug.print('log10(||residual||) = {x}', x = jnp.log10(jnp.linalg.norm(m_bdf(y,xjm1,xjm2,xjm3,xjm4,uj))) )

            return x, uu, args, dict_vals

        x, _, _, _ = jax.lax.fori_loop(4, nt, body, (x, uu, func_args, tuple(dict_args.values())), )

        return x

    else:  # type == 'backward'

        p = jnp.zeros((nt, N))
        p = p.at[-1, :].set(x0)

        # first few steps with polynomial interpolation technique
        p1234 = solver_start_rev(jnp.hstack((x0, x0, x0, x0)))
        p = p.at[-2, :].set(p1234[:N])
        p = p.at[-3, :].set(p1234[N:2 * N])
        p = p.at[-4, :].set(p1234[2 * N:3 * N])

        # after that bdf method
        @jax.profiler.annotate_function
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

