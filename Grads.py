import jax.numpy as jnp


def Calc_Grad(lamda, mask, f, qs_adj, **kwargs):
    dL_du = lamda['q_reg'] * f + mask.transpose() @ qs_adj

    return dL_du


def Calc_Grad_PODG(lamda, mask, f, V, as_adj, **kwargs):
    qs_adj = V @ as_adj
    dL_du = lamda['q_reg'] * f + mask.transpose() @ qs_adj
    return dL_du


def Calc_Grad_sPODG(lamda, mask, f, Vs_a, as_adj, as_, X, **kwargs):
    qs_adj = jnp.zeros((mask.shape[0], f.shape[1]))
    for i in range(f.shape[1]):
        z = jnp.asarray([as_[-1, i]])
        Vd = jnp.zeros_like(Vs_a)
        for col in range(Vs_a.shape[1]):
            Vd = Vd.at[:, col].set(jnp.interp(X + z, X, Vs_a[:, col], period=X[-1]))
        qs_adj = qs_adj.at[:, i].set(Vd @ as_adj[:-1, i])

    dL_du = lamda['q_reg'] * f + mask.transpose() @ qs_adj

    return dL_du
