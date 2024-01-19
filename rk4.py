from Helper import *


def rk4(RHS: callable,
        q0: jnp.ndarray,
        u: jnp.ndarray,
        dt,
        *args) -> jnp.ndarray:
    k1 = RHS(q0, u, *args)
    k2 = RHS(q0 + dt / 2 * k1, u, *args)
    k3 = RHS(q0 + dt / 2 * k2, u, *args)
    k4 = RHS(q0 + dt * k3, u, *args)

    u1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return u1


def rk4_sp(RHS: callable,
           q0: jnp.ndarray,
           u: jnp.ndarray,
           dt,
           *args):
    k1 = RHS(q0, u, *args)
    k2 = RHS(q0 + dt / 2 * k1, u, *args)
    k3 = RHS(q0 + dt / 2 * k2, u, *args)
    k4 = RHS(q0 + dt * k3, u, *args)

    u1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return u1, k1
