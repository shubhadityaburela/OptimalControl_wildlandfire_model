from Helper import *


def rk4(RHS: callable,
        q0: np.ndarray,
        u: np.ndarray,
        dt,
        *args) -> np.ndarray:
    k1 = RHS(q0, u, *args)
    k2 = RHS(q0 + dt / 2 * k1, u, *args)
    k3 = RHS(q0 + dt / 2 * k2, u, *args)
    k4 = RHS(q0 + dt * k3, u, *args)

    u1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return u1


def rk4_(RHS: callable,
         q0: np.ndarray,
         u: np.ndarray,
         dt,
         *args):
    k1 = RHS(q0, u, *args)
    k2 = RHS(q0 + dt / 2 * k1, u, *args)
    k3 = RHS(q0 + dt / 2 * k2, u, *args)
    k4 = RHS(q0 + dt * k3, u, *args)

    u1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return u1, [k1, k2, k3, k4]



def rk4__(RHS: callable,
          q0: np.ndarray,
          u: np.ndarray,
          dt,
          q0_dot,
          z_dot,
          *args):
    k1 = RHS(q0, u, q0_dot[3], z_dot[3], *args)
    k2 = RHS(q0 + dt / 2 * k1, u, q0_dot[2], z_dot[2], *args)
    k3 = RHS(q0 + dt / 2 * k2, u, q0_dot[1], z_dot[1], *args)
    k4 = RHS(q0 + dt * k3, u, q0_dot[0], z_dot[0], *args)

    u1 = q0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return u1
