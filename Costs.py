from Helper import *


def Calc_Cost(q, q_target, f, lamda, **kwargs):
    q_res = q - q_target
    cost = 1 / 2 * (integrate_cost(q_res, **kwargs)) + (lamda['q_reg'] / 2) * (trapezoidal_integration(f, **kwargs))

    return cost


def Calc_Cost_PODG(V, as_, qs_target, f, psi, lamda, **kwargs):
    q_res = V @ as_ - qs_target

    cost = 1 / 2 * (integrate_cost(q_res, **kwargs)) + (lamda['q_reg'] / 2) * (
        trapezoidal_integration_control(f, **kwargs))

    return cost


def Calc_Cost_sPODG(V, as_, qs_target, f, psi, lamda, intIds, weights, **kwargs):
    q = np.zeros_like(qs_target)
    for i in range(f.shape[1]):
        V_delta = weights[i] * V[intIds[i]] + (1 - weights[i]) * V[intIds[i] + 1]
        q[:, i] = V_delta @ as_[:-1, i]

    q_res = q - qs_target

    cost = 1 / 2 * (integrate_cost(q_res, **kwargs)) + (lamda['q_reg'] / 2) * (
        trapezoidal_integration_control(f, **kwargs))

    return cost


def Calc_Cost_sPODG_new(as_, as_target, z_target, f, lamda, **kwargs):
    a_res = as_[:-1, :] - as_target
    z_res = as_[-1:, :] - z_target

    cost = 1 / 2 * (integrate_cost_TA(a_res, **kwargs)) + 1 / 2 * (integrate_cost_TA(z_res, **kwargs)) \
           + (lamda['q_reg'] / 2) * (trapezoidal_integration_control(f, **kwargs))

    return cost
