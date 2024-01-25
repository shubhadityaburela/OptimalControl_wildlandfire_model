from Helper import *


def Calc_Cost(q, q_target, f, lamda, **kwargs):
    q_res = q - q_target
    cost = 1 / 2 * (L2norm(q_res, **kwargs)) ** 2 + (lamda['q_reg'] / 2) * (L2norm(f, **kwargs)) ** 2

    return cost


def Calc_Cost_PODG(V, as_, qs_target, f, lamda, **kwargs):
    a_res = V @ as_ - qs_target

    cost = 1 / 2 * (L2norm_ROM(a_res, **kwargs)) ** 2 + (lamda['q_reg'] / 2) * (L2norm_ROM(f, **kwargs)) ** 2

    return cost


def Calc_Cost_sPODG(V, as_, qs_target, f, lamda, intIds, weights, **kwargs):
    q = np.zeros_like(qs_target)
    for i in range(f.shape[1]):
        V_delta = weights[i] * V[intIds[i]] + (1 - weights[i]) * V[intIds[i] + 1]
        q[:, i] = V_delta @ as_[:-1, i]

    q_res = q - qs_target

    cost = 1 / 2 * (L2norm_ROM(q_res, **kwargs)) ** 2 + (lamda['q_reg'] / 2) * (L2norm_ROM(f, **kwargs)) ** 2

    return cost
