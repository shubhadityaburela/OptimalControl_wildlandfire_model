import numpy as np


def Calc_Grad(lamda, mask, f, qs_adj):
    dL_du = lamda['q_reg'] * f + mask.transpose() @ qs_adj

    return dL_du


def Calc_Grad_PODG(lamda, mask, f, V, as_adj):
    qs_adj = V @ as_adj
    dL_du = lamda['q_reg'] * f + mask.transpose() @ qs_adj
    return dL_du


def Calc_Grad_sPODG(lamda, mask, f, V, intIds, weights, as_adj):
    qs_adj = np.zeros((mask.shape[0], f.shape[1]))
    for i in range(f.shape[1]):
        V_delta = weights[i] * V[intIds[i]] + (1 - weights[i]) * V[intIds[i] + 1]
        qs_adj[:, i] = V_delta @ as_adj[:-1, i]

    dL_du = lamda['q_reg'] * f + mask.transpose() @ qs_adj

    return dL_du
