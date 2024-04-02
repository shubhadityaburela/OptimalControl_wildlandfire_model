import numpy as np

from Helper_sPODG import make_Da


def Calc_Grad(lamda, mask, f, qs_adj):
    dL_du = lamda['q_reg'] * f + mask.transpose() @ qs_adj

    return dL_du


def Calc_Grad_PODG(lamda, psir_p, f, as_adj):
    dL_du = lamda['q_reg'] * f + psir_p.transpose() @ as_adj
    return dL_du


def Calc_Grad_sPODG(lamda, f, ct, intIds, weights, as_, as_adj, **kwargs):
    as_adj_1 = np.zeros_like(f)
    as_adj_2 = np.zeros_like(f)
    for i in range(f.shape[1]):
        m1 = (weights[i] * ct[intIds[i]][0] + (1 - weights[i]) * ct[intIds[i] + 1][0]).transpose()
        m2 = (weights[i] * ct[intIds[i]][1] + (1 - weights[i]) * ct[intIds[i] + 1][1]).transpose()
        Da = make_Da(as_[:, i])
        as_adj_1[:, i] = m1 @ as_adj[:-1, i]
        as_adj_2[:, i] = m2 @ (Da @ as_adj[-1:, i])

    dL_du = lamda['q_reg'] * f + as_adj_1 + as_adj_2

    return dL_du
