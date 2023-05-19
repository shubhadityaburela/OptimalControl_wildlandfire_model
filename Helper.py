import numpy as np


def Calc_Cost(q, q_target, f, lamda, num_var):

    T = q[:int(q.shape[0]) // num_var, :]
    T_target = q_target[:int(q.shape[0]) // num_var, :]
    f_T = f[:int(q.shape[0]) // num_var, :]

    return (np.linalg.norm(T - T_target)) ** 2 / 2 + (lamda / 2) * (np.linalg.norm(f_T)) ** 2


def Update_Control(f, omega, lamda, mask, qs_adj):

    mask = np.concatenate((mask, np.zeros_like(mask)), axis=0)

    return (1 - omega * lamda) * f - omega * mask * qs_adj
