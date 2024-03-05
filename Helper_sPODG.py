import numpy as np
from sklearn.utils.extmath import randomized_svd
from scipy import interpolate
from scipy.linalg import block_diag
from scipy.optimize import root
from scipy import sparse

import sys
import os

sys.path.append('./sPOD/lib/')

########################################################################################################################
# sPOD Galerkin helper functions
from sPOD_tools import shifted_rPCA, shifted_POD, build_all_frames, give_interpolation_error
from transforms import transforms


# This function is totally problem / setup dependent. We need to update all the aspects of this function accordingly
def Shifts_1D(SnapShotMatrix, X, t):
    Nx = int(np.size(X))
    Nt = int(np.size(t))
    NumComovingFrames = 1
    delta = np.zeros((NumComovingFrames, Nt), dtype=float)

    FlameFrontRightPos = np.zeros(Nt, dtype=float)
    for n in range(Nt):
        Var = SnapShotMatrix[:, n]
        FlameFrontRightPos[n] = X[np.argmax(Var)]
    refvalue_rightfront = FlameFrontRightPos[0]
    for n in range(Nt):
        delta[0, n] = - abs(FlameFrontRightPos[n] - refvalue_rightfront)

    # # Correction for the very first time step. As the suppy mass fraction is 1 at the 0th time step therefore the
    # # shifts cannot be computed for that time step therefore we assume the shifts at 0th time step to be equal to the
    # # 1st time step.
    # delta[0, 0] = delta[0, 1]

    deltaold = delta.copy()

    tmpShift = [delta[0, :]]
    # smoothing
    f1 = interpolate.interp1d(np.asarray([0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]),
                              np.asarray([tmpShift[0][0],
                                          tmpShift[0][Nt // 4],
                                          tmpShift[0][Nt // 2],
                                          tmpShift[0][3 * Nt // 4],
                                          tmpShift[0][-1]]),
                              kind='cubic')
    s1 = f1(np.arange(0, Nt))
    delta[0, :] = s1

    deltanew = delta

    return deltanew, deltaold


def srPCA_1D(q, delta, X, t, spod_iter):
    Nx = np.size(X)
    Nt = np.size(t)

    data_shape = [Nx, 1, 1, Nt]
    dx = X[1] - X[0]
    L = [X[-1]]

    # Create the transformations
    trafo_1 = transforms(data_shape, L, shifts=delta[0],
                         dx=[dx],
                         use_scipy_transform=False,
                         interp_order=5)

    # Run the algorithm
    trafos = [trafo_1]

    # Transformation interpolation error
    interp_err = give_interpolation_error(np.reshape(q, data_shape), trafo_1)
    print("Transformation interpolation error =  %4.4e " % interp_err)
    qmat = np.reshape(q, [-1, Nt])
    [N, M] = np.shape(qmat)
    mu0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.01
    lambd0 = 1 / np.sqrt(np.maximum(M, N)) * 1
    ret = shifted_rPCA(qmat, trafos, nmodes_max=60, eps=1e-5, Niter=spod_iter, use_rSVD=True, mu=mu0, lambd=lambd0,
                       dtol=1e-4)

    # Extract frames modes and error
    qframes, qtilde, rel_err = ret.frames, ret.data_approx, ret.rel_err_hist
    modes_list = [qframes[0].Nmodes]
    V = [qframes[0].modal_system["U"]]

    qframes = [qframes[0].build_field()]

    return qtilde, modes_list[0], V[0], qframes[0]


# Offline phase general functions for primal system
def central_FDMatrix(order, Nx, dx):
    if order == 2:
        pass
    elif order == 4:
        pass
    elif order == 6:
        Coeffs = np.array([-1, 9, -45, 0, 45, -9, 1]) / 60
        diagonalLow = int(-(len(Coeffs) - 1) / 2)
        diagonalUp = int(-diagonalLow)

        D_1 = sparse.csr_matrix(np.zeros((Nx, Nx), dtype=float))

        for k in range(diagonalLow, diagonalUp + 1):
            D_1 = D_1 + Coeffs[k - diagonalLow] * sparse.csr_matrix(np.diag(np.ones(Nx - abs(k)), k))
            if k < 0:
                D_1 = D_1 + Coeffs[k - diagonalLow] * sparse.csr_matrix(
                    np.diag(np.ones(abs(k)), Nx + k))
            if k > 0:
                D_1 = D_1 + Coeffs[k - diagonalLow] * sparse.csr_matrix(
                    np.diag(np.ones(abs(k)), -Nx + k))

    return D_1 * (1 / dx)


def subsample(X, num_sample):
    active_subspace_factor = -1

    # sampling points for the shifts
    delta_samples = np.linspace(0, X[-1], num_sample)

    delta_sampled = [active_subspace_factor * delta_samples,
                     np.zeros_like(delta_samples),
                     delta_samples]

    return np.array(delta_sampled)


def get_T(delta_s, X, t):
    from transforms import transforms

    Nx = len(X)
    Nt = len(t)

    data_shape = [Nx, 1, 1, Nt]
    dx = X[1] - X[0]
    L = [X[-1]]

    # Create the transformations
    trafo_1 = transforms(data_shape, L, shifts=delta_s[0],
                         dx=[dx],
                         use_scipy_transform=False,
                         interp_order=5)

    return trafo_1.shifts_pos, trafo_1


def make_V_W_delta(U, T_delta, D, num_sample):
    V_delta = []
    W_delta = []

    for it in range(num_sample):
        V11 = T_delta[it] @ U
        V_delta.append(V11)

        W11 = D @ (T_delta[it] @ U)
        W_delta.append(W11)

    return V_delta, W_delta


def make_LHS_mat_offline_primal(V_delta, W_delta):
    LHS_mat = []

    # D(a) matrices are dynamic in nature thus need to be included in the time integration part
    for it in range(len(V_delta)):
        LHS11 = V_delta[it].transpose() @ V_delta[it]
        LHS12 = V_delta[it].transpose() @ W_delta[it]
        LHS22 = W_delta[it].transpose() @ W_delta[it]

        LHS_mat.append([LHS11, LHS12, LHS22])

    return LHS_mat


def make_RHS_mat_offline_primal(V_delta, W_delta, A):
    RHS_mat = []

    for it in range(len(V_delta)):
        A_1 = (V_delta[it].transpose() @ A) @ V_delta[it]
        A_2 = (W_delta[it].transpose() @ A) @ V_delta[it]

        RHS_mat.append([A_1, A_2])

    return RHS_mat


def make_control_mat_offline_primal(V_delta, W_delta, psi):
    C_mat = []

    for it in range(len(V_delta)):
        C_1 = (V_delta[it].transpose() @ psi)
        C_2 = (W_delta[it].transpose() @ psi)

        C_mat.append([C_1, C_2])

    return C_mat


def make_target_term_matrices(Vd_p, Vd_a, Wd_a):
    T1 = []
    for it in range(len(Vd_p)):
        T_11 = Vd_a[it].transpose() @ Vd_p[it]
        T_21 = Wd_a[it].transpose() @ Vd_p[it]

        T1.append([T_11, T_21])

    return T1


def make_control_update_mat(V_delta, W_delta, psi):
    Ct_mat = []

    for it in range(len(V_delta)):
        Ct_1 = psi.transpose() @ V_delta[it]
        Ct_2 = psi.transpose() @ W_delta[it]

        Ct_mat.append([Ct_1, Ct_2])

    return Ct_mat


# Online phase functions
def findIntervalAndGiveInterpolationWeight_1D(xPoints, xStar):
    intervalBool_arr = np.where(xStar >= xPoints, 1, 0)
    mixed = intervalBool_arr[:-1] * (1 - intervalBool_arr)[1:]
    index = np.sum(mixed * np.arange(0, mixed.shape[0]))

    intervalIdx = index
    alpha = (xPoints[intervalIdx + 1] - xStar) / (
            xPoints[intervalIdx + 1] - xPoints[intervalIdx])

    return intervalIdx, alpha


def make_Da(a):
    D_a = a[:len(a) - 1]

    return D_a[:, np.newaxis]


def make_LHS_mat_online_primal(LHS_matrix, Da, intervalIdx, weight):
    M11 = weight * LHS_matrix[intervalIdx][0] + (1 - weight) * LHS_matrix[intervalIdx + 1][0]
    M12 = (weight * LHS_matrix[intervalIdx][1] + (1 - weight) * LHS_matrix[intervalIdx + 1][1]) @ Da
    M21 = M12.transpose()
    M22 = (Da.transpose() @ (weight * LHS_matrix[intervalIdx][2] +
                                             (1 - weight) * LHS_matrix[intervalIdx + 1][2])) @ Da
    M = np.block([
        [M11, M12],
        [M21, M22]
    ])

    # if np.linalg.cond(M, p='fro') == np.inf:
    #     M_reg = M + 1e-12 * np.eye(M.shape[0], M.shape[1])
    # else:
    #     M_reg = M

    return M


def make_RHS_mat_online_primal(RHS_matrix, Da, intervalIdx, weight):
    A11 = weight * RHS_matrix[intervalIdx][0] + (1 - weight) * RHS_matrix[intervalIdx + 1][0]
    A21 = Da.transpose() @ (weight * RHS_matrix[intervalIdx][1] +
                                            (1 - weight) * RHS_matrix[intervalIdx + 1][1])
    A = np.block([
        [A11, np.zeros((A11.shape[0], 1))],
        [A21, np.zeros((A21.shape[0], 1))]
    ])

    return A


def make_control_mat_online_primal(f, C, Da, intervalIdx, weight):
    C1 = (weight * C[intervalIdx][0] + (1 - weight) * C[intervalIdx + 1][0]) @ f
    C2 = Da.transpose() @ ((weight * C[intervalIdx][1] + (1 - weight) * C[intervalIdx + 1][1]) @ f)

    C = np.concatenate((C1, C2))

    return C


def make_target_mat_online_primal(T_a, Vda, Wda, qs_target, a_, Da, intervalIdx, weight):

    VdaTVdp = (weight * T_a[intervalIdx][0] + (1 - weight) * T_a[intervalIdx + 1][0])
    WdaTVdp = (weight * T_a[intervalIdx][1] + (1 - weight) * T_a[intervalIdx + 1][1])
    V_a = weight * Vda[intervalIdx] + (1 - weight) * Vda[intervalIdx + 1]
    W_a = weight * Wda[intervalIdx] + (1 - weight) * Wda[intervalIdx + 1]
    C1 = (VdaTVdp @ a_ - V_a.transpose() @ qs_target)
    C2 = Da.transpose() @ (WdaTVdp @ a_ - W_a.transpose() @ qs_target)

    C = np.concatenate((C1, C2))

    return C


def get_online_state(T_trafo, V, a, X, t):
    Nx = len(X)
    Nt = len(t)
    qs_online = np.zeros((Nx, Nt))
    q = V @ a

    qs_online += T_trafo[0].apply(q)

    return qs_online


# Auxiliary functions
def findIntervals(delta_s, delta):
    Nt = len(delta)
    intIds = []
    weights = []
    for i in range(Nt):
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(delta_s[2], -delta[i])
        intIds.append(intervalIdx)
        weights.append(weight)

    return intIds, weights


# FOTR reduced functions
def make_LHS_mat_offline_primal_red(V_delta, W_delta):

    # D(a) matrices are dynamic in nature thus need to be included in the time integration part
    LHS11 = V_delta[0].transpose() @ V_delta[0]
    LHS12 = V_delta[0].transpose() @ W_delta[0]
    LHS22 = W_delta[0].transpose() @ W_delta[0]

    LHS_mat = [LHS11, LHS12, LHS22]

    return LHS_mat


def make_RHS_mat_offline_primal_red(V_delta, W_delta, A):
    A_1 = (V_delta[0].transpose() @ A) @ V_delta[0]
    A_2 = (W_delta[0].transpose() @ A) @ V_delta[0]

    RHS_mat = [A_1, A_2]

    return RHS_mat


def make_LHS_mat_online_primal_red(LHS_matrix, Da):
    M11 = LHS_matrix[0]
    M12 = LHS_matrix[1] @ Da
    M21 = M12.transpose()
    M22 = (Da.transpose() @ LHS_matrix[2]) @ Da

    M = np.block([
        [M11, M12],
        [M21, M22]
    ])

    return M


def make_RHS_mat_online_primal_red(RHS_matrix, Da):
    A11 = RHS_matrix[0]
    A21 = Da.transpose() @ RHS_matrix[1]

    A = np.block([
        [A11, np.zeros((A11.shape[0], 1))],
        [A21, np.zeros((A21.shape[0], 1))]
    ])

    return A

