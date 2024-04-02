import numpy as np
from sklearn.utils.extmath import randomized_svd
from scipy import interpolate
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

    # sampling points for the shifts (The shift values can range from 0 to X/2 and then is a mirror image for X/2 to X)
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


def make_target_term_matrices(Vd_p, Vd_a, Wd_a, qs_target):
    T1 = []
    T2 = []
    for it in range(len(Vd_p)):
        T_11 = Vd_a[it].transpose() @ Vd_p[it]
        T_12 = Vd_a[it].transpose() @ qs_target
        T_21 = Wd_a[it].transpose() @ Vd_p[it]
        T_22 = Wd_a[it].transpose() @ qs_target

        T1.append([T_11, T_21])
        T2.append([T_12, T_22])

    return T1, T2


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


################################# sPODG adjoint helper functions ##################################
def D_dash(z, r):
    arr = np.repeat(z, r)
    return np.diag(arr)


def DT_dash(arr):
    return np.atleast_2d(arr).T


def dv_dt(Dfd, V, dz_dt):
    return (Dfd @ V) * dz_dt


def dvT_dt(Dfd, V, dz_dt):
    return (Dfd @ V).transpose() * dz_dt


def dw_dt(Dfd, W, dz_dt):
    return (Dfd @ W) * dz_dt


def dwT_dt(Dfd, W, dz_dt):
    return (Dfd @ W).transpose() * dz_dt


def dD_dt(a_dot):
    return np.atleast_2d(a_dot)


def dDT_dt(a_dot):
    return dD_dt(a_dot).transpose()


def dN_dt(Dfd, V, W, dz_dt):
    return dvT_dt(Dfd, V, dz_dt) @ W + V.transpose() @ dw_dt(Dfd, W, dz_dt)


def dNT_dt(Dfd, V, W, dz_dt):
    return dwT_dt(Dfd, W, dz_dt) @ V + W.transpose() @ dv_dt(Dfd, V, dz_dt)


def dM1_dt(Dfd, V, dz_dt):
    return dvT_dt(Dfd, V, dz_dt) @ V + V.transpose() @ dv_dt(Dfd, V, dz_dt)


def dM2_dt(Dfd, W, dz_dt):
    return dwT_dt(Dfd, W, dz_dt) @ W + W.transpose() @ dw_dt(Dfd, W, dz_dt)


def V_dash(Dfd, V):
    return Dfd @ V


def VT_dash(Dfd, V):   # We have assumed that (V')^T = (V^T)'
    return V_dash(Dfd, V).transpose()


def W_dash(Dfd, W):
    return Dfd @ W


def WT_dash(Dfd, W):
    return W_dash(Dfd, W).transpose()


def N_dash(Dfd, V, W):
    return VT_dash(Dfd, V) @ W + V.transpose() @ W_dash(Dfd, W)


def NT_dash(Dfd, V, W):
    return WT_dash(Dfd, W) @ V + W.transpose() @ V_dash(Dfd, V)


def M1_dash(Dfd, V):
    return VT_dash(Dfd, V) @ V + V.transpose() @ V_dash(Dfd, V)


def M2_dash(Dfd, W):
    return WT_dash(Dfd, W) @ W + W.transpose() @ W_dash(Dfd, W)


def A1_dash(Dfd, V, A):
    return VT_dash(Dfd, V) @ (A @ V) + (V.transpose() @ A) @ V_dash(Dfd, V)


def A2_dash(Dfd, V, W, A):
    return WT_dash(Dfd, W) @ (A @ V) + (W.transpose() @ A) @ V_dash(Dfd, V)


def dv_dash_dt(Dfd, V, dz_dt):
    return (Dfd @ V_dash(Dfd, V)) * dz_dt


def dw_dash_dt(Dfd, W, dz_dt):
    return (Dfd @ W_dash(Dfd, W)) * dz_dt


def dvT_dash_dt(Dfd, V, dz_dt):
    return (Dfd @ VT_dash(Dfd, V).transpose()).transpose() * dz_dt


def dwT_dash_dt(Dfd, W, dz_dt):
    return (Dfd @ WT_dash(Dfd, W).transpose()).transpose() * dz_dt


def dM1_dash_dt(Dfd, V, dz_dt):
    return dvT_dash_dt(Dfd, V, dz_dt) @ V + VT_dash(Dfd, V) @ dv_dt(Dfd, V, dz_dt) \
        + dvT_dt(Dfd, V, dz_dt) @ V_dash(Dfd, V) + V.transpose() @ dv_dash_dt(Dfd, V, dz_dt)


def dM2_dash_dt(Dfd, W, dz_dt):
    return dwT_dash_dt(Dfd, W, dz_dt) @ W + WT_dash(Dfd, W) @ dw_dt(Dfd, W, dz_dt) \
        + dwT_dt(Dfd, W, dz_dt) @ W_dash(Dfd, W) + W.transpose() @ dw_dash_dt(Dfd, W, dz_dt)


def dN_dash_dt(Dfd, V, W, dz_dt):
    return dvT_dash_dt(Dfd, V, dz_dt) @ W + VT_dash(Dfd, V) @ dw_dt(Dfd, W, dz_dt) \
        + dvT_dt(Dfd, V, dz_dt) @ W_dash(Dfd, W) + V.transpose() @ dw_dash_dt(Dfd, W, dz_dt)


# Mass matrix assembly
def Q11(M1):
    return M1


def Q12(N, D):
    return D.transpose() @ N.transpose()


def Q21(N, D):
    return N @ D


def Q22(M2, D):
    return D.transpose() @ (M2 @ D)


def B11(Dfd, N, A1, V, z_dot, r):
    return A1 - N @ D_dash(z_dot, r) + dM1_dt(Dfd, V, z_dot)


def B12(Dfd, M2, N, A2, psi, D, V, W, a_dot, z_dot, a_s, z, u, r):

    mat1 = dDT_dt(a_dot) @ N.transpose()
    mat2 = D.transpose() @ dNT_dt(Dfd, V, W, z_dot)
    mat3 = DT_dash(N.transpose() @ a_dot)
    mat4 = DT_dash(M2 @ (D @ z_dot))
    mat5 = D.transpose() @ (M2 @ D_dash(z_dot, r))
    mat6 = DT_dash(A2 @ a_s)
    mat7 = D.transpose() @ A2
    mat8 = DT_dash(W.transpose() @ (psi @ u))

    return mat1 + mat2 - mat3 - mat4 - mat5 + mat6 + mat7 + mat8


def B21(Dfd, N, A, psi, D, V, W, a_dot, z_dot, a_s, z, u):

    mat1 = dN_dt(Dfd, V, W, z_dot) @ D
    mat2 = N @ dD_dt(a_dot)
    mat3 = M1_dash(Dfd, V) @ a_dot
    mat4 = N_dash(Dfd, V, W) @ (D @ z_dot)
    mat5 = A1_dash(Dfd, V, A) @ a_s
    mat6 = VT_dash(Dfd, V) @ (psi @ u)

    return mat1 + mat2 - mat3 - mat4 + mat5 + mat6


def B22(Dfd, A, M2, D, psi, V, W, a_dot, z_dot, a_s, z, u):

    mat1 = dDT_dt(a_dot) @ (M2 @ D)
    mat2 = D.transpose() @ (dM2_dt(Dfd, W, z_dot) @ D)
    mat3 = D.transpose() @ (M2 @ dD_dt(a_dot))
    mat4 = D.transpose() @ (NT_dash(Dfd, V, W) @ a_dot)
    mat5 = D.transpose() @ (M2_dash(Dfd, W) @ (D @ z_dot))
    mat6 = D.transpose() @ (A2_dash(Dfd, V, W, A) @ a_s)
    mat7 = D.transpose() @ (WT_dash(Dfd, W) @ (psi @ u))

    return mat1 + mat2 + mat3 - mat4 - mat5 + mat6 + mat7


def C1(V, qs_target, a_s):
    return a_s - V.transpose() @ qs_target


def C2(Dfd, V, qs_target, a_s):
    return np.atleast_1d((V_dash(Dfd, V) @ a_s).transpose() @ (V @ a_s - qs_target))


def Z11(M1):
    return M1


def Z12(N, D):
    return D.transpose() @ N.transpose()


def Z21(N, D):
    return N @ D


def Z22(M2, D):
    return D.transpose() @ (M2 @ D)



def make_LHS_mat_online_adjoint(LHS_matrix, Da, intervalIdx, weight):

    M1 = weight * LHS_matrix[intervalIdx][0] + (1 - weight) * LHS_matrix[intervalIdx + 1][0]
    N = weight * LHS_matrix[intervalIdx][1] + (1 - weight) * LHS_matrix[intervalIdx + 1][1]
    M2 = weight * LHS_matrix[intervalIdx][2] + (1 - weight) * LHS_matrix[intervalIdx + 1][2]

    Q1_1 = Q11(M1)
    Q1_2 = Q12(N, Da)
    Q2_1 = Q21(N, Da)
    Q2_2 = Q22(M2, Da)

    LHS = np.block([
        [Q1_1.transpose(), Q1_2.transpose()],
        [Q2_1.transpose(), Q2_2.transpose()]
    ])

    return LHS



def make_RHS_mat_online_adjoint(RHS_matrix, LHS_matrix, A, psi, a_dot, z_dot, Da, a_, Dfd, Vdp, Wdp, u, intervalIdx,
                                weight):
    r = len(a_) - 1
    a_s = np.atleast_2d(a_[:-1]).T
    z = np.atleast_2d(a_[-1:]).T
    a_dot = np.atleast_2d(a_dot).T
    z_dot = np.atleast_2d(z_dot).T
    u = np.atleast_2d(u).T

    V = weight * Vdp[intervalIdx] + (1 - weight) * Vdp[intervalIdx + 1]
    W = weight * Wdp[intervalIdx] + (1 - weight) * Wdp[intervalIdx + 1]
    N = weight * LHS_matrix[intervalIdx][1] + (1 - weight) * LHS_matrix[intervalIdx + 1][1]
    M2 = weight * LHS_matrix[intervalIdx][2] + (1 - weight) * LHS_matrix[intervalIdx + 1][2]

    A1 = weight * RHS_matrix[intervalIdx][0] + (1 - weight) * RHS_matrix[intervalIdx + 1][0]
    A2 = weight * RHS_matrix[intervalIdx][1] + (1 - weight) * RHS_matrix[intervalIdx + 1][1]

    B1_1 = B11(Dfd, N, A1, V, z_dot, r)
    B1_2 = B12(Dfd, M2, N, A2, psi, Da, V, W, a_dot, z_dot, a_s, z, u, r)
    B2_1 = B21(Dfd, N, A, psi, Da, V, W, a_dot, z_dot, a_s, z, u)
    B2_2 = B22(Dfd, A, M2, Da, psi, V, W, a_dot, z_dot, a_s, z, u)

    RHS = np.block([
        [B1_1.transpose(), B1_2.transpose()],
        [B2_1.transpose(), B2_2.transpose()]
    ])

    return RHS



def make_target_mat_online_adjoint(a_, Dfd, Vdp, qs_target, intervalIdx, weight):

    V = weight * Vdp[intervalIdx] + (1 - weight) * Vdp[intervalIdx + 1]
    C1_1 = C1(V, qs_target, a_[:-1])
    C2_1 = C2(Dfd, V, qs_target, a_[:-1])

    C = np.concatenate((C1_1, C2_1))

    return C



def make_target_mat_online_adjoint_newcost(a_, var_target):

    as_target = var_target[:-1]
    z_target = var_target[-1:]
    C1_1 = a_[:-1] - as_target
    C2_1 = a_[-1:] - z_target

    C = np.concatenate((C1_1, C2_1))

    return C



# FRTO sPOD simplified functions
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
        [A11, np.zeros((A11.shape[0], Da.shape[1]))],
        [A21, np.zeros((A21.shape[0], Da.shape[1]))]
    ])

    return A



def Q11_red(M1):
    return M1


def Q12_red(N, D):
    return D.transpose() @ N.transpose()


def Q21_red(N, D):
    return N @ D


def Q22_red(M2, D):
    return D.transpose() @ (M2 @ D)


def B11_red(N, A1, z_dot, r):
    return A1 - N @ D_dash(z_dot, r)


def B12_red(M2, N, A2, psi, D, W, a_dot, z_dot, a_s, u, r):
    mat1 = dDT_dt(a_dot) @ N.transpose()
    mat3 = DT_dash(N.transpose() @ a_dot)
    mat4 = DT_dash(M2 @ (D @ z_dot))
    mat5 = D.transpose() @ (M2 @ D_dash(z_dot, r))
    mat6 = DT_dash(A2 @ a_s)
    mat7 = D.transpose() @ A2
    mat8 = DT_dash(W.transpose() @ (psi @ u))

    return mat1 - mat3 - mat4 - mat5 + mat6 + mat7 + mat8


def B21_red(Dfd, N, psi, V, a_dot, u):
    mat2 = N @ dD_dt(a_dot)
    mat6 = VT_dash(Dfd, V) @ (psi @ u)
    return mat2 + mat6


def B22_red(Dfd, M2, D, psi, W, a_dot, u):

    mat1 = dDT_dt(a_dot) @ (M2 @ D)
    mat3 = D.transpose() @ (M2 @ dD_dt(a_dot))
    mat7 = D.transpose() @ (WT_dash(Dfd, W) @ (psi @ u))

    return mat1 + mat3 + mat7


def Z11_red(M1):
    return M1


def Z12_red(N, D):
    return D.transpose() @ N.transpose()


def Z21_red(N, D):
    return N @ D


def Z22_red(M2, D):
    return D.transpose() @ (M2 @ D)


def make_LHS_mat_online_adjoint_red(LHS_matrix, Da):

    M1 = LHS_matrix[0]
    N = LHS_matrix[1]
    M2 = LHS_matrix[2]

    Q1_1_red = Q11_red(M1)
    Q1_2_red = Q12_red(N, Da)
    Q2_1_red = Q21_red(N, Da)
    Q2_2_red = Q22_red(M2, Da)

    LHS = np.block([
        [Q1_1_red.transpose(), Q1_2_red.transpose()],
        [Q2_1_red.transpose(), Q2_2_red.transpose()]
    ])

    return LHS



def make_RHS_mat_online_adjoint_red(RHS_matrix, LHS_matrix, psi, a_dot, z_dot, Da, a_, Dfd, Vdp, Wdp, u,
                                    intervalIdx, weight):
    r = len(a_) - 1
    a_s = np.atleast_2d(a_[:-1]).T
    z = np.atleast_2d(a_[-1:]).T
    a_dot = np.atleast_2d(a_dot).T
    z_dot = np.atleast_2d(z_dot).T
    u = np.atleast_2d(u).T

    V = weight * Vdp[intervalIdx] + (1 - weight) * Vdp[intervalIdx + 1]
    W = weight * Wdp[intervalIdx] + (1 - weight) * Wdp[intervalIdx + 1]

    N = LHS_matrix[1]
    M2 = LHS_matrix[2]
    A1 = RHS_matrix[0]
    A2 = RHS_matrix[1]

    B1_1_red = B11_red(N, A1, z_dot, r)
    B1_2_red = B12_red(M2, N, A2, psi, Da, W, a_dot, z_dot, a_s, u, r)
    B2_1_red = B21_red(Dfd, N, psi, V, a_dot, u)
    B2_2_red = B22_red(Dfd, M2, Da, psi, W, a_dot, u)

    RHS = np.block([
        [B1_1_red.transpose(), B1_2_red.transpose()],
        [B2_1_red.transpose(), B2_2_red.transpose()]
    ])

    return RHS


def check_invertability_red(LHS_matrix, a_):

    M1 = LHS_matrix[0]
    N = LHS_matrix[1]
    M2 = LHS_matrix[2]
    D = make_Da(a_)


    Z1_1_red = Z11_red(M1)
    Z1_2_red = Z12_red(N, D)
    Z2_1_red = Z21_red(N, D)
    Z2_2_red = Z22_red(M2, D)

    Z = np.block([
        [Z1_1_red.transpose(), Z1_2_red.transpose()],
        [Z2_1_red.transpose(), Z2_2_red.transpose()]
    ])

    if np.linalg.cond(Z) < 1 / sys.float_info.epsilon:
        return True
    else:
        return False

