from Helper import *


# This function is totally problem / setup dependent. We need to update all the aspects of this function accordingly
def Shifts_1D(SnapShotMatrix, X, t):
    Nx = int(np.size(X))
    Nt = int(np.size(t))
    NumComovingFrames = 1
    delta = np.zeros((NumComovingFrames, Nt), dtype=float)

    FlameFrontLeftPos = np.zeros(Nt, dtype=float)
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
    mu0 = N * M / (4 * np.sum(np.abs(qmat))) * 0.5
    lambd0 = 1 / np.sqrt(np.maximum(M, N)) * 1
    ret = shifted_rPCA(qmat, trafos, nmodes_max=60, eps=1e-5, Niter=spod_iter, use_rSVD=True, mu=mu0, lambd=lambd0,
                       dtol=1e-4)

    # Extract frames modes and error
    qframes, qtilde, rel_err = ret.frames, ret.data_approx, ret.rel_err_hist
    modes_list = [qframes[0].Nmodes]
    V = [qframes[0].modal_system["U"]]

    qframes = [qframes[0].build_field()]

    return qtilde, modes_list[0], V[0], qframes


# Offline phase general functions for primal system
def central_FDMatrix(order, Nx, dx):
    from scipy.sparse import spdiags
    from jax.experimental import sparse

    # column vectors of all ones
    enm2 = jnp.ones(Nx - 2)
    enm4 = jnp.ones(Nx - 4)
    enm6 = jnp.ones(Nx - 6)

    # column vectors of all zeros
    z4 = jnp.zeros(4)
    z6 = jnp.zeros(6)
    znm2 = jnp.zeros_like(enm2)

    # determine the diagonal entries 'diagonals_D' and the corresponding
    # diagonal indices 'indices' based on the specified order
    if order == 2:
        pass
    elif order == 4:
        pass
    elif order == 6:
        diag3 = jnp.hstack([-enm6, z6])
        diag2 = jnp.hstack([5, 9 * enm6, 5, z4])
        diag1 = jnp.hstack([-30, -40, -45 * enm6, -40, -30, -60, 0])
        diag0 = jnp.hstack([-60, znm2, 60])
        diagonals_D = (1 / 60) * jnp.array([diag3, diag2, diag1, diag0,
                                            -jnp.flipud(diag1), -jnp.flipud(diag2), -jnp.flipud(diag3)])
        indices = [-3, -2, -1, 0, 1, 2, 3]
    else:
        print("Order of accuracy %i is not supported.", order)
        exit()

    # assemble the output matrix
    D = sparse.BCOO.fromdense(spdiags(diagonals_D, indices, format="csr").todense())

    return D * (1 / dx)


def subsample(X, num_sample):
    active_subspace_factor = 1

    # sampling points for the shifts (The shift values can range from 0 to X/2 and then is a mirror image for X/2 to X)
    delta_samples = jnp.linspace(0, X[-1], num_sample)

    delta_sampled = [active_subspace_factor * delta_samples,
                     jnp.zeros_like(delta_samples),
                     delta_samples]

    return delta_sampled


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

    return [trafo_1.shifts_pos], [trafo_1]


def make_V_W_delta(U, T_delta, X, num_sample):
    V_delta = []
    W_delta = []
    Nx = len(X)
    dx = X[1] - X[0]

    D = central_FDMatrix(order=6, Nx=Nx, dx=dx)
    for it in range(num_sample):
        V11 = T_delta[0][it] @ U
        V_delta.append(V11)

        W11 = D @ (T_delta[0][it] @ U)
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

    return jnp.array(LHS_mat)


def make_RHS_mat_offline_primal(V_delta, W_delta, A):
    RHS_mat = []

    for it in range(len(V_delta)):
        A_1 = (V_delta[it].transpose() @ A) @ V_delta[it]
        A_2 = (W_delta[it].transpose() @ A) @ V_delta[it]

        RHS_mat.append([A_1, A_2])

    return jnp.array(RHS_mat)


def make_control_mat_offline_primal(V_delta, W_delta, psi):
    C_mat = []

    for it in range(len(V_delta)):
        C_1 = (V_delta[it].transpose() @ psi)
        C_2 = (W_delta[it].transpose() @ psi)

        C_mat.append([C_1, C_2])

    return jnp.array(C_mat)


def make_control_update_mat(V_delta, W_delta, psi):
    Ct_mat = []

    for it in range(len(V_delta)):
        Ct_1 = psi.transpose() @ V_delta[it]
        Ct_2 = psi.transpose() @ W_delta[it]

        Ct_mat.append([Ct_1, Ct_2])

    return jnp.array(Ct_mat)


# Online phase functions
def findIntervalAndGiveInterpolationWeight_1D(xPoints, xStar):
    intervalBool_arr = jnp.where(xStar >= xPoints, 1, 0)
    mixed = intervalBool_arr[:-1] * (1 - intervalBool_arr)[1:]
    index = jnp.sum(mixed * jnp.arange(0, mixed.shape[0]))

    intervalIdx = index
    alpha = (xPoints.at[intervalIdx + 1].get() - xStar) / (
                xPoints.at[intervalIdx + 1].get() - xPoints.at[intervalIdx].get())

    return intervalIdx, alpha


def make_Da(a):
    D_a = a[:len(a) - 1]

    return D_a


def make_LHS_mat_online_primal(LHS_matrix, Da, intervalIdx, weight):
    M11 = weight * LHS_matrix[intervalIdx][0] + (1 - weight) * LHS_matrix[intervalIdx + 1][0]
    M12 = (weight * LHS_matrix[intervalIdx][1] + (1 - weight) * LHS_matrix[intervalIdx + 1][1]) @ Da[:, jnp.newaxis]
    M21 = M12.transpose()
    M22 = (Da[:, jnp.newaxis].transpose() @ (weight * LHS_matrix[intervalIdx][2] +
                                              (1 - weight) * LHS_matrix[intervalIdx + 1][2])) @ Da[:, jnp.newaxis]
    M = jnp.block([
        [M11, M12],
        [M21, M22]
    ])

    return M


def make_RHS_mat_online_primal(RHS_matrix, Da, intervalIdx, weight):

    A11 = weight * RHS_matrix[intervalIdx][0] + (1 - weight) * RHS_matrix[intervalIdx + 1][0]
    A21 = Da[:, jnp.newaxis].transpose() @ (weight * RHS_matrix[intervalIdx][1] +
                             (1 - weight) * RHS_matrix[intervalIdx + 1][1])
    A = jnp.block([
        [A11, jnp.zeros((A11.shape[0], 1))],
        [A21, jnp.zeros((A21.shape[0], 1))]
    ])

    return A


def make_control_mat_online_primal(f, C, Da, intervalIdx, weight):
    C1 = (weight * C[intervalIdx][0] + (1 - weight) * C[intervalIdx + 1][0]) @ f
    C2 = Da[:, jnp.newaxis].transpose() @ ((weight * C[intervalIdx][1] + (1 - weight) * C[intervalIdx + 1][1]) @ f)

    C = jnp.concatenate((C1, C2))

    return C


def get_online_state(T_trafo, V, a, X, t):
    Nx = len(X)
    Nt = len(t)
    qs_online = jnp.zeros((Nx, Nt))
    q = V @ a

    qs_online += T_trafo[0].apply(q)

    return qs_online


# Auxiliary functions
def findIntervals(delta_s, delta):
    Nt = len(delta)
    intIds = []
    weights = []
    for i in range(Nt):
        intervalIdx, weight = findIntervalAndGiveInterpolationWeight_1D(delta_s, delta[i])
        intIds.append(intervalIdx)
        weights.append(weight)

    return intIds, weights
