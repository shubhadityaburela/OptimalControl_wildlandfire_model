from Costs import *
from Grads import *
from Helper import *
from time import perf_counter
from memory_profiler import profile


def Update_Control(f, q0, qs_adj, qs_target, mask, A_p, J_prev, omega, lamda, max_Armijo_iter,
                   wf, delta, ti_method, verbose, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5

    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad(lamda, mask, f, qs_adj)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Calc_Grad t_cpu = %1.6f" % time_odeint)
    for k in range(max_Armijo_iter):
        f_new = f - omega * dL_du

        # Solve the primal equation
        qs = wf.TimeIntegration_primal(q0, f_new, A_p, mask, ti_method=ti_method)

        if np.isnan(qs).any() and k < max_Armijo_iter - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / 4
        elif np.isnan(qs).any() and k == max_Armijo_iter - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost(qs, qs_target, f_new, lamda, **kwargs)
            dJ = J_prev - delta * omega * np.linalg.norm(dL_du) ** 2
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, np.linalg.norm(dL_du)
            elif J >= dJ or np.isnan(J):
                if k == max_Armijo_iter - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, np.linalg.norm(dL_du)
                else:
                    if J == dJ:
                        if verbose: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                              f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 4
                        count = count + 1
                        if count > itr:
                            J_opt = J
                            f_opt = f_new
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, np.linalg.norm(dL_du)
                    else:
                        if verbose: print(f"No NANs found but step size omega = {omega} too large!",
                              f"Reducing omega at iter={k + 1}")
                        omega = omega / 4



def Update_Control_PODG(f, a0_primal, as_adj, qs_target, V, Ar_p, psir_p, psi, J_prev, omega, lamda,
                        max_Armijo_iter, wf, delta, ti_method, verbose, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5

    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad_PODG(lamda, psir_p, f, as_adj)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Calc_Grad t_cpu = %1.6f" % time_odeint)
    for k in range(max_Armijo_iter):
        f_new = f - omega * dL_du

        # Solve the primal equation
        as_ = wf.TimeIntegration_primal_PODG(a0_primal, f_new, Ar_p, psir_p, ti_method)

        if np.isnan(as_).any() and k < max_Armijo_iter - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / 4
        elif np.isnan(as_).any() and k == max_Armijo_iter - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost_PODG(V, as_, qs_target, f_new, psi, lamda, **kwargs)
            dJ = J_prev - delta * omega * np.linalg.norm(dL_du) ** 2
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, np.linalg.norm(dL_du), 0, False
            elif J >= dJ or np.isnan(J):
                if k == max_Armijo_iter - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, np.linalg.norm(dL_du), 0, True
                else:
                    if J == dJ:
                        if verbose: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                              f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 4
                        count = count + 1
                        if count > itr:
                            J_opt = J
                            f_opt = f_new
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, np.linalg.norm(dL_du), 0, True
                    else:
                        if verbose: print(f"No NANs found but step size omega = {omega} too large!",
                              f"Reducing omega at iter={k + 1}")
                        omega = omega / 4


def Update_Control_PODG_newcost(f, a0_primal, as_adj, Tarr_a, V, Ar_p, psir_p, psi, J_prev, omega, lamda,
                                max_Armijo_iter, wf, delta, ti_method, verbose, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5

    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad_PODG(lamda, psir_p, f, as_adj)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Calc_Grad t_cpu = %1.6f" % time_odeint)
    for k in range(max_Armijo_iter):
        f_new = f - omega * dL_du

        # Solve the primal equation
        as_ = wf.TimeIntegration_primal_PODG(a0_primal, f_new, Ar_p, psir_p, ti_method)

        if np.isnan(as_).any() and k < max_Armijo_iter - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / 4
        elif np.isnan(as_).any() and k == max_Armijo_iter - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost_PODG_new(as_, Tarr_a, f_new, psi, lamda, **kwargs)
            dJ = J_prev - delta * omega * np.linalg.norm(dL_du) ** 2
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, np.linalg.norm(dL_du), 0, False
            elif J >= dJ or np.isnan(J):
                if k == max_Armijo_iter - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, np.linalg.norm(dL_du), 0, False
                else:
                    if J == dJ:
                        if verbose: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                              f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 4
                        count = count + 1
                        if count > itr:
                            J_opt = J
                            f_opt = f_new
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, np.linalg.norm(dL_du), 0, True
                    else:
                        if verbose: print(f"No NANs found but step size omega = {omega} too large!",
                              f"Reducing omega at iter={k + 1}")
                        omega = omega / 4


def Update_Control_sPODG(f, lhs, rhs, c, Vd_p, a0_primal, as_, as_adj, qs_target, delta_s, J_prev, psi, intIds, weights,
                         omega, lamda, max_Armijo_iter, wf, delta, ti_method, verbose, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5

    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad_sPODG(lamda, f, c, intIds, weights, as_, as_adj, **kwargs)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Calc_Grad t_cpu = %1.6f" % time_odeint)
    for k in range(max_Armijo_iter):
        f_new = f - omega * dL_du

        # Solve the primal equation
        as_, _ = wf.TimeIntegration_primal_sPODG(lhs, rhs, c, a0_primal, f_new, delta_s, ti_method)

        if np.isnan(as_).any() and k < max_Armijo_iter - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / 4
        elif np.isnan(as_).any() and k == max_Armijo_iter - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost_sPODG(Vd_p, as_, qs_target, f_new, psi, lamda, intIds, weights, **kwargs)
            dJ = J_prev - delta * omega * np.linalg.norm(dL_du) ** 2
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, np.linalg.norm(dL_du), 0, False
            elif J >= dJ or np.isnan(J):
                if k == max_Armijo_iter - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, np.linalg.norm(dL_du), 1, False
                else:
                    if J == dJ:
                        if verbose: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                              f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 4
                        count = count + 1
                        if count > itr:
                            J_opt = J
                            f_opt = f_new
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, np.linalg.norm(dL_du), 0, True
                    else:
                        if verbose: print(f"No NANs found but step size omega = {omega} too large!",
                              f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 4



def Update_Control_sPODG_red(f, lhs, rhs, c, Vd_p, a0_primal, as_, as_adj, qs_target, delta_s, J_prev, psi, intIds, weights,
                         omega, lamda, max_Armijo_iter, wf, delta, ti_method, verbose, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5

    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad_sPODG(lamda, f, c, intIds, weights, as_, as_adj, **kwargs)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Calc_Grad t_cpu = %1.6f" % time_odeint)
    for k in range(max_Armijo_iter):
        f_new = f - omega * dL_du

        # Solve the primal equation
        as_, _ = wf.TimeIntegration_primal_sPODG_red(lhs, rhs, c, a0_primal, f_new, delta_s, ti_method)

        if np.isnan(as_).any() and k < max_Armijo_iter - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / 4
        elif np.isnan(as_).any() and k == max_Armijo_iter - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost_sPODG(Vd_p, as_, qs_target, f_new, psi, lamda, intIds, weights, **kwargs)
            dJ = J_prev - delta * omega * np.linalg.norm(dL_du) ** 2
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, np.linalg.norm(dL_du), 0, False
            elif J >= dJ or np.isnan(J):
                if k == max_Armijo_iter - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, np.linalg.norm(dL_du), 0, False
                else:
                    if J == dJ:
                        if verbose: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                              f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 4
                        count = count + 1
                        if count > itr:
                            J_opt = J
                            f_opt = f_new
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, np.linalg.norm(dL_du), 0, True
                    else:
                        if verbose: print(f"No NANs found but step size omega = {omega} too large!",
                              f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 4



def Update_Control_sPODG_newcost_red(f, lhs, rhs, c, a0_primal, as_, as_adj, as_target, z_target, delta_s, J_prev,
                                     intIds, weights, omega, lamda, max_Armijo_iter, wf, delta, ti_method,
                                     verbose, **kwargs):
    print("Armijo iterations.........")
    count = 0
    itr = 5

    time_odeint = perf_counter()  # save timing
    dL_du = Calc_Grad_sPODG(lamda, f, c, intIds, weights, as_, as_adj, **kwargs)
    time_odeint = perf_counter() - time_odeint
    if verbose: print("Calc_Grad t_cpu = %1.6f" % time_odeint)
    for k in range(max_Armijo_iter):
        f_new = f - omega * dL_du

        # Solve the primal equation
        as_, _ = wf.TimeIntegration_primal_sPODG_red(lhs, rhs, c, a0_primal, f_new, delta_s, ti_method)

        if np.isnan(as_).any() and k < max_Armijo_iter - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / 4
        elif np.isnan(as_).any() and k == max_Armijo_iter - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            J = Calc_Cost_sPODG_new(as_, as_target, z_target, f_new, lamda, **kwargs)
            dJ = J_prev - delta * omega * np.linalg.norm(dL_du) ** 2
            if J < dJ:
                J_opt = J
                f_opt = f_new
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt, J_opt, np.linalg.norm(dL_du), 0, False
            elif J >= dJ or np.isnan(J):
                if k == max_Armijo_iter - 1:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration reached maximum limit thus exiting the Armijo loop......")
                    return f_opt, J_opt, np.linalg.norm(dL_du), 0, False
                else:
                    if J == dJ:
                        if verbose: print(f"J has started to saturate now so we reduce the omega = {omega}!",
                              f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 4
                        count = count + 1
                        if count > itr:
                            J_opt = J
                            f_opt = f_new
                            print(
                                f"Armijo iteration reached a point where J does not change thus exiting the Armijo loop......")
                            return f_opt, J_opt, np.linalg.norm(dL_du), 0, True
                    else:
                        if verbose: print(f"No NANs found but step size omega = {omega} too large!",
                              f"Reducing omega at iter={k + 1}, with J={J}")
                        omega = omega / 4