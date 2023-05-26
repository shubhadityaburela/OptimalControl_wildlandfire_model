import numpy as np


def Calc_Cost(q, q_target, f, lamda):
    return (np.linalg.norm(q - q_target)) ** 2 / 2 + (lamda / 2) * (np.linalg.norm(f)) ** 2


def Calc_Grad(lamda, sigma, f, qs_adj):
    return lamda * f + sigma * qs_adj


def Update_Control(f, omega, lamda, sigma, q0_init, qs, qs_target, J_prev, max_Armijo_iter=5,
                   base_model_class=None, delta=0.5):

    print("Armijo iterations.........")
    for k in range(max_Armijo_iter):
        f_new = f - omega * Calc_Grad(lamda, sigma, f)

        # Solve the primal equation
        qs = base_model_class.TimeIntegration_primal(q0_init, np.concatenate((f_new, np.zeros_like(f_new)), axis=0),
                                                     sigma_s)
        if np.isnan(qs).any() and k < max_Armijo_iter - 1:
            print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
            omega = omega / 4
        elif np.isnan(qs).any() and k == max_Armijo_iter - 1:
            print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
            exit()
        else:
            qs = qs[:int(qs.shape[0]) // base_model_class.NumConservedVar, :]
            J = Calc_Cost(qs, qs_target_s, f_new, lamda)
            if J < J_prev - delta * omega * np.linalg.norm(f_new) ** 2:
                J_opt = J
                f_opt = np.concatenate((f_new, np.zeros_like(f_new)))
                print(f"Armijo iteration converged after {k + 1} steps")
                return f_opt
            elif J >= J_prev or np.isnan(J):
                omega = omega / 4