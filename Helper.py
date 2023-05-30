import numpy as np
import jax.numpy as jnp

import jax


class Cost_functional:
    def __init__(self, qs_target, sigma, qs_0, lamda, wf):

        self.qs_target = qs_target.reshape((-1))
        self.sigma = sigma
        self.qs_0 = qs_0
        self.lamda = lamda
        self.wf = wf

    def C_JAX(self, f):

        qs = self.wf.TimeIntegration(self.qs_0, f, self.sigma)

        qs = qs.reshape((-1))
        f = f.reshape((-1))

        return (qs - self.qs_target).T @ (qs - self.qs_target) / 2 + (self.lamda / 2) * f.T @ f

    def C_JAX_cost(self, qs, f):
        qs = qs.reshape((-1))
        f = f.reshape((-1))
        return (qs - self.qs_target).T @ (qs - self.qs_target) / 2 + (self.lamda / 2) * f.T @ f

    def C_JAX_update(self, f, omega, dJ_dx, J_prev, max_Armijo_iter=5, delta=0.5):

        print("Armijo iterations.........")
        for k in range(max_Armijo_iter):
            f_new = f - omega * dJ_dx

            print(jnp.linalg.norm(f_new-f))

            # Solve the primal equation
            qs = self.wf.TimeIntegration(self.qs_0, f_new, self.sigma)
            if jnp.isnan(qs).any() and k < max_Armijo_iter - 1:
                print(f"Warning!!! step size omega = {omega} too large!", f"Reducing the step size at iter={k + 1}")
                omega = omega / 4
            elif jnp.isnan(qs).any() and k == max_Armijo_iter - 1:
                print("With the given Armijo iterations the procedure did not converge. Increase the max_Armijo_iter")
                exit()
            else:
                J = self.C_JAX_cost(qs, f_new)
                if J < J_prev - delta * omega * f_new.T @ f_new:
                    J_opt = J
                    f_opt = f_new
                    print(f"Armijo iteration converged after {k + 1} steps")
                    return f_opt, J_opt
                elif J >= J_prev or np.isnan(J):
                    omega = omega / 4
