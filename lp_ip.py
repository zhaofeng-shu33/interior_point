# LP solver using barrier method (interior point)
import numpy as np
from scipy.optimize import minimize

def newton_opt_single(A, b, c, y, t):
    # compute delta_y
    d = 1 / (A.T @ y - c)
    Fy = t * b + A @ d
    n_JF_y = A @ np.diag(d ** 2) @ A.T # negative Jacobian
    return np.linalg.solve(n_JF_y, Fy)

def Hessian(A, b, c, y, t):
    d = 1 / (A.T @ y - c)
    return A @ np.diag(d ** 2) @ A.T # negative Jacobian

def Jacobian(A, b, c, y, t):
    d = 1 / (A.T @ y - c)
    return -(t * b + A @ d)

def Object_Function(A, b, c, y, t):
    if np.any(c - A.T @ y <= 0):
        return np.infty
    val = t * np.dot(b, y) + np.sum(np.log(c - A.T @ y))
    return -1.0 * val

def newton_opt(A, b, c, y, t, eps):
    _object_function = lambda x: Object_Function(A, b, c, x, t)
    _jacobian = lambda x: Jacobian(A, b, c, x, t)
    _hessian = lambda x: Hessian(A, b, c, x, t)

    res = minimize(_object_function, y, method='Newton-CG', jac=_jacobian,
        hess=_hessian, options={'xtol': eps})
    return res.x

def lp_ip(A, b, c, y0=None, eps=1e-4, mu=2):
    '''
    Returns
    -------
    argmax_y: array-like
    max_y: float
    '''
    m, n = A.shape
    if y0 is None: # stage one
        b_prime = np.zeros([m + 1])
        b_prime[-1] = 1
        y0 = b_prime.copy()
        y0[-1] = 1 + np.max(c)
        A_prime = np.hstack([A.T, np.ones(m)])
        y0, _ = lp_ip(A_prime, b_prime, c, y0=y0, eps=1e-2)
    else:
        t = mu
        y = y0
        while True:
            y = newton_opt(A, b, c, y, t, eps=eps)
            if n / t < eps:
                break
            t *= mu
    return (y, np.dot(b, y))

