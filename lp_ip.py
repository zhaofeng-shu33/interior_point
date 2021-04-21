# LP solver using barrier method (interior point)
import numpy as np

def newton_opt_single(A, b, c, y, t):
    # compute delta_y
    d = 1 / (A.T @ y - c)
    Fy = t * b + A @ d
    n_JF_y = A @ np.diag(d ** 2) @ A.T # negative Jacobian
    return np.linalg.solve(n_JF_y, Fy)

def newton_opt(A, b, c, y, t, eps):
    delta_y = 1
    y_clone = y.copy()
    while np.linalg.norm(delta_y) >= eps:
        delta_y = newton_opt_single(A, b, c, y_clone, t)
        y_clone += delta_y
    return y_clone

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

