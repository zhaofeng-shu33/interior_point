import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse import identity
from scipy.sparse import hstack, vstack
import scipy.linalg

def construct_sparse_A(n):
    row = np.array(list(range(0, n)))
    col = np.array([i * n + i for i in range(0, n)])
    data = np.ones(n)
    return coo_matrix((data, (row, col)), shape=(n, n * n))

def construct_dense_A(n):
    A = np.array([n, n ** 2])
    for i in range(0, n):
        A[i, i * n + i] = 1
    return A

def construct_diff_F(S, X):
    '''
    S, X: dense matrix with dimension n times n
    '''
    n = S.shape[0]
    A = construct_sparse_A(n)
    _id = np.eye(n)
    S_id = coo_matrix(np.kron(S, _id))
    id_X = coo_matrix(np.kron(_id, X))
    sparse_id = identity(n * n, format='csr')
    v1 = hstack((coo_matrix((n*n, n*n)), A.T, sparse_id))
    v2 = hstack((A, coo_matrix((n, n + n ** 2))))
    v3 = hstack((S_id, coo_matrix((n*n, n)), id_X))
    return vstack((v1, v2, v3)).tocsr()

def get_maximal_step_length(X, delta_X):
    vals, Q = np.linalg.eigh(X)
    L_inv = np.diag(1 / np.sqrt(vals)) @ Q.T
    target_matrix = -1 * L_inv @ delta_X @ L_inv.T
    n = Q.shape[0]
    lambda_max = scipy.linalg.eigh(target_matrix, eigvals_only=True, eigvals=[n-1, n-1])[0]
    return 1 / lambda_max

def lp_ip_pd(C, eps=1e-3, mu=0.5, max_iter=40):
    '''
    primal dual method for solving min C * X, s.t. X_{ii} = 1, for i=1,2,\dots, n

    use dense matrix operation

    Returns
    -------
    '''
    tau = 0.99
    n = C.shape[0]
    m = n
    b = np.ones(n)
    X = np.eye(n)
    S = np.eye(n)
    y = np.zeros(m)
    _id = np.eye(n)
    n_square = n ** 2
    F_diff = np.zeros([2 * n_square + m, 2 * n_square + m])
    A = construct_dense_A(n)
    F_diff[:n_square, n_square:(m + n_square)] = A.T
    F_diff[n_square:(m + n_square), :n_square] = A
    F_diff[:n_square, (m + n_square): (2 * n_square + m)] = np.eye(n_square)
    err = 1
    iter_cnt = 0
    while err > eps and iter_cnt < max_iter:
        iter_cnt += 1
        sigma = mu * np.sum(X * S)  / n
        F_1 = (np.diag(y) + S - C).T.reshape(-1)
        F_2 = np.diag(X) - b
        F_3 = (X @ S - sigma * np.eye(n)).T.reshape(-1)
        F = np.concatenate((F_1, F_2, F_3))
        err = sigma + np.linalg.norm(F)
        if err < eps:
            break
        S_id = np.kron(S, _id)
        id_X = np.kron(_id, X)        
        F_diff[(m + n_square):,:n_square] = S_id
        F_diff[(m + n_square):,(m + n_square):] = id_X
        packed_sol = np.linalg.solve(F_diff, -F)
        # unpack the value

        # always update by unit length
        delta_X = packed_sol[:n_square].reshape(n, n) # omit the transpose supposing symmetric matrix
        delta_X = (delta_X + delta_X.T) / 2
        delta_S = packed_sol[(m + n_square):].reshape(n, n)
        alpha = get_maximal_step_length(X, delta_X)
        beta = get_maximal_step_length(S, delta_S)
        v = np.min([1, tau * alpha, tau * beta])
        X += v * delta_X
        y += v * packed_sol[n_square:(m + n_square)]
        S += v * delta_S
    return (y, np.dot(b, y))

if __name__ == '__main__':
    n = 4
    S = np.eye(4)
    X = np.eye(4)
    diff_F = construct_diff_F(S, X)
    print(type(diff_F))