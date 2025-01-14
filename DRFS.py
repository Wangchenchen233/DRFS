import math
import numpy as np

eps = np.spacing(1)


def DRFS(X, y, L_M_intras, M_inter, para):
    # L_M_intras: class-specific feature similarity matrices
    # M_inter: Global feature similarity matrix
    # para = (alpha,beta,gamma) alpha: class-specific; gamma: global; beta: L_21-norm.
    n_smp, n_fea = X.shape
    n_cls = y.shape[1]
    alpha, beta, gamma = para
    cls2 = n_cls * (n_cls - 1)

    XX = np.dot(X.T, X)
    XY = np.dot(X.T, y)
    W = np.ones((n_fea, n_cls)) / n_cls

    D = np.identity(n_fea)
    I = np.identity(n_fea)
    Q = np.identity(n_cls)
    Z = W

    mu = 1
    C = np.zeros_like(W)
    rho = 1.1
    mu_max = 1e8
    max_iter = 50
    obj = np.zeros(max_iter)
    for iter_ in range(max_iter):
        # update W
        W_old = W
        W_up = np.dot(XY, Q) + mu * Z - C
        W_down = XX + beta * D + mu * I

        W_upp = np.maximum(0, W_up)
        W_upn = -np.minimum(0, W_up)
        W_downp = np.maximum(0, W_down)
        W_downn = -np.minimum(0, W_down)

        W_temp = np.divide(W_downn @ W + W_upp, W_downp @ W + W_upn)
        W = W * np.array(W_temp)

        # update D
        temp = np.sqrt((W * W).sum(1))
        temp[temp < 1e-16] = 1e-16
        temp = 0.5 / temp
        D = np.diag(temp)

        # update Z
        Z_old = Z
        for k in range(n_cls):
            w_k = W[:, k]
            sum_A_z_i = sum([M_inter @ Z[:, i] for i in range(n_cls) if i != k]) / cls2
            difference = mu * w_k + C[:, k] - gamma / cls2 * sum_A_z_i
            Z_inv_k = np.linalg.inv(alpha / n_cls * L_M_intras[k] + mu * I)
            Z[:, k] = np.dot(Z_inv_k, difference)
            Z[Z < 1e-16] = 1e-16

        # update mu, C
        mu = min(rho * mu, mu_max)
        C = C + mu * (W - Z)

        # calculate objective function
        obj[iter_] = np.linalg.norm(X @ W @ Q - y, 'fro') ** 2 + \
                     beta * calculate_l21_norm(W) + \
                     mu * np.linalg.norm(W - Z + C / mu, 'fro') ** 2 + \
                     calculate_w_reg(Z, L_M_intras, M_inter, alpha / n_cls, gamma / cls2)

        if iter_ >= 1 and max(np.linalg.norm(W_old - W, ord=np.inf),
                              np.linalg.norm(Z_old - Z, ord=np.inf)) < 1e-3 and np.linalg.norm(W - Z,
                                                                                               ord=np.inf) < 1e-4:
            break

        if iter_ >= 1 and (math.fabs(obj[iter_] - obj[iter_ - 1])) / math.fabs(obj[iter_ - 1]) < 1e-3:
            break
    return W, obj


def calculate_l21_norm(X):
    return (np.sqrt(np.multiply(X, X).sum(1))).sum()


def calculate_w_reg(W, L_M_intra, M_inter, alpha, beta):
    first_term = alpha * np.sum([np.dot(W[:, k].T, np.dot(L_M_intra[k], W[:, k])) for k in range(W.shape[1])])
    second_term = beta * np.sum(
        [np.dot(W[:, p].T, np.dot(M_inter, W[:, q])) for p in range(W.shape[1]) for q in range(W.shape[1]) if p != q])
    result = first_term + second_term
    return result
