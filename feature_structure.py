# -*- coding: utf-8 -*-
# @Time    : 11/16/2023 9:15 PM
# @Author  : WANG CC
# @Email   : wangchenchen233@163.com
# @File    : feature_structure.py
from scipy.sparse import *
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist, squareform
import numpy as np
import dcor

def feature_structure(X):
    A = similar_matrix(X.T, 10, 1).todense()
    A = np.array(A)
    return A, np.diag(A.sum(1)) - A


def feature_structure_intra_class(X, y, n_class):
    L_M = []
    M = []
    for i in range(n_class):
        A, LA = feature_structure(X[y == i + 1])
        L_M.append(LA)
        M.append(A)
    return L_M, M

def similar_matrix(x, k, t_c):
    """
    :param t_c: scale for para t
    :param x: N D
    :param k:
    :return:
    """
    # compute pairwise euclidean distances
    n_samples, n_features = x.shape
    D = pairwise_distances(x)
    D **= 2
    # sort the distance matrix D in ascending order
    dump = np.sort(D, axis=1)
    idx = np.argsort(D, axis=1)
    # 0值:沿着每一列索引值向下执行方法(axis=0代表往跨行)分别对每一列
    # 1值:沿着每一行(axis=1代表跨列) 分别对每一行
    idx_new = idx[:, 0:k + 1]
    dump_new = dump[:, 0:k + 1]
    # compute the pairwise heat kernel distances
    # t = np.percentile(D.flatten(), 20)  # 20210816 tkde13
    t = np.mean(D)
    t = t_c * t
    dump_heat_kernel = np.exp(-dump_new / (2 * t))
    G = np.zeros((n_samples * (k + 1), 3))
    G[:, 0] = np.tile(np.arange(n_samples), (k + 1, 1)).reshape(-1)  # 第一个参数为Y轴扩大倍数，第二个为X轴扩大倍数
    G[:, 1] = np.ravel(idx_new, order='F')  # 按列顺序重塑 n_samples*(k+1)
    G[:, 2] = np.ravel(dump_heat_kernel, order='F')
    # build the sparse affinity matrix W
    W = csc_matrix((G[:, 2], (G[:, 0], G[:, 1])), shape=(n_samples, n_samples))
    bigger = np.transpose(W) > W
    W = W - W.multiply(bigger) + np.transpose(W).multiply(bigger)
    # np.transpose(W).multiply(bigger)不等于np.multiply(W,bigger)
    return W

