import matplotlib.pyplot as plt
import numpy as np
from data_load import dataset_pro, construct_label_matrix
from feature_structure import feature_structure, feature_structure_intra_class
import DRFS

for dname in ['lung_discrete']:
    # load dataset
    X, y, n_Classes = dataset_pro(dname, 'scale')
    Y = construct_label_matrix(y)

    # calculate feature similarity matrix
    L_M_intra, M = feature_structure_intra_class(X, y, n_Classes)
    M_inter, _ = feature_structure(X)

    # the number of selected features
    fea_nums = 100

    # parameter setting
    para = (1, 1, 1)

    # Dual Regularized Feature Selection
    Weight, obj = DRFS.DRFS(X, Y, L_M_intra, M_inter, para)

    # plot objection value
    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(1, len(obj) + 1), obj, '-v', markersize=10, linewidth=2.0, color='goldenrod')
    plt.xlabel('Iterative Number')
    plt.ylabel('Objective Function Value')
    plt.show()
