import numpy as np
import itertools


def compute_Y_matrix(conductance_matrix):
    Y_matrix = -np.copy(conductance_matrix)  # Modified conductance matrix, convenient for computing OPF
    for i in range(conductance_matrix.shape[0]):
        Y_matrix[i, i] = np.sum(conductance_matrix[i])
        return Y_matrix


def dict_to_matrix(d, *indices):
    res = np.empty(list(len(inds) for inds in indices))
    for ind in itertools.product(*indices):
        if callable(d[ind]):
            res[ind] = d[ind]()
        else:
            res[ind] = d[ind]
    return res
