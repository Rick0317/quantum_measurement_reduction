import numpy as np
import sympy as sp
from openfermion import FermionOperator


def symmetric_tensor_array(name, n):
    symmetric_tensor = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if (i, j, k, l) <= (l, k, j, i):
                        symmetric_tensor.append(sp.Symbol(f"{name}_{i}{j}{k}{l}"))

    print("Length of Symmetric Tensor", len(symmetric_tensor))

    return sp.Matrix(symmetric_tensor)


def symmetric_tensor(T, n):
    """
    What symmetry do we apply?
    :param T:
    :param n:
    :return:
    """

    tensor = np.zeros((n, n, n, n), dtype=object)
    idx = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if (i, j, k, l) <= (l, k, j, i):
                        tensor[i, j, k, l] = T[idx]
                        tensor[l, k, j, i] = T[idx]
                        idx += 1

    return tensor


def tensor_to_ferm_op(tensor, n):
    ferm_op = FermionOperator()

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if (i, j, k, l) <= (l, k, j, i):
                        ferm_op += FermionOperator(f"{i}^ {j} {k}^ {l}",
                                                   tensor[i, j, k, l])
                        ferm_op += FermionOperator(f"{l}^ {k} {j}^ {i}",
                                                   tensor[l, k, j, i])

    return ferm_op
