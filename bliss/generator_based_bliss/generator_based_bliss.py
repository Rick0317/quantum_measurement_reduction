from openfermion import (
FermionOperator
)
import numpy as np
import sympy as sp
from bliss.majorana.custom_majorana_transform import get_custom_majorana_operator
from typing import List
import random


def params_to_tensor_op(params, n):
    """
    Given the parameters array, construct the coefficients tensor.
    This construct the xi_{ijkl} part in the Killer.
    :param params: Parameter array
    :param n: The number of sites
    :return: Fermionic Operator representation of xi_{ijkl} F_{ijkl}
    """
    tensor = np.zeros((n, n, n, n))
    idx = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if (i, j, k, l) <= (l, k, j, i):
                        tensor[i, j, k, l] = params[idx]
                        tensor[l, k, j, i] = params[idx]
                        idx += 1

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


def construct_HF_BLISS(H, params, N, Ne, j, b):
    total_number_operator = FermionOperator()
    for mode in range(N):
        total_number_operator += FermionOperator(((mode, 1), (mode, 0)))

    result = H
    t = params[1:1+int(N * (N + 1) // 2) ** 2]

    t_ferm = params_to_tensor_op(t, N)

    virtual_orbitals = [p for p in range(N - Ne) if p != j]
    occupied_orbitals = [p for p in range(N - Ne, N) if p != b]

    for empty in virtual_orbitals:
        result -= t_ferm * (FermionOperator(((empty, 1), (empty, 0))))

    for occupied in occupied_orbitals:
        result -= t_ferm * (FermionOperator(((occupied, 1), (occupied, 0))) - 1)

    result -= t_ferm * (total_number_operator - Ne)

    return result


def optimize_non_overlap_BLISS(H, N, Ne, idx_lists, j, b):
    """
    Get the analytical form of the 1-Norm with respect to the coefficients
    of the killer terms.
    :param H: Hamiltonian
    :param N: Number of orbitals
    :param Ne: Number of electrons
    :param idx_lists: The list of lists of indices that will contribute to
    the killer for each index.
    :return:
    """
    one_norm_func, one_norm_expr = generate_analytical_one_norm_U_based(H, N, Ne, idx_lists, j, b)
    def optimization_wrapper(params):
        t_vals = []
        prev_index = 0
        for i in range(N - 2):
            ind_length = len(idx_lists[i])
            t_vals.extend(params[prev_index: prev_index + ind_length])
            prev_index += ind_length
        return one_norm_func(*t_vals)

    t_val = construct_symmetric_tensor_specific(idx_lists)
    initial_guess = t_val

    return optimization_wrapper, initial_guess


def generate_analytical_one_norm_U_based(ferm_op, N, Ne, idx_lists, j, b):
    """
    Generate the analytical one-norm of a parameterized Majorana operator.
    This is H - K. We have to choose what killers we want to use.
    :param ferm_op: The input Hamiltonian H or any operator O
    :param N: Number of sites
    :param ne: Number of electrons
    :param idx_lists: The list of indices that will contribute
    """
    T_tensor_list = []
    lamda_variables = []

    sites_list = [p for p in range(N) if p != j and p != b]

    for i in range(len(sites_list)):
        index = sites_list[i]
        T = symmetric_tensor_array_specific(f'T{index}', N, idx_lists[i])
        lamda_variables.append(T)
        T_tensor = symmetric_tensor_from_triangle_specific(T, N, idx_lists[i])

        T_tensor_list.append(T_tensor)

    majorana_terms = construct_majorana_terms_non_overlap(ferm_op, N, Ne, T_tensor_list, j, b)

    # Compute the symbolic one-norm
    one_norm_expr = sum(
        sp.Abs(coeff) for term, coeff in majorana_terms.terms.items() if
        term != ())

    flat_vars = [var for sublist in lamda_variables for var in sublist]
    one_norm_func = sp.lambdify(flat_vars, one_norm_expr,
                                modules=['numpy'])

    print("Analytical 1-Norm Complete")
    return one_norm_func, one_norm_expr


def symmetric_tensor_array_specific(name, n, candidate_idx):
    """
    idx_list is the list of indices corresponding to the parameters that will
    contribute to the 1-Norm reduction.

    This function constructs the Symbolic representation of the Hamiltonian.
    :param name:
    :param n:
    :param idx_list:
    :return:
    """
    symmetric_tensor = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if (i, j, k, l) in candidate_idx:
                        symmetric_tensor.append(sp.Symbol(f"{name}_{i}{j}{k}{l}"))

    print("Length of Symmetric Tensor", len(symmetric_tensor))
    print("Predicted: ", len(candidate_idx))
    return sp.Matrix(symmetric_tensor)


def symmetric_tensor_from_triangle_specific(T, n, candidate_idx):
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
                    if (i, j, k, l) in candidate_idx:
                        tensor[i, j, k, l] = T[idx]
                        tensor[l, k, j, i] = T[idx]
                        idx += 1

    return tensor


def construct_majorana_terms_non_overlap(ferm_op, N, Ne, T_list: List, j, b):
    """
    Constructs the Majorana operators using the Hamiltonian operator.
    H - K
    :param ferm_op: Hamiltonian
    :param N:
    :param Ne:
    :param T_list: The list of symbolic tensors for Killers
    There should be N - 2 numbers of these.
    :param j: The virtual index
    :param b: The occupied index
    :return:
    """

    t_ferm_op_list = []
    for T in T_list:
        t_ferm_op_list.append(tensor_to_ferm_op(T, N))

    param_op = ferm_op


    virtual_orbitals = [p for p in range(N - Ne) if p != j]
    occupied_orbitals = [p for p in range(N - Ne, Ne) if p != b]


    for empty in virtual_orbitals:
        param_op -= t_ferm_op_list[empty] * (FermionOperator(((empty, 1), (empty, 0))))

    for occupied in occupied_orbitals:
        param_op -= t_ferm_op_list[occupied] * (FermionOperator(((occupied, 1), (occupied, 0))) - 1)

    majo = get_custom_majorana_operator(param_op)

    return majo


def tensor_to_ferm_op(tensor, N):
    ferm_op = FermionOperator()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    if (i, j, k, l) <= (l, k, j, i):
                        ferm_op += FermionOperator(f"{i}^ {j} {k}^ {l}",
                                                   tensor[i, j, k, l])
                        ferm_op += FermionOperator(f"{l}^ {k} {j}^ {i}",
                                                   tensor[l, k, j, i])

    return ferm_op


def construct_symmetric_tensor_specific(idx_lists):
    total_len = 0
    for idx_list in idx_lists:
        total_len += len(idx_list)
    symm_tensor = np.zeros(total_len)
    prev_idx = 0
    for idx_list in idx_lists:
        prev_len = len(idx_list)
        for i in range(prev_len):
            t = random.random() * 0.01
            symm_tensor[prev_idx + i] = t
        prev_idx += prev_len

    return symm_tensor
