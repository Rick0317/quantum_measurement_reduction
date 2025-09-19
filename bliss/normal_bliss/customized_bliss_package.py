from openfermion import FermionOperator
import numpy as np
from bliss.normal_bliss.customized_one_norm_func import (
    generate_analytical_one_norm_3_body_specific,
    construct_symmetric_tensor_specific)


def get_param_num(n):
    result = (2 * (8) ** 2 + 1) * (4 * (8) - 8) - (8 * (8) - 16) ** 2 // 2
    # return (result - 136) // 2
    return 66


def params_to_tensor_specific_op(params, n, candidate_idx):

    tensor = np.zeros((n, n, n, n))
    # candidate_list = [10, 11] + idx_list
    idx = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    # if len({i, j, k, l}) == 4:
                    #     evens_ij = sum(1 for x in [i, j] if x % 2 == 0)
                    #     evens_kl = sum(1 for x in [k, l] if x % 2 == 0)
                    #     if evens_ij == evens_kl:
                    #         if any(x in idx_list for x in [i, j, k, l]):
                    #             if not all(x in idx_list for x in [i, j, k, l]):
                    #                 if (i, j, k, l) <= (l, k, j, i):
                    #                     tensor[i, j, k, l] = params[idx]
                    #                     tensor[l, k, j, i] = params[idx]
                    #                     idx += 1
                    if (i, j, k, l) in candidate_idx:
                        tensor[i, j, k, l] = params[idx]
                        tensor[l, k, j, i] = params[idx]
                        idx += 1

    ferm_op = FermionOperator()

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    ferm_op += FermionOperator(f"{i}^ {j} {k}^ {l}", tensor[i, j, k, l])

    return ferm_op


def construct_H_bliss_mu3_customizable(H, params, N, Ne, idx_list):
    total_number_operator = FermionOperator()
    for mode in range(N):
        total_number_operator += FermionOperator(((mode, 1), (mode, 0)))

    idx_len = len(idx_list)
    result = H

    mu_3 = params[0]
    t = params[1:1 + idx_len]

    t_ferm = params_to_tensor_specific_op(t, N, idx_list)

    result -= mu_3 * (total_number_operator ** 3 - Ne ** 3)
    result -= t_ferm * (total_number_operator - Ne)

    return result, t_ferm * (total_number_operator - Ne)


def optimize_bliss_mu3_customizable(H, N, Ne, idx_list):
    """
    Get the analytical form of the 1-Norm with respect to the coefficients
    of the killer terms.
    :param H: Hamiltonian
    :param N: Number of orbitals
    :param Ne: Number of electrons
    :param idx_list:
    :return:
    """
    one_norm_func, one_norm_expr = generate_analytical_one_norm_3_body_specific(H, N, Ne, idx_list)
    idx_len = len(idx_list)
    def optimization_wrapper(params):
        z_val = params[0]
        t_val = params[1:1 + idx_len]

        return one_norm_func(z_val, t_val)

    z_val = 0

    t_val = construct_symmetric_tensor_specific(N, idx_list)

    initial_guess = np.concatenate((np.array([z_val]), t_val))
    # initial_guess = np.array([z_val])

    return optimization_wrapper, initial_guess
