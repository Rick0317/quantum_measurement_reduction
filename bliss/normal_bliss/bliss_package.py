import numpy as np
from openfermion import (
    FermionOperator,
    get_ground_state,
    get_sparse_operator,
    jw_get_ground_state_at_particle_number,
)

from bliss.normal_bliss.one_norm_func_gen import (
    construct_symmetric_matrix,
    construct_symmetric_tensor,
    generate_analytical_one_norm_2_body,
    generate_analytical_one_norm_3_body,
    generate_analytical_one_norm_3_body_cheap,
    generate_analytical_one_norm_3_body_simple,
)


def params_to_matrix_op(params, n):
    """
    Given a parameter set, return Fermionic Operator object representing the
    Hamiltonian with the parameters used in the coefficient tensor.
    :param params: upper-triangular matrix params: n*(n+1)/2
    :param n:
    :return:
    """
    ferm_op = FermionOperator()
    upper_tri_indices = np.triu_indices(n)
    sym_matrix = np.zeros((n, n))
    sym_matrix[upper_tri_indices] = params
    sym_matrix = sym_matrix + sym_matrix.T - np.diag(np.diag(sym_matrix))
    for i in range(n):
        for j in range(n):
            ferm_op += FermionOperator(f"{i}^ {j}", sym_matrix[i, j])

    return ferm_op


def params_to_tensor_op(params, n):

    tensor = np.zeros((n, n, n, n))
    idx = 0
    for i in range(n):
        for j in range(i, n):
            for k in range(n):
                for l in range(k, n):
                    tensor[i, j, k, l] = params[idx]
                    tensor[j, i, k, l] = params[idx]
                    tensor[i, j, l, k] = params[idx]
                    tensor[j, i, l, k] = params[idx]
                    idx += 1

    ferm_op = FermionOperator()

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    ferm_op += FermionOperator(f"{i}^ {j} {k}^ {l}", tensor[i, j, k, l])

    return ferm_op


def params_to_tensor_specific_op(params, n, idx_list):

    tensor = np.zeros((n, n, n, n))
    idx = 0
    for i in idx_list:
        for j in idx_list:
            for k in idx_list:
                for l in idx_list:
                    if (i, j, k, l) <= (l, k, j, i):
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


def construct_H_bliss_mu3_o2(H, params, N, Ne):
    total_number_operator = FermionOperator()
    for mode in range(N):
        total_number_operator += FermionOperator(((mode, 1), (mode, 0)))

    result = H
    mu_3 = params[0]
    t = params[1 : 1 + int(N * (N + 1) // 2) ** 2]

    t_ferm = params_to_tensor_op(t, N)

    result -= mu_3 * (total_number_operator**3 - Ne**3)
    result -= t_ferm * (total_number_operator - Ne)

    return result


def construct_H_bliss_mu3_cheapo(H, params, N, Ne, idx_list):
    total_number_operator = FermionOperator()
    for mode in range(N):
        total_number_operator += FermionOperator(((mode, 1), (mode, 0)))

    idx_len = len(idx_list)
    result = H
    mu_3 = params[0]
    t = params[1 : 1 + int(idx_len**2 * (idx_len**2 + 1) // 2)]

    t_ferm = params_to_tensor_specific_op(t, N, idx_list)

    result -= mu_3 * (total_number_operator**3 - Ne**3)
    result -= t_ferm * (total_number_operator - Ne)

    return result


def construct_H_bliss_mu123_o12(H, params, N, Ne):
    total_number_operator = FermionOperator()
    for mode in range(N):
        total_number_operator += FermionOperator(((mode, 1), (mode, 0)))
    result = H
    mu_1 = params[0]
    mu_2 = params[1]
    mu_3 = params[2]
    o_1 = params[3 : int(N * (N + 1) // 2) + 3]
    o_2 = params[3 + int(N * (N + 1) // 2) :]
    o1_ferm = params_to_matrix_op(o_1, N)
    o2_ferm = params_to_matrix_op(o_2, N)

    result -= mu_1 * (total_number_operator - Ne)
    result -= mu_2 * (total_number_operator**2 - Ne**2)
    result -= mu_3 * (total_number_operator**3 - Ne**3)
    result -= o1_ferm * (total_number_operator - Ne)
    result -= o2_ferm * (total_number_operator**2 - Ne**2)

    return result


def construct_H_bliss_m12_o1(H, params, N, Ne):
    total_number_operator = FermionOperator()
    for mode in range(N):
        total_number_operator += FermionOperator(((mode, 1), (mode, 0)))
    result = H
    mu_1 = params[0]
    mu_2 = params[1]
    o_1 = params[2 : 2 + int(N * (N + 1) // 2)]

    o1_ferm = params_to_matrix_op(o_1, N)

    result -= mu_1 * (total_number_operator - Ne)
    result -= mu_2 * (total_number_operator**2 - Ne**2)
    result -= o1_ferm * (total_number_operator - Ne)

    return result


def optimize_bliss_mu3_o2(H, N, Ne):
    one_norm_func, one_norm_expr = generate_analytical_one_norm_3_body_simple(H, N, Ne)

    def optimization_wrapper(params):
        z_val = params[0]
        t_val = params[1 : 1 + int(N * (N + 1) // 2) ** 2]

        return one_norm_func(z_val, t_val)

    z_val = 0

    t_val = construct_symmetric_tensor(N)

    initial_guess = np.concatenate((np.array([z_val]), t_val))
    # initial_guess = np.array([z_val])

    return optimization_wrapper, initial_guess


def optimize_bliss_mu3_cheapo(H, N, Ne):
    one_norm_func, one_norm_expr = generate_analytical_one_norm_3_body_cheap(H, N, Ne)

    def optimization_wrapper(params):
        z_val = params[0]
        o_val = params[1 : 1 + int(N * (N + 1) // 2)]
        o2_val = params[1 + int(N * (N + 1) // 2) :]

        return one_norm_func(z_val, o_val, o2_val)

    z_val = 0

    o_val = construct_symmetric_matrix(N)
    o2_val = construct_symmetric_matrix(N)

    initial_guess = np.concatenate((np.array([z_val]), o_val, o2_val))
    # initial_guess = np.array([z_val])

    return optimization_wrapper, initial_guess


def optimize_bliss_mu3_cheapo(H, N, Ne, idx_list):
    one_norm_func, one_norm_expr = generate_analytical_one_norm_3_body_specific(
        H, N, Ne, idx_list
    )
    idx_len = len(idx_list)

    def optimization_wrapper(params):
        z_val = params[0]
        t_val = params[1 : 1 + int(idx_len**2 * (idx_len**2 + 1) // 2)]

        return one_norm_func(z_val, t_val)

    z_val = 0

    t_val = construct_symmetric_tensor(N)

    initial_guess = np.concatenate((np.array([z_val]), t_val))
    # initial_guess = np.array([z_val])

    return optimization_wrapper, initial_guess


def optimize_bliss_mu123_o12(H, N, Ne):
    one_norm_func, one_norm_expr = generate_analytical_one_norm_3_body(H, N, Ne)

    def optimization_wrapper(params):
        x_val = params[0]
        y_val = params[1]
        z_val = params[2]
        o_vals = params[3 : int(N * (N + 1) // 2) + 3]
        o_val2 = params[int(N * (N + 1) // 2) + 3 :]
        return one_norm_func(x_val, y_val, z_val, o_vals, o_val2)

    x_val = 0
    y_val = 0
    z_val = 0
    o_val = construct_symmetric_matrix(N)
    o_val2 = construct_symmetric_matrix(N)

    initial_guess = np.concatenate((np.array([x_val, y_val, z_val]), o_val, o_val2))

    return optimization_wrapper, initial_guess


def optimization_bliss_mu12_o1(H, N, Ne):

    one_norm_func, one_norm_expr = generate_analytical_one_norm_2_body(H, N, Ne)

    def optimization_wrapper(params):
        x_val = params[0]
        y_val = params[1]
        o_vals = params[2 : 2 + int(N * (N + 1) // 2)]
        return one_norm_func(x_val, y_val, o_vals)

    x_val = 0
    y_val = 0
    # o_val is 1 by n*(n+1)//2 array
    o_val = construct_symmetric_matrix(N)

    initial_guess = np.concatenate((np.array([x_val, y_val]), o_val))

    return optimization_wrapper, initial_guess


def check_correctness(H_orig: FermionOperator, H_bliss: FermionOperator, Ne):
    """
    Checks that the H_bliss operator is correct.
    :param H_orig:
    :param H_bliss:
    :return:
    """
    sparse_orig1 = get_sparse_operator(H_orig)
    sparse_bliss1 = get_sparse_operator(H_bliss)

    # 1, Subspace min energy check
    gs_orig = jw_get_ground_state_at_particle_number(sparse_orig1, Ne)[0]
    gs_bliss = jw_get_ground_state_at_particle_number(sparse_bliss1, Ne)[0]

    print("Original Subspace Min:", gs_orig)
    print("Bliss Subspace Min:", gs_bliss)

    assert np.isclose(gs_orig, gs_bliss, atol=1e-3), "Subspace Min changed"

    # 2, Spectral range check
    sparse_orig2 = get_sparse_operator(H_orig)
    sparse_bliss2 = get_sparse_operator(H_bliss)

    min_orig = get_ground_state(sparse_orig2)[0]
    min_bliss = get_ground_state(sparse_bliss2)[0]

    sparse_orig3 = get_sparse_operator(H_orig)
    sparse_bliss3 = get_sparse_operator(H_bliss)

    max_orig = get_ground_state(-sparse_orig3)[0]
    max_bliss = get_ground_state(-sparse_bliss3)[0]

    spect_range_orig = -max_orig - min_orig
    spect_range_bliss = -max_bliss - min_bliss

    print("Original Spectral Range:", spect_range_orig)
    print("Bliss Spectral Range:", spect_range_bliss)

    assert spect_range_orig + 1e-3 >= spect_range_bliss, "Spectral range reduced"
