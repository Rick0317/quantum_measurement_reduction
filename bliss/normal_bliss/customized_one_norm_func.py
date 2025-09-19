import numpy as np
import random
from openfermion import FermionOperator
from bliss.majorana.custom_majorana_transform import get_custom_majorana_operator
import sympy as sp


def get_param_num(n):
    result = (2 * (8) ** 2 + 1) * (4 * (8) - 8) - (8 * (8) - 16) ** 2 // 2
    # return (result - 136) // 2
    return 66


def construct_symmetric_tensor_specific(n, idx_list):
    idx_len = len(idx_list)
    symm_tensor = np.zeros(idx_len)
    for i in range(idx_len):
        t = random.random() * 0.01
        # t = 0
        symm_tensor[i] = t

    return symm_tensor


def symmetric_tensor_from_triangle_specific(T, n, candidate_idx):
    """
    What symmetry do we apply?
    :param T:
    :param n:
    :return:
    """

    tensor = np.zeros((n, n, n, n), dtype=object)
    idx = 0
    # candidate_list = [10, 11] + idx_list
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
                    #                     tensor[i, j, k, l] = T[idx]
                    #                     tensor[l, k, j, i] = T[idx]
                    #                     idx += 1
                    if (i, j, k, l) in candidate_idx:
                        tensor[i, j, k, l] = T[idx]
                        tensor[l, k, j, i] = T[idx]
                        idx += 1

    return tensor


def symmetric_tensor_array_specific(name, n, candidate_idx):
    """
    idx_list is the list of indices corresponding to A
    :param name:
    :param n:
    :param idx_list:
    :return:
    """
    symmetric_tensor = []
    # candidate_list = [10, 11] + idx_list
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
                    #                     symmetric_tensor.append(sp.Symbol(f"{name}_{i}{j}{k}{l}"))
                    if (i, j, k, l) in candidate_idx:
                        symmetric_tensor.append(sp.Symbol(f"{name}_{i}{j}{k}{l}"))

    print("Length of Symmetric Tensor", len(symmetric_tensor))
    print("Predicted: ", len(candidate_idx))
    return sp.Matrix(symmetric_tensor)


def tensor_to_ferm_op(tensor, N):
    ferm_op = FermionOperator()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    ferm_op += FermionOperator(f'{i}^ {j} {k}^ {l}', tensor[i, j, k, l])

    return ferm_op


def construct_specific_tensor(n, idx_list):
    idx_len = len(idx_list)
    symm_tensor = np.zeros(idx_len)
    for i in range(idx_len):
        t = random.random() * 0.01
        # t = 0
        symm_tensor[i] = t

    return symm_tensor


def construct_majorana_terms_3_body_specific(ferm_op, N, ne, z, T):
    """Construct the Majorana terms from a parameterized FermionOperator."""
    total_number_operator = FermionOperator()
    for mode in range(N):
        total_number_operator += FermionOperator(((mode, 1), (mode, 0)))

    t_ferm_op = tensor_to_ferm_op(T, N)

    param_op = (ferm_op
                - z * (total_number_operator ** 3 - ne ** 3)
                - t_ferm_op * (total_number_operator - ne)
                )

    majo = get_custom_majorana_operator(param_op)

    return majo


def generate_analytical_one_norm_3_body_specific(ferm_op, N, ne, idx_list):
    """Generate the analytical one-norm of a parameterized Majorana operator."""
    z = sp.symbols('z')
    T = symmetric_tensor_array_specific('T', N, idx_list)

    T_tensor = symmetric_tensor_from_triangle_specific(T, N, idx_list)

    # Construct the Majorana terms manually
    majorana_terms = construct_majorana_terms_3_body_specific(ferm_op, N, ne, z, T_tensor )

    invariant_terms_counter = 0

    # for term, coeff in majorana_terms.terms.items():
    #     if coeff != 0 and str(coeff)[-1] == 'I':
    #         print(term, coeff)

    # Compute the symbolic one-norm
    one_norm_expr = sum(
        sp.Abs(coeff) for term, coeff in majorana_terms.terms.items() if
        term != ())

    one_norm_func = sp.lambdify((z, T), one_norm_expr,
                                modules=['numpy'])

    print("Analytical 1-Norm Complete")
    return one_norm_func, one_norm_expr
