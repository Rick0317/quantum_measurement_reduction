from scipy.optimize import minimize
from openfermion import FermionOperator as FO
from bliss.hf_bliss.iterative_bliss import *
from bliss.hf_bliss.iterative_filter import filter_indices_iterative_HF
from copy import deepcopy

def copy_ferm_hamiltonian(H: FO):
    H_copy = FO().zero()

    for t, s in H.terms.items():
        H_copy += s * FO(t)

    assert (H - H_copy) == FO().zero()
    return H_copy


def params_to_tensor_specific_op(params, n, candidate_idx):

    tensor = np.zeros((n, n, n, n))
    idx = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
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


def construct_H_bliss_HF(H: FermionOperator, params, N, Ne, idx_lists):
    total_number_operator = FermionOperator()
    for mode in range(N):
        total_number_operator += FermionOperator(((mode, 1), (mode, 0)))

    occupation = [0 for _ in range(N - Ne)] + [1 for _ in range(Ne)]
    result = H
    prev_idx = 0
    for i in range(len(idx_lists)):
        idx_list = idx_lists[i]
        idx_len = len(idx_list)
        t = params[prev_idx:prev_idx + idx_len]
        prev_idx += idx_len

        t_ferm = params_to_tensor_specific_op(t, N, idx_list)
        result -= t_ferm * (FermionOperator(((i, 1), (i, 0))) - occupation[i])
    copied_result = deepcopy(result)
    for term, coeff in copied_result.terms.items():
        if term[-1][1] == 0 and term[-1][0] < N - Ne:
            print(term)
            result -= FermionOperator(term, coeff)

    return result, t_ferm * (total_number_operator - Ne)


def bliss_three_body_indices_filtered_HF(H, N, Ne):
    """
    Apply BLISS to three body Hamiltonian with indices filter used.
    :param H:
    :param N:
    :param Ne:
    :return:
    """
    idx_lists = filter_indices_iterative_HF(H, N, Ne)
    if len(idx_lists) != 0:
        H_input = copy_ferm_hamiltonian(H)

        optimization_wrapper, initial_guess = optimize_HF_BLISS(
            H_input, N, Ne, idx_lists)

        res = minimize(optimization_wrapper, initial_guess, method='Powell',
                       options={'disp': True, 'maxiter': 100000})

        H_before_modification = copy_ferm_hamiltonian(H)
        bliss_output, killer = construct_H_bliss_HF(
            H_before_modification, res.x, N, Ne, idx_lists)

        return bliss_output
    else:
        return H
