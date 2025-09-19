import numpy as np
import sympy as sp
from openfermion import FermionOperator
from bliss.majorana.custom_majorana_transform import get_custom_majorana_operator
from bliss.normal_bliss.one_norm_func_gen import upper_triangle_array, symmetric_matrix_from_upper_triangle, matrix_to_fermion_operator
import re
import pickle


def copy_ferm_hamiltonian(H: FermionOperator):
    H_copy = FermionOperator().zero()

    for t, s in H.terms.items():
        H_copy += s * FermionOperator(t)

    assert (H - H_copy) == FermionOperator().zero()
    return H_copy


def symmetric_tensor_array(name, n):
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
    # candidate_list = [10, 11] + idx_list
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if (i, j, k, l) <= (l, k, j, i):
                        tensor[i, j, k, l] = T[idx]
                        tensor[l, k, j, i] = T[idx]
                        idx += 1

    return tensor


def params_to_tensor_specific_op(params, n):

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
                    ferm_op += FermionOperator(f"{i}^ {j} {k}^ {l}", tensor[i, j, k, l])

    return ferm_op

def tensor_to_ferm_op(tensor, n):
    ferm_op = FermionOperator()

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    ferm_op += FermionOperator(f"{i}^ {j} {k}^ {l}",
                                               tensor[i, j, k, l])
                    # ferm_op += FermionOperator(f"{i}^ {j} {k}^ {l}",
                    #                            1)

    return ferm_op


def filter_indices(H, N, Ne):
    """
    Return the indices of which we apply the killers to
    But we want to
    :param H:
    :param N:
    :param Ne:
    :return:
    """
    # The killer coefficient candidates. We are going to remove unncessary ones from this
    killer_coeff_candidate = symmetric_tensor_array('T', N)

    # Tensor representation of the coefficient candidates
    tensor_repr = symmetric_tensor(killer_coeff_candidate, N)

    total_number_operator = FermionOperator()
    for mode in range(N):
        total_number_operator += FermionOperator(((mode, 1), (mode, 0)))

    # Get the fermion operator representation
    tensor_ferm_op = tensor_to_ferm_op(tensor_repr, N)

    print("Tensor Ferm Op obtained")

    killer = tensor_ferm_op * (total_number_operator - Ne)

    print("Fermion Representation Killer obtained")

    killer_in_majo = get_custom_majorana_operator(killer)
    killer_keys = killer_in_majo.terms.keys()

    H_in_majo = get_custom_majorana_operator(H)

    print("Majorana H obtained")

    candidate_coeff = set()
    remove_list = set()

    all_terms_counter = 0
    candidates_counter = 0
    invariant_1norm = 0



    print("Filter starts")
    for term, coeff in H_in_majo.terms.items():
        all_terms_counter += 1
        if term in killer_keys:
            expre = killer_in_majo.terms[term]
            matches = re.findall(r'T_(\d{4})', str(expre))
            result = {tuple(map(int, match)) for match in
                      matches}  # Use set comprehension

            if coeff != 0:
                candidates_counter += 1
                candidate_coeff.update(result)
            # else:
            #     remove_list.update(result)
        else:
            invariant_1norm += abs(coeff)

    print(f"Total Majo terms: {all_terms_counter}")
    print(f"optimized Majo terms: {candidates_counter}")
    print(f"Invariant 1-Norm: {invariant_1norm}")
    print("Filter ends")

    # Use set difference for efficiency
    result = list(candidate_coeff - remove_list)

    print(f"Number of parameters to be optimized: {len(result)}")

    return result


def filter_indices_normal_bliss(H, N, Ne):
    """
    Return the indices of which we apply the killers to
    But we want to
    :param H:
    :param N:
    :param Ne:
    :return:
    """
    # The killer coefficient candidates. We are going to remove unncessary ones from this
    killer_coeff_O = upper_triangle_array('O', N)

    # Tensor representation of the coefficient candidates
    O_matrix = symmetric_matrix_from_upper_triangle(killer_coeff_O, N)

    total_number_operator = FermionOperator()
    for mode in range(N):
        total_number_operator += FermionOperator(((mode, 1), (mode, 0)))

    # Get the fermion operator representation
    tensor_ferm_op = matrix_to_fermion_operator(O_matrix, N)

    print("Tensor Ferm Op obtained")

    killer = tensor_ferm_op * (total_number_operator - Ne)

    print("Fermion Representation Killer obtained")

    killer_in_majo = get_custom_majorana_operator(killer)
    killer_keys = killer_in_majo.terms.keys()

    H_in_majo = get_custom_majorana_operator(H)

    print("Majorana H obtained")

    candidate_coeff = set()
    remove_list = set()

    all_terms_counter = 0
    candidates_counter = 0
    invariant_1norm = 0



    print("Filter starts")
    for term, coeff in H_in_majo.terms.items():
        all_terms_counter += 1
        if term in killer_keys:
            expre = killer_in_majo.terms[term]
            matches = re.findall(r'O_(\d{2})', str(expre))
            result = {tuple(map(int, match)) for match in
                      matches}  # Use set comprehension

            if coeff != 0:
                candidates_counter += 1
                candidate_coeff.update(result)
            # else:
            #     remove_list.update(result)
        else:
            invariant_1norm += abs(coeff)

    print(f"Total Majo terms: {all_terms_counter}")
    print(f"optimized Majo terms: {candidates_counter}")
    print(f"Invariant 1-Norm: {invariant_1norm}")
    print("Filter ends")

    # Use set difference for efficiency
    result = list(candidate_coeff - remove_list)

    print(f"Number of parameters to be optimized: {len(result)}")

    return result


if __name__ == '__main__':
    N = 8
    Ne = 4

    filename = f'../../utils/ham_lib/h4_sto-3g.pkl'
    with open(filename, 'rb') as f:
        Hamil = pickle.load(f)

    # idx_list = [0, 1, 2, 3]
    G = FermionOperator('6^ 2 7^ 5') - FermionOperator('5^ 7 2^ 6')
    hamil_copy1 = copy_ferm_hamiltonian(Hamil)
    hamil_copy2 = copy_ferm_hamiltonian(Hamil)
    g_copy1 = copy_ferm_hamiltonian(G)
    g_copy2 = copy_ferm_hamiltonian(G)

    commutator = hamil_copy1 * g_copy1 - g_copy2 * hamil_copy2

    filter_indices(commutator, N, Ne)
