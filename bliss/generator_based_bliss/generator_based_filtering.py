import re

import numpy as np
import sympy as sp
from openfermion import FermionOperator

from bliss.majorana.custom_majorana_transform import get_custom_majorana_operator


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
                        ferm_op += FermionOperator(
                            f"{i}^ {j} {k}^ {l}", tensor[i, j, k, l]
                        )
                        ferm_op += FermionOperator(
                            f"{l}^ {k} {j}^ {i}", tensor[l, k, j, i]
                        )

    return ferm_op


def filter_indices_iterative_non_overlap(H, N, Ne, j, b):
    """

    :param H: The Hamiltonian to which we filter the indices.
    :param N: The number of qubits
    :param Ne: The number of occupations
    :param j: The virtual index
    :param b: The occupied index
    :return: List of lists of indices that will be retained after filtering.
    The list should only contain N - 2 elements
    """

    occupation = [0 for _ in range(N - Ne)] + [1 for _ in range(Ne)]

    sites_list = [p for p in range(N) if p != j and p != b]

    idx_lists = []
    for i in sites_list:
        print(i)
        killer_coeff_candidate = symmetric_tensor_array(f"T{i}", N)

        # Tensor representation of the coefficient candidates
        tensor_repr = symmetric_tensor(killer_coeff_candidate, N)

        # Get the fermion operator representation
        tensor_ferm_op = tensor_to_ferm_op(tensor_repr, N)

        print("Tensor Ferm Op obtained")

        killer = tensor_ferm_op * (FermionOperator(((i, 1), (i, 0))) - occupation[i])

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
                matches = re.findall(f"T{i}_(\\d{{4}})", str(expre))
                result = {
                    tuple(map(int, match)) for match in matches
                }  # Use set comprehension

                if coeff != 0:
                    candidates_counter += 1
                    candidate_coeff.update(result)
                else:
                    remove_list.update(result)
            else:
                invariant_1norm += abs(coeff)

        print(f"Total Majo terms: {all_terms_counter}")
        print(f"optimized Majo terms: {candidates_counter}")
        print(f"Invariant 1-Norm: {invariant_1norm}")
        print("Filter ends")

        # Use set difference for efficiency
        result = list(candidate_coeff - remove_list)

        idx_lists.append(result)
        print(f"Number of parameters to be optimized: {len(result)}")

    return idx_lists
