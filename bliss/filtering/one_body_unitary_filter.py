from bliss.arrays_construction.tensor_fermion_operator import *
from bliss.majorana.custom_majorana_transform import get_custom_majorana_operator
import re
import numpy as np


def filter_one_body_generator(H, N, theta, j, b):
    """
    Construct the killer for one body generator unitary
    :param H: Operator to simplify
    :param N: The number of sites
    :param theta: The previously calculated parameter
    :param j: The index of the virtual site
    :param b: The index of the occupied site
    :return: The list of indices that are filtered out.
    """
    if np.sin(theta) == 0 or np.cos(theta) == 0:
        return []
    killer_coeff_candidate = symmetric_tensor_array(f'T{i}', N)

    # Tensor representation of the coefficient candidates
    tensor_repr = symmetric_tensor(killer_coeff_candidate, N)

    # Get the fermion operator representation
    tensor_ferm_op = tensor_to_ferm_op(tensor_repr, N)

    print("Tensor Ferm Op obtained")

    tan_theta = np.tan(theta)
    invtan_theta = 1 / tan_theta

    killer = tensor_ferm_op * (tan_theta * FermionOperator(f"{j}^ {b}")
                               + invtan_theta * FermionOperator(f"{b}^ {j}")
                               - 1)

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
            matches = re.findall(f'T{i}_(\\d{{4}})', str(expre))
            result = {tuple(map(int, match)) for match in
                      matches}  # Use set comprehension

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
    return result
