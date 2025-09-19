from openfermion import FermionOperator as FO
from openfermion import is_hermitian, normal_ordered
from scipy.optimize import minimize

from bliss.generator_based_bliss.generator_based_bliss import *
from bliss.generator_based_bliss.generator_based_filtering import (
    filter_indices_iterative_non_overlap,
)


def copy_ferm_hamiltonian(H: FO):
    H_copy = FO().zero()

    for t, s in H.terms.items():
        H_copy += s * FO(t)

    assert (H - H_copy) == FO().zero()
    return H_copy


def params_to_tensor_specific_op(params, n, candidate_idx):
    """
    Converts a set of parameters to a tensor representation and FermionOperator
    xi_{ijkl} F_{ijkl} part
    :param params:
    :param n:
    :param candidate_idx:
    The filtered indices that contribute to 1-Norm Reduction
    :return
    """

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
                    if (i, j, k, l) <= (l, k, j, i):
                        ferm_op += FermionOperator(
                            f"{i}^ {j} {k}^ {l}", tensor[i, j, k, l]
                        )
                        ferm_op += FermionOperator(
                            f"{l}^ {k} {j}^ {i}", tensor[l, k, j, i]
                        )

    return ferm_op


def construct_H_bliss_non_overlap(H, params, N, Ne, idx_lists, j, b):
    """
    Reconstruct the Hamiltonian with the killer applied: H - K
    :param H: Hamiltonian
    :param params:
    :param N: Number of sites
    :param Ne: Number of electrons
    :param idx_lists: The list of lists of indices that contribute to
    1-Norm Reduction. There should be N - 2 number of such lists
    :param j: The virtual index of the generator used
    :param b: The occupied index of the generator used
    :return:
    """

    occupation = [0 for _ in range(N - Ne)] + [1 for _ in range(Ne)]

    sites_list = [p for p in range(N) if p != j and p != b]
    result = H
    prev_idx = 0
    for i in range(N - 2):
        site = sites_list[i]
        idx_list = idx_lists[i]
        idx_len = len(idx_list)
        t = params[prev_idx : prev_idx + idx_len]
        prev_idx += idx_len

        t_ferm = params_to_tensor_specific_op(t, N, idx_list)
        print(f"is hermitian {is_hermitian(t_ferm)}")
        result -= t_ferm * (
            FermionOperator(((site, 1), (site, 0))) - occupation[sites_list[i]]
        )

    return result


def bliss_three_body_indices_filtered_non_overlap(H, N, Ne, j, b):
    """
    Apply BLISS to three body Hamiltonian with indices filter used.
    :param H: Hamiltonian
    :param N: Number of sites
    :param Ne: Number of electrons
    :param j: The virtual index of the generator used
    :param b: The occupied index of the generator used
    :return:
    """

    # N - 2 indices lists
    idx_lists = filter_indices_iterative_non_overlap(H, N, Ne, j, b)
    if len(idx_lists) != 0:
        H_input = copy_ferm_hamiltonian(H)

        optimization_wrapper, initial_guess = optimize_non_overlap_BLISS(
            H_input, N, Ne, idx_lists, j, b
        )

        res = minimize(
            optimization_wrapper,
            initial_guess,
            method="Powell",
            options={"disp": True, "maxiter": 100000},
        )

        H_before_modification = copy_ferm_hamiltonian(H)
        bliss_output = construct_H_bliss_non_overlap(
            H_before_modification, res.x, N, Ne, idx_lists, j, b
        )

        return normal_ordered(bliss_output)
    else:
        return H
