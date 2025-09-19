import time

import numpy as np
from coefficient_optimizer import (
    get_meas_alloc,
    get_split_measurement_variance_unconstrained,
    optimize_coeffs,
    optimize_coeffs_parallel,
)
from openfermion import expectation
from openfermion import get_sparse_operator as gso
from shared_paulis import (
    get_all_pw_indices,
    get_coefficient_orderings,
    get_overlapping_decomp,
    get_pauli_coeff_map,
    get_pw_grp_idxes_no_fix_len,
    get_sharable_only_decomp,
    get_sharable_paulis,
    get_share_pauli_only_decomp,
    qubit_op_to_list,
)


def apply_shared_pauli(H_q, decomposition, N, Ne, state):
    """
    Apply the Shared Pauli method to the decomposition of the quantum Hamiltonian
    :param H_q:
    :param decomposition:
    :param N:
    :param Ne:
    :param state:
    :return:
    """
    start = time.time()
    coeff_map = get_pauli_coeff_map(decomposition)

    sharable_paulis_dict, sharable_paulis_list, sharable_pauli_indices_list = (
        get_sharable_paulis(decomposition)
    )
    sharable_paulis_fixed_list = [
        indices[-1] for indices in sharable_pauli_indices_list
    ]

    # 2
    fragment_idx_to_sharable_paulis, pw_grp_idxes_no_fix, pw_grp_idxes_fix = (
        get_share_pauli_only_decomp(sharable_paulis_dict)
    )

    pw_grp_idxes_no_fix_len = get_pw_grp_idxes_no_fix_len(pw_grp_idxes_no_fix)

    # 3
    all_sharable_contained_decomp, all_sharable_no_fixed_decomp = (
        get_overlapping_decomp(sharable_paulis_dict, decomposition, pw_grp_idxes_fix)
    )

    # 4
    sharable_only_decomp, sharable_only_no_fixed_decomp = get_sharable_only_decomp(
        sharable_paulis_dict, decomposition, pw_grp_idxes_fix, coeff_map
    )

    # 5
    (
        fixed_grp,
        all_sharable_contained_no_fixed_decomp,
        new_sharable_pauli_indices_list,
        new_grp_len_list,
        new_grp_idx_start,
    ) = get_coefficient_orderings(
        sharable_only_decomp,
        sharable_paulis_list,
        sharable_pauli_indices_list,
        coeff_map,
    )

    # 6
    pw_indices = get_all_pw_indices(
        sharable_paulis_list,
        sharable_only_no_fixed_decomp,
        pw_grp_idxes_no_fix,
        new_grp_idx_start,
    )

    meas_alloc = get_meas_alloc(decomposition, N, state)

    # Get the linear equation for finding the gradient descent direction
    matrix, b = optimize_coeffs_parallel(
        pw_grp_idxes_fix,
        pw_grp_idxes_no_fix_len,
        meas_alloc,
        sharable_only_decomp,
        sharable_only_no_fixed_decomp,
        pw_indices,
        state,
        N,
        decomposition,
        alpha=0.001,
    )
    print("M, b obtained")
    sol = np.linalg.lstsq(matrix, b.T, rcond=None)
    x0 = sol[0]
    coeff = x0.T[0]

    end = time.time()
    print(f"Updated coefficient obtained in {end - start}")

    # Update the fragment by modifying the coefficients of shared Pauli operators.
    last_var, meas_alloc, var, fixed_grp_coefficients, measured_groups = (
        get_split_measurement_variance_unconstrained(
            coeff,
            decomposition,
            sharable_paulis_list,
            sharable_paulis_fixed_list,
            sharable_only_no_fixed_decomp,
            pw_grp_idxes_no_fix,
            new_grp_idx_start,
            H_q,
            N,
            state,
            Ne,
        )
    )

    expectation_v = 0
    for fragment in measured_groups:
        expectation_v += expectation(gso(fragment, N), state)

    return var, last_var, measured_groups, expectation_v
