from copy import deepcopy
from multiprocessing import Pool

import numpy as np
from openfermion import (
    FermionOperator,
    QubitOperator,
    bravyi_kitaev,
    expectation,
)
from openfermion import get_ground_state as ggs
from openfermion import get_sparse_operator as gso
from openfermion import (
    normal_ordered,
    variance,
)
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

from utils.math_utils import (
    commutator_variance,
    cov_frag_pauli,
    cov_frag_pauli_iterative,
    cov_pauli_pauli,
    variance_of_group,
)


def variance_metric(H, decomp, N):
    psi = ggs(gso(H, N))[1]

    vars = np.zeros(len(decomp), dtype=np.complex128)
    for i, frag in enumerate(decomp):
        vars[i] = variance(gso(frag, N), psi)
    return np.sum((vars) ** (1 / 2)) ** 2


def abs_of_dict_value(x):
    return np.abs(x[1])


def copy_hamiltonian(H):
    H_copy = QubitOperator().zero()

    for t, s in H.terms.items():
        H_copy += s * QubitOperator(t)

    assert (H - H_copy) == QubitOperator().zero()
    return H_copy


def get_pauli_word_tuple(P: QubitOperator):
    """Given a single pauli word P, extract the tuple representing the word."""
    words = list(P.terms.keys())
    if len(words) != 1:
        raise (ValueError("P given is not a single pauli word"))
    return words[0]


def optimize_coeffs(
    pw_grp_idxes_fix,
    pw_grp_idxes_no_fix_len,
    meas_alloc,
    shara_pauli_only_decomp,
    shara_pauli_only_no_fixed_decomp,
    pw_indices_dict,
    psi,
    n_qubits,
    original_decomp,
    alpha,
):
    """
    Prepares the linear system M @ x = b
    pw_grp_idxes_fix: Mapping of each Pauli to its corresponding fixed index. (Last index of appearance in the decomposition)
    pw_grp_idxes_no_fix_len: List of number of appearences of each sharable Pauli except for the fixed index.
    meas_alloc: Measurement allocations for each fragment of the Hamiltonian.
    shara_pauli_only_decomp: The fragmentation that only contains sharable Pauli words
    every_pauli_contained_decomp: Overlapping allowed decomposition/
    pw_indices_dict: Mapping each Pauli word to the corresponding indices
    :return: M, b
    """
    # The number of total appearances of sharable Paulis except for the fixed ones.
    # @ Verified
    matrix_row_column_size = np.sum(pw_grp_idxes_no_fix_len)
    print(f"Matrix Size: {matrix_row_column_size}")
    total_fragments = len(shara_pauli_only_no_fixed_decomp)
    print(f"Number of fragments: {len(shara_pauli_only_no_fixed_decomp)}")
    matrix = np.zeros((matrix_row_column_size, matrix_row_column_size))
    b = np.zeros((1, matrix_row_column_size))

    row_idx = 0
    for frag_idx, fragment in enumerate(shara_pauli_only_no_fixed_decomp):
        print(f"Fragment Index: {frag_idx} / {total_fragments}")
        meas_a = meas_alloc[frag_idx]

        # For each sharable Pauli in the fragment, except for the fixed fragment
        # we will calculate the gradient in the Variance-coefficient space
        for spw_idx, pws in enumerate(fragment.terms):
            pauli_word_s = QubitOperator(term=pws)

            # For each sharable Pauli in the same fragment,
            for tpw_idx, pwt in enumerate(shara_pauli_only_decomp[frag_idx].terms):

                # Coefficients of the Pauli word t in the fragment.
                pauli_word_t = QubitOperator(term=pwt)
                # coeff_t = shara_pauli_only_decomp[frag_idx].terms[pwt]

                # Covariance of P_s and P_t.
                cov_st_paulis = (1 - alpha) * cov_pauli_pauli(
                    gso(pauli_word_s, n_qubits=n_qubits),
                    gso(pauli_word_t, n_qubits=n_qubits),
                    psi,
                )

                if pwt == pws:
                    cov_st_paulis += alpha * var_avg(n_qubits)

                # Get the indices in the coefficient vector where P_t shows up
                pw_idx_dict = pw_indices_dict[get_pauli_word_tuple(pauli_word_t)]

                # If the pauli word pwt is also a sharable pauli in the fragment,
                if pwt in shara_pauli_only_no_fixed_decomp[frag_idx].terms:
                    matrix[row_idx, pw_idx_dict[frag_idx]] -= (
                        cov_st_paulis.real / meas_a
                    )

                # If tpw is not a sharable Pauli.
                else:
                    # For each appearance ofx the P_t in the coefficient vector,
                    for idxs in pw_idx_dict.values():
                        matrix[row_idx, idxs] += cov_st_paulis.real / meas_a

                del pauli_word_t, cov_st_paulis, pw_idx_dict

            cov_paulis_H = cov_frag_pauli(
                gso(original_decomp[frag_idx], n_qubits=n_qubits),
                gso(pauli_word_s, n_qubits=n_qubits),
                psi,
            )
            b[0, row_idx] += (1.0 - alpha) * cov_paulis_H.real / meas_a

            # The index of the fragment where the spw appears in the end
            # @ Verified
            fixed_group_index = pw_grp_idxes_fix[pws]

            for kpw_idx, kpw in enumerate(
                shara_pauli_only_decomp[fixed_group_index].terms
            ):
                pauli_word_k = QubitOperator(term=kpw)
                # coeff_k = shara_pauli_only_decomp[fixed_group_index].terms[kpw]

                cov_ks_paulis = (1 - alpha) * cov_pauli_pauli(
                    gso(pauli_word_s, n_qubits=n_qubits),
                    gso(pauli_word_k, n_qubits=n_qubits),
                    psi,
                )

                if pws == kpw:
                    cov_ks_paulis += alpha * var_avg(n_qubits)

                pw_idx_dict = pw_indices_dict[get_pauli_word_tuple(pauli_word_k)]

                # If the pauli word tpw is also a sharable pauli,
                if kpw in shara_pauli_only_no_fixed_decomp[fixed_group_index].terms:
                    matrix[row_idx, pw_idx_dict[fixed_group_index]] += (
                        cov_ks_paulis.real / meas_alloc[fixed_group_index]
                    )

                # If tpw is not a sharable Pauli.
                else:
                    for idxs in pw_idx_dict.values():
                        matrix[row_idx, idxs] -= (
                            cov_ks_paulis.real / meas_alloc[fixed_group_index]
                        )

                del pauli_word_k, cov_ks_paulis, pw_idx_dict

            cov_paulik_H = cov_frag_pauli(
                gso(original_decomp[fixed_group_index], n_qubits=n_qubits),
                gso(pauli_word_s, n_qubits=n_qubits),
                psi,
            )

            b[0, row_idx] -= (
                (1.0 - alpha) * cov_paulik_H.real / meas_alloc[fixed_group_index]
            )

            row_idx += 1

    return matrix, b


def process_fragment(args):
    """Computes a portion of the matrix and b for a given fragment index."""
    (
        frag_idx,
        fragment,
        shara_pauli_only_decomp,
        original_decomp,
        pw_indices_dict,
        meas_alloc,
        alpha,
        psi,
        n_qubits,
        var_avg,
        gso,
        cov_pauli_pauli,
        cov_frag_pauli_iterative,
        get_pauli_word_tuple,
        pw_grp_idxes_fix,
        shara_pauli_only_no_fixed_decomp,
        row_idx_start,
    ) = args

    row_results = []
    b_results = []

    row_idx = row_idx_start
    meas_a = meas_alloc[frag_idx]

    print(f"Processing Fragment: {frag_idx}")

    for spw_idx, pws in enumerate(fragment.terms):
        pauli_word_s = QubitOperator(term=pws)

        row_data = {}

        for tpw_idx, pwt in enumerate(shara_pauli_only_decomp[frag_idx].terms):
            pauli_word_t = QubitOperator(term=pwt)
            cov_st_paulis = (1 - alpha) * cov_pauli_pauli(
                gso(pauli_word_s, n_qubits=n_qubits),
                gso(pauli_word_t, n_qubits=n_qubits),
                psi,
            )
            if pwt == pws:
                cov_st_paulis += alpha * var_avg(n_qubits)

            pw_idx_dict = pw_indices_dict[get_pauli_word_tuple(pauli_word_t)]

            if pwt in fragment.terms:
                row_data[pw_idx_dict[frag_idx]] = -cov_st_paulis.real / meas_a
            else:
                for idxs in pw_idx_dict.values():
                    if idxs not in row_data:
                        row_data[idxs] = cov_st_paulis.real / meas_a
                    else:
                        row_data[idxs] += cov_st_paulis.real / meas_a

        cov_paulis_H = cov_frag_pauli_iterative(
            original_decomp[frag_idx], pauli_word_s, psi, n_qubits, alpha
        )
        # cov_frag_pauli(
        #     gso(original_decomp[frag_idx], n_qubits=n_qubits),
        #     gso(pauli_word_s, n_qubits=n_qubits),
        #     psi
        # )
        b_value = cov_paulis_H.real / meas_a

        fixed_group_index = pw_grp_idxes_fix[pws]

        for kpw_idx, kpw in enumerate(shara_pauli_only_decomp[fixed_group_index].terms):
            pauli_word_k = QubitOperator(term=kpw)
            cov_ks_paulis = (1 - alpha) * cov_pauli_pauli(
                gso(pauli_word_s, n_qubits=n_qubits),
                gso(pauli_word_k, n_qubits=n_qubits),
                psi,
            )
            if pws == kpw:
                cov_ks_paulis += alpha * var_avg(n_qubits)

            pw_idx_dict = pw_indices_dict[get_pauli_word_tuple(pauli_word_k)]

            if kpw in shara_pauli_only_no_fixed_decomp[fixed_group_index].terms:
                if pw_idx_dict[fixed_group_index] not in row_data:
                    row_data[pw_idx_dict[fixed_group_index]] = (
                        cov_ks_paulis.real / meas_alloc[fixed_group_index]
                    )
                else:
                    row_data[pw_idx_dict[fixed_group_index]] += (
                        cov_ks_paulis.real / meas_alloc[fixed_group_index]
                    )
            else:
                for idxs in pw_idx_dict.values():
                    if idxs not in row_data:
                        row_data[idxs] = (
                            -cov_ks_paulis.real / meas_alloc[fixed_group_index]
                        )
                    else:
                        row_data[idxs] -= (
                            cov_ks_paulis.real / meas_alloc[fixed_group_index]
                        )

        cov_paulik_H = cov_frag_pauli_iterative(
            original_decomp[fixed_group_index], pauli_word_s, psi, n_qubits, alpha
        )
        b_value -= cov_paulik_H.real / meas_alloc[fixed_group_index]

        row_results.append((row_idx, row_data))
        b_results.append((row_idx, b_value))

        row_idx += 1  # Increment row index correctly

    return row_results, b_results


def optimize_coeffs_parallel(
    pw_grp_idxes_fix,
    pw_grp_idxes_no_fix_len,
    meas_alloc,
    shara_pauli_only_decomp,
    shara_pauli_only_no_fixed_decomp,
    pw_indices_dict,
    psi,
    n_qubits,
    original_decomp,
    alpha,
):

    matrix_row_column_size = np.sum(pw_grp_idxes_no_fix_len)
    print(f"Matrix Size: {matrix_row_column_size}")
    total_fragments = len(shara_pauli_only_no_fixed_decomp)
    print(f"Number of fragments: {total_fragments}")

    matrix = np.zeros((matrix_row_column_size, matrix_row_column_size))
    b = np.zeros((1, matrix_row_column_size))

    # Precompute starting row indices for each fragment
    row_idx_starts = {}
    row_idx = 0
    for frag_idx, fragment in enumerate(shara_pauli_only_no_fixed_decomp):
        row_idx_starts[frag_idx] = row_idx
        row_idx += len(fragment.terms)  # Increment by number of terms in fragment

    pool = Pool()  # Adjust processes if needed
    tasks = [
        (
            frag_idx,
            fragment,
            shara_pauli_only_decomp,
            original_decomp,
            pw_indices_dict,
            meas_alloc,
            alpha,
            psi,
            n_qubits,
            var_avg,
            gso,
            cov_pauli_pauli,
            cov_frag_pauli_iterative,
            get_pauli_word_tuple,
            pw_grp_idxes_fix,
            shara_pauli_only_no_fixed_decomp,
            row_idx_starts[frag_idx],
        )
        for frag_idx, fragment in enumerate(shara_pauli_only_no_fixed_decomp)
    ]

    results = pool.map(process_fragment, tasks)
    pool.close()
    pool.join()

    # Merge results
    for row_results, b_results in results:
        for row_idx, row_data in row_results:
            for col_idx, value in row_data.items():
                matrix[row_idx, col_idx] += value
        for row_idx, b_value in b_results:
            b[0, row_idx] += b_value

    return matrix, b


def get_pauli_word_coefficient(P: QubitOperator, ghosts=None):
    """Given a single pauli word P, extract its coefficient."""
    if ghosts is not None:
        if P in ghosts:
            coeffs = [0.0]
        else:
            coeffs = list(P.terms.values())
    else:
        coeffs = list(P.terms.values())
    return coeffs[0]


def remove_term(qubit_op: QubitOperator, removed_term):
    new_qubit_op = QubitOperator()
    for idx, term in enumerate(qubit_op.terms):
        if term != removed_term:
            new_qubit_op += QubitOperator(term) * qubit_op.terms[term]
        else:
            print(f"Found ther term: {term}")

    return new_qubit_op


def get_split_measurement_variance_unconstrained(
    coeff,
    no_group,
    sharable_paulis_list,
    sharable_paulis_fixed_list,
    sharable_no_fixed_decomp,
    pw_grp_idxes_no_fix,
    new_grp_idx_start,
    H_q,
    n_qubits,
    wfs,
    Ne,
):
    """
    Finds the estimator variance for a given coefficient split.
    Args:
        coeff (np.array): Array of coefficients ordered by group first then pauli words in new_overlapped_group.
        no_group (List[List[QubitOperator]]): Fragmentation before updaing
        sharable_paulis_list(List[QubitOperator]): List of sharable Pauli operators.
        sharable_paulis_fixed_list (List[int]): List of fixed indices of each sharable Pauli operator.
        sharable_no_fixed_decomp (List[List[QubitOperator]]): Overlapped Grouping with the fixed group Pauli removed (last group that the Pauli appears in).
        pw_grp_idxes_no_fix (List[List[int]]): List of list of corresponding group indexes that each Pauli word appears without the last indices.
        new_grp_idx_start (List[int]): List of the starting index of each group (in the coefficient vector).
        n_qubits (int): Number of qubits of the Hamiltonian.
    Returns:
        variance (float): Variance of the Hamiltonian estimator calculated with optimal measurement allocation.
        meas_alloc (List[float]): Optimal measurement allocation for the coefficient split.
        var (List[Float]): List of the fragment variances
        fixed_grp_coefficients (List[float]): List of coefficients in the fixed group.
    """

    # 1. Find the fixed group coefficients for each Paulis
    # This part seems to be correct: This index was correct:
    # (new_grp_idx_start[grp_idx] + qubit_op_to_list(sharable_no_fixed_decomp[grp_idx]).index(pw))
    fixed_grp_coefficients = []
    for pw_idx, pw in enumerate(sharable_paulis_list):
        coefficient = 0.0
        for grp_idx in pw_grp_idxes_no_fix[pw]:
            coefficient -= coeff[
                new_grp_idx_start[grp_idx]
                + qubit_op_to_list(sharable_no_fixed_decomp[grp_idx]).index(pw)
            ]

        fixed_grp_coefficients.append(coefficient)

    """
    Computing Group Variances & Total Variance
    """

    var = []
    measured_groups = deepcopy(no_group)

    # 2. Update the coefficients
    for pw_idx, pw in enumerate(sharable_paulis_list):

        # For each sharable fragment that this pw can belog to.
        for grp_idx in pw_grp_idxes_no_fix[pw]:
            added = False

            copy_measured_groups = deepcopy(measured_groups)

            # Search through all the Pauli words in the fragment
            # Look for whether the Pauli word already exists or not.
            for igp, group_word in enumerate(copy_measured_groups[grp_idx].terms):
                if group_word == pw:
                    # measured_groups[grp_idx] += (coeff[new_grp_idx_start[grp_idx] + qubit_op_to_list(sharable_no_fixed_decomp[grp_idx]).index(pw)] + original_coeff_map[group_word]) * QubitOperator(pw)

                    # Updating the coefficient by adding the
                    # change rate to the coefficient
                    measured_groups[grp_idx] += (
                        coeff[
                            new_grp_idx_start[grp_idx]
                            + qubit_op_to_list(sharable_no_fixed_decomp[grp_idx]).index(
                                pw
                            )
                        ]
                    ) * QubitOperator(pw)
                    added = True

            if not (added):
                measured_groups[grp_idx] += coeff[
                    new_grp_idx_start[grp_idx]
                    + qubit_op_to_list(sharable_no_fixed_decomp[grp_idx]).index(pw)
                ] * QubitOperator(pw)

        if not (sharable_paulis_fixed_list[pw_idx] == None):
            fixed_idx = sharable_paulis_fixed_list[pw_idx]
            added = False
            copy_measured_groups = deepcopy(measured_groups)
            for igp, group_word in enumerate(copy_measured_groups[fixed_idx].terms):
                if group_word == pw:
                    measured_groups[fixed_idx] += (
                        fixed_grp_coefficients[pw_idx]
                    ) * QubitOperator(pw)
                    added = True
            if not (added):
                measured_groups[fixed_idx] += fixed_grp_coefficients[
                    pw_idx
                ] * QubitOperator(pw)

    # 3. Get the variances.
    var_new = commutator_variance(H_q, measured_groups, n_qubits, wfs)
    print(f"Updated variance: {var_new}")
    var, variance = get_measurement_variance_simple(measured_groups, wfs, n_qubits)

    meas_alloc = get_meas_alloc_update(measured_groups, var, n_qubits, alpha=0.001)

    return variance, meas_alloc, var, fixed_grp_coefficients, measured_groups


def var_avg(n_qubits):
    return 1.0 - 1.0 / (2**n_qubits + 1)


def get_meas_alloc_update(groups, variances, n_qubits, alpha=0.01):

    avg_vars = np.zeros(len(groups))
    for i, group in enumerate(groups):
        c_sq = 0.0
        for pw in group:
            c_sq += np.abs(get_pauli_word_coefficient(pw)) ** 2
        avg_vars[i] = c_sq * var_avg(n_qubits)

    sqrt_vars = np.abs(
        np.sqrt(np.array(alpha * avg_vars + (1.0 - alpha) * np.array(variances)))
    )
    return sqrt_vars / np.sum(sqrt_vars)


def get_measurement_variance_simple(groupings, wfs, n_qubits):
    """Compute the measurement variance given the number of measurement allocated to each group.
    This should work for both overlapping and non-overlapping measurement.
    Args:
        groupings (List[List[QubitOperator]]): List of groups, where each group is a list of single pauli-word.
        group_measurements (List[float]): List of measurements for each group.
            For flexibility numbers of measurement are allowed to be positive float.
        wfs (ndarray): The wavefunction to compute variance with. Obtained from openfermion.
        n_qubits (int): The number of qubits
        ev_dict (Dict[tuple, float]): The dictionary where ev_dict[pw_tuple] = expectation(pw) given specified wavefunction.

    Returns:
        v (float): The measurement variance following tex file under writeup/ (eps^2 in the text).
    """

    # Compute measurement variance
    var = []

    for grp_idx, group in enumerate(groupings):
        var_val = variance_of_group(group, wfs, n_qubits)
        var.append(var_val)

    sqrt_var = np.sqrt(np.abs(np.real_if_close(np.array(var))))
    variance = np.sum(sqrt_var) ** 2
    return var, variance


def get_meas_alloc(original_decomp, N, psi):
    n_frag = len(original_decomp)
    vars = np.zeros(n_frag, dtype=np.complex128)
    sqrt_vars = np.zeros(n_frag, dtype=np.complex128)
    for i, frag in enumerate(original_decomp):
        vars[i] = variance(gso(frag, N), psi)
        sqrt_vars[i] = np.sqrt(vars[i])

    total = sum(sqrt_vars).real
    return [np.sqrt(vars[i].real) / total for i in range(n_frag)]


# def CS_heuristic_mol(initial_meas_alloc, decomp, sharable_paulis_list, sharable_paulis_fixed_list, sharable_only_no_fixed_decomp, pw_grp_idxes_no_fix, pw_indices, coeff_map, new_grp_idx_start, n_qubits, wfs, n_cycles = 5):
#     """
#     Optimises coefficient split by iterating 1) fixing measurement allocation and optimising coefficient split then 2) fixing resulting CS and optimising MA.
#     Args:
#         initial_meas_alloc (List[float]): List of measurements for each group.
#             For flexibility numbers of measurement are allowed to be positive float.
#         overlapped_group (List[List[QubitOperator]]): Overlapped Grouping.
#         pw_list(List[QubitOperator]): List of all pauli words in multiple measurement groups.
#         pw_grp_idxs (List[List[int]]): List of list of corresponding group indexes that each Pauli word appears in.
#         fixed_grp: List[int]: List of the fixed group index of each Pauli, corresponding to pw_list[idx].
#         new_overlapped_group (List[List[QubitOperator]]): Overlapped Grouping with the fixed group Pauli removed (last group that the Pauli appears in).
#         new_pw_grp_idxs (List[List[int]]): List of list of corresponding group indexes that each Pauli word appears in new_overlapped_group.
#         new_grp_len_list (List[int]): List of length of each group in new_overlapped_group.
#         new_grp_idx_start (List[int]): List of the starting index of each group (in the coefficient vector).
#         n_qubits (int): Number of qubits of the Hamiltonian.
#         ev_dict (Dict[tuple, float]): The dictionary where ev_dict[pw_tuple] = expectation(pw) given specified wavefunction.
#         ev_dict_ref (Dict[tuple, float]): The dictionary where ev_dict[pw_tuple] = expectation(pw) given a reference wavefunction (HF, CISD, FCI)
#         n_cycles (int): Number of times process 1) and 2) is carried out (each).
#         return_coeff (Bool): If True, returns the final coefficient split in optimisation.
#         initial_CS (List[float]): Starting CS to initiate ICS from.
#         save_CS (Bool): Save coefficient split at each iteration if True.
#     Returns:
#         var (Dict[int]): Dictionary that gives variance at each stage of the iterative process.
#         (actual_var (Float): If ev_dict_ref is used, this calculates the true FCI variance of the calculated CS.)
#         (coeff (np.array): Final coefficient split.)
#         grp_var (List[float]): Variance of all groups in a list.
#         fixed_grp_coefficients (List[float]): Fixed group coefficients in order of pw_list.
#     """
#     meas_alloc = {}
#     meas_alloc[0] = initial_meas_alloc
#     measured_groups = {}
#     measured_groups[0] = decomp
#     var = np.zeros(n_cycles)
#
#     for i in range(n_cycles):
#         matrix, b = optimize_coeffs(pw_grp_idxes_fix,
#             pw_grp_idxes_no_fix_len,
#             meas_alloc[i],
#             sharable_only_decomp,
#             sharable_only_no_fixed_decomp,
#             pw_indices, psi, N, measured_groups[i], alpha=0.001)
#         print('matrix ready')
#         sol = np.linalg.lstsq(matrix, b.T, rcond=None)
#         x0 = sol[0]
#         coeff = x0.T[0]
#
#         # variance, meas_alloc, var, fixed_grp_coefficients, measured_groups
#         var[i], meas_alloc[i+1], grp_var, fixed_grp_coefficients, measured_groups[i+1] \
#             = get_split_measurement_variance_unconstrained(
#             coeff,
#             decomp,
#             sharable_paulis_list,
#             sharable_paulis_fixed_list,
#             sharable_only_no_fixed_decomp,
#             pw_grp_idxes_no_fix,
#             new_grp_idx_start,
#             coeff_map,
#             n_qubits,
#             wfs
#         )
#
#         for jjj in range(len(meas_alloc[i+1])):
#             if meas_alloc[i+1][jjj] < 1e-8:
#                 meas_alloc[i+1][jjj] = 1e-8
#
#         meas_alloc[i+1] = meas_alloc[i+1]/np.sum(meas_alloc[i+1])
#
#         print(f"Variance at cycle {i}: {var[i]}")
#         if i > 0 and (var[i] >= var[i-1] or np.abs(var[i] - var[i-1])/np.abs(var[i-1]) <= 1e-2):
#             for index in range(i+1, len(var)):
#                 var[index] = var[i]
#             break
#
#     minarg = np.argmin(np.abs(var))
#     varrrr, varianceeee = get_measurement_variance_simple(measured_groups[minarg], wfs, n_qubits)
#     return var[minarg], coeff, grp_var, fixed_grp_coefficients, measured_groups[minarg]
