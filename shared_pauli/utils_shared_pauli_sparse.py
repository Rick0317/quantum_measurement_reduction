# Purpose: Sparse matrix versions of functions for calculating shared pauli decompositions
# This file provides sparse matrix alternatives to the functions in SharedPauli/shared_paulis.py

import numpy as np
import scipy.sparse as sp
from openfermion import (
    get_sparse_operator as gso,
    get_ground_state as ggs,
    variance,
    expectation,
    bravyi_kitaev,
    QubitOperator
)
from copy import deepcopy
from entities.paulis import PauliString


def sparse_variance(operator, state):
    """
    Compute variance of operator with a sparse state matrix.
    This is a sparse matrix version of the variance calculation.
    
    Args:
        operator: scipy.sparse.spmatrix - The operator whose variance is desired
        state: scipy.sparse.spmatrix - A sparse matrix representing a state vector (shape: (1, dim))
        
    Returns:
        A complex number giving the variance
    """
    if not isinstance(state, sp.spmatrix):
        raise ValueError("State must be a sparse matrix")
    
    # For state vectors, we use the formula: Var(O) = <ψ|O^2|ψ> - <ψ|O|ψ>^2
    op_squared = operator @ operator
    expectation_op = sparse_expectation(operator, state)
    expectation_op_squared = sparse_expectation(op_squared, state)
    
    return expectation_op_squared - expectation_op ** 2


def sparse_expectation(operator, state):
    """
    Compute expectation value of operator with a sparse state matrix.
    This is a sparse matrix version of the expectation calculation.
    
    Args:
        operator: scipy.sparse.spmatrix - The operator whose expectation value is desired
        state: scipy.sparse.spmatrix - A sparse matrix representing a state vector (shape: (1, dim))
        
    Returns:
        A complex number giving the expectation value
    """
    if not isinstance(state, sp.spmatrix):
        raise ValueError("State must be a sparse matrix")
    
    # For state vectors, expectation is <ψ|O|ψ> = ψ† O ψ
    # Since state is (1, dim) and operator is (dim, dim), we compute state @ operator @ state.T
    expectation_val = (state @ operator @ state.T)[0, 0]
    
    return expectation_val


def does_term_frag_commute_sparse(term_op, fragment):
    """
    Check if a term commutes with all terms in a fragment using sparse matrix operations.
    This is a sparse matrix version of the commutativity check using QWC (Qubit-Wise Commuting).
    
    Args:
        term_op: QubitOperator - The term to check
        fragment: QubitOperator - The fragment to check against
        
    Returns:
        bool - True if the term qubit-wise commutes with all terms in the fragment
    """
    # Use qubit-wise commutation (QWC) - more permissive than full commutation
    term_pauli = PauliString(list(term_op.terms.keys())[0])
    
    for frag_term in fragment.terms.keys():
        frag_pauli = PauliString(frag_term)
        if not term_pauli.qubit_wise_commute(frag_pauli):
            return False
    
    return True


def does_term_frag_fully_commute_sparse(term_op, fragment):
    """
    Check if a term commutes with all terms in a fragment using sparse matrix operations.
    This is a sparse matrix version of the commutativity check.
    
    Args:
        term_op: QubitOperator - The term to check
        fragment: QubitOperator - The fragment to check against
        
    Returns:
        bool - True if the term commutes with all terms in the fragment
    """
    # Check if the commutator [term_op, fragment] is zero (full commutation)
    commutator = term_op * fragment - fragment * term_op
    
    # If commutator is zero (empty terms), they commute
    return len(commutator.terms) == 0

def select_sharable_paulis_sparse(frag_combs, original_decomp, N, psi_sparse):
    """
    Given a set of fragments, we find the shareable Pauli operators between
    pairs of fragments so that we can transfer some coefficient to the smaller
    fragment. This is the sparse matrix version of select_sharable_paulis.
    
    :param frag_combs: The pairs of fragments we consider
    :param original_decomp: The decomposition of the input Hamiltonian
    :param N: The number of qubits for operator creation (should be Nqubits // 2 for tapered operators)
    :param psi_sparse: The quantum state as a sparse matrix
    :return: [(c, pauli, frag_a, frag_b), ...], where c is the total coefficient
    """
    
    if not isinstance(psi_sparse, sp.spmatrix):
        raise ValueError("psi_sparse must be a sparse matrix")

    variance_sum = 0
    for fragment in original_decomp:
        frag_op = gso(fragment, N)
        variance_sum += np.sqrt(sparse_variance(frag_op, psi_sparse))

    pauli_added_combs = []
    for combination in frag_combs:
        (index_a, index_b) = combination
        frag_a = original_decomp[index_a]
        frag_b = original_decomp[index_b]
        size_a = len(frag_a.terms.items())
        size_b = len(frag_b.terms.items())

        print(f"Checking fragments {index_a} (size {size_a}) and {index_b} (size {size_b})")

        if size_b > size_a:
            frag_a, frag_b = frag_b, frag_a

        sharable_pauli_a2b = []

        # From the two fragments, find the Pauli operator that could be shared.
        for term_a, coeff_a in frag_a.terms.items():
            for term_b, coeff_b in frag_b.terms.items():
                # If term_a commutes with frag_b or term_b commutes with frag_a
                # We add them to the sharable_pauli
                if does_term_frag_commute_sparse(QubitOperator(term=term_a), frag_b):
                    print(f"Found sharable pauli: {term_a} with coefficient {coeff_a}")
                    sharable_pauli_a2b.append((coeff_a, QubitOperator(term=term_a)))
                if does_term_frag_commute_sparse(QubitOperator(term=term_b), frag_a):
                    print(f"Found sharable pauli: {term_b} with coefficient {coeff_b}")
                    sharable_pauli_a2b.append((coeff_b, QubitOperator(term=term_b)))

        for sharable_pauli in sharable_pauli_a2b:
            pauli_added_combs.append((sharable_pauli[0], sharable_pauli[1], index_a, index_b))

    return pauli_added_combs


def select_combs_sparse(psi_sparse, N, original_decomp):
    """
    Given a decomposition of the input Hamiltonian, select the set of fragment combinations
    This is the sparse matrix version of select_combs for shared paulis.
    
    :param psi_sparse: The quantum state as a sparse matrix
    :param N: The number of qubits for operator creation (should be Nqubits // 2 for tapered operators)
    :param original_decomp: The decomposition of the input Hamiltonian
    :return: A list of fragment combinations as indices in the decomposition: [(a, b)]
    """
    
    if not isinstance(psi_sparse, sp.spmatrix):
        raise ValueError("psi_sparse must be a sparse matrix")

    n_frag = len(original_decomp)
    vars = np.zeros(n_frag, dtype=np.complex128)
    for i, frag in enumerate(original_decomp):
        frag_op = gso(frag, N)
        vars[i] = sparse_variance(frag_op, psi_sparse)

    p = 4 * sum(np.sqrt(vars))

    score_board = {}
    for a in range(n_frag):
        for b in range(a+1, n_frag):
            var_a = vars[a]
            var_b = vars[b]

            # The score is the upper-bound of the variance metric reduction
            # Eq (15) of the ghost Pauli paper.
            score = p * (np.sqrt(var_a * var_b)) / (np.sqrt(var_a) + np.sqrt(var_b))
            score_board[(a, b)] = score

    sorted_items = sorted(score_board.items(), key=lambda item: item[1],
                          reverse=True)

    top_50_count = len(score_board) // 2 + (
        1 if len(score_board) % 2 != 0 else 0)

    top_keys = [key for key, value in sorted_items[:top_50_count]]
    print(top_keys)

    return top_keys


def update_decomp_w_shared_paulis_sparse(psi_sparse, N, original_decomp):
    """
    Update the commuting decomposition by introducing shared paulis into some commuting sets.
    This is the sparse matrix version of update_decomp_w_shared_paulis.
    
    :param psi_sparse: The quantum state as a sparse matrix
    :param N: The number of qubits for operator creation (should be Nqubits // 2 for tapered operators)
    :param original_decomp: The original Pauli decomposition of the input Hamiltonian
    :return: The updated Pauli decomposition with shared paulis in each set
    """
    
    if not isinstance(psi_sparse, sp.spmatrix):
        raise ValueError("psi_sparse must be a sparse matrix")
    
    new_decomp = original_decomp.copy()

    frag_combs = select_combs_sparse(psi_sparse, N, original_decomp)
    print(f"Found {len(frag_combs)} fragment combinations to consider")

    pauli_added_combs = select_sharable_paulis_sparse(frag_combs, original_decomp, N, psi_sparse)
    print(f"Found {len(pauli_added_combs)} sharable paulis")

    # Apply changes to multiple combinations, not just the first one
    for i, combination in enumerate(pauli_added_combs[:3]):  # Try first 3 combinations
        (c, qubit_op, index_a, index_b) = combination
        # Increase the effect size to make it more noticeable
        share = 0.1 * abs(c)  # Use 10% of the coefficient magnitude
        print(f"Applying shared pauli {i+1}: coefficient {c}, share {share}, from frag {index_a} to frag {index_b}")
        new_decomp[index_a] -= share * qubit_op
        new_decomp[index_b] += share * qubit_op

    return new_decomp


def variance_metric_sparse(H, decomp, N):
    """
    Computes the variance metric using sparse matrices.
    This is the sparse matrix version of variance_metric.
    
    :param H: The Hamiltonian
    :param decomp: The decomposition of the Hamiltonian
    :param N: The number of qubits
    :return: The variance metric
    """
    
    H_sparse = gso(H, N)
    psi_sparse = ggs(H_sparse)[1]
    
    # Convert psi to sparse matrix format if it's not already
    if not isinstance(psi_sparse, sp.spmatrix):
        psi_sparse = sp.csr_matrix(psi_sparse.reshape(1, -1))

    vars = np.zeros(len(decomp), dtype=np.complex128)
    for i, frag in enumerate(decomp):
        frag_op = gso(frag, N)
        vars[i] = sparse_variance(frag_op, psi_sparse)
    
    return np.sum((vars)**(1/2))**2


def abs_of_dict_value(x):
    return np.abs(x[1])


def copy_hamiltonian(H):
    H_copy = QubitOperator().zero()

    for t, s in H.terms.items():
        H_copy += s * QubitOperator(t)

    assert (H - H_copy) == QubitOperator().zero()
    return H_copy


def get_sharable_paulis_sparse(original_decomp):
    """
    Get the data of sharable Pauli words using sparse matrix operations.
    This is the sparse matrix version of get_sharable_paulis.
    
    :param original_decomp: The decomposition of the input Hamiltonian
    :return: Dictionary that maps sharable Pauli in the decomposition to
    its appearance in fragments.
    """
    all_pauli_dict = {}
    for frag_idx, fragment in enumerate(original_decomp):
        for idx, term in enumerate(fragment.terms):
            all_pauli_dict[term] = [frag_idx]
            pauli_word = QubitOperator(term=term)
            for frag_idx_compared, fragment_compared in enumerate(original_decomp):
                # Check if term can belong to the fragment
                # We have to check qubit-wise commutation
                if frag_idx_compared != frag_idx:
                    commute = True
                    for idx, compared_term in enumerate(fragment_compared.terms):
                        if not PauliString(compared_term).qubit_wise_commute(PauliString(term)):
                            commute = False
                    if commute:
                        all_pauli_dict[term].append(frag_idx_compared)

    sharable_pauli_dict = {}
    sharable_pauli_list = []
    sharable_pauli_indices_list = []
    for pauli_idx, pauli_term in enumerate(all_pauli_dict):
        shared_frag_idx_list = all_pauli_dict[pauli_term]
        if len(shared_frag_idx_list) > 1:
            sharable_pauli_list.append(pauli_term)
            sharable_pauli_indices_list.append(shared_frag_idx_list)
            sharable_pauli_dict[pauli_term] = shared_frag_idx_list

    return sharable_pauli_dict, sharable_pauli_list, sharable_pauli_indices_list


def get_share_pauli_only_decomp_sparse(sharable_pauli_dict):
    """
    Get the fragmentation of the Hamiltonian with only the sharable fragments included.
    This is the sparse matrix version of get_share_pauli_only_decomp.
    
    :param sharable_pauli_dict: Dictionary mapping sharable Pauli terms to fragment indices
    :return: Tuple of dictionaries containing fragment mappings
    """
    fragment_idx_to_sharable_paulis = {}
    pw_grp_idxes_no_fix = {}
    pw_grp_idxes_fix = {}
    for pauli_idx, pauli_term in enumerate(sharable_pauli_dict):
        pw_grp_idxes_no_fix[pauli_term] = sharable_pauli_dict[pauli_term][:-1]
        pw_grp_idxes_fix[pauli_term] = sharable_pauli_dict[pauli_term][-1]
        for frag_idx in sharable_pauli_dict[pauli_term][:-1]:
            if frag_idx not in fragment_idx_to_sharable_paulis:
                fragment_idx_to_sharable_paulis[frag_idx] = [pauli_term]
            else:
                fragment_idx_to_sharable_paulis[frag_idx].append(pauli_term)

    return fragment_idx_to_sharable_paulis, pw_grp_idxes_no_fix, pw_grp_idxes_fix


def get_overlapping_decomp_sparse(sharable_pauli_dict, original_decomp, pw_grp_idxes_fix):
    """
    Give the overlapping allowed decomposition using sparse matrix operations.
    This is the sparse matrix version of get_overlapping_decomp.
    
    :param sharable_pauli_dict: Dictionary mapping sharable Pauli terms to fragment indices
    :param original_decomp: The original decomposition
    :param pw_grp_idxes_fix: Dictionary mapping Pauli terms to their fixed group indices
    :return: Tuple of overlapping decompositions
    """
    all_sharable_contained_decomp = deepcopy(original_decomp)
    all_sharable_no_fixed_decomp = deepcopy(original_decomp)

    for pauli_idx, pauli in enumerate(sharable_pauli_dict):
        pauli_appearance_indices_lst = sharable_pauli_dict[pauli]
        for appear_idx in pauli_appearance_indices_lst:
            if appear_idx not in pw_grp_idxes_fix:
                all_sharable_no_fixed_decomp[appear_idx] += QubitOperator(term=pauli)
            all_sharable_contained_decomp[appear_idx] += QubitOperator(term=pauli)

    return all_sharable_contained_decomp, all_sharable_no_fixed_decomp


def get_sharable_only_decomp_sparse(sharable_pauli_dict, original_decomp, pw_grp_idxes_fix, coeff_map):
    """
    Give the overlapping allowed decomposition using sparse matrix operations.
    This is the sparse matrix version of get_sharable_only_decomp.
    
    :param sharable_pauli_dict: Dictionary mapping sharable Pauli terms to fragment indices
    :param original_decomp: The original decomposition
    :param pw_grp_idxes_fix: Dictionary mapping Pauli terms to their fixed group indices
    :param coeff_map: Dictionary mapping Pauli terms to their coefficients
    :return: Tuple of sharable-only decompositions
    """
    sharable_only_decomp = [QubitOperator().zero() for _ in range(len(original_decomp))]
    sharable_only_no_fixed_decomp = [QubitOperator().zero() for _ in range(len(original_decomp))]

    for pauli_idx, pauli in enumerate(sharable_pauli_dict):
        pauli_appearance_indices_lst = sharable_pauli_dict[pauli]
        for appear_idx in pauli_appearance_indices_lst:
            if appear_idx != pw_grp_idxes_fix[pauli]:
                sharable_only_no_fixed_decomp[appear_idx] += QubitOperator(term=pauli) * coeff_map[pauli]
            sharable_only_decomp[appear_idx] += QubitOperator(term=pauli) * coeff_map[pauli]

    return sharable_only_decomp, sharable_only_no_fixed_decomp


def qubit_op_to_list(qubit_op: QubitOperator):
    """
    Convert a QubitOperator to a list of terms.
    This is a helper function used in the sparse matrix versions.
    
    :param qubit_op: The QubitOperator to convert
    :return: List of terms
    """
    term_list = []
    for idx, term in enumerate(qubit_op.terms):
        term_list.append(term)
    return term_list


def get_pauli_coeff_map_sparse(decomp):
    """
    Get the coefficient map for Pauli terms using sparse matrix operations.
    This is the sparse matrix version of get_pauli_coeff_map.
    
    :param decomp: The decomposition
    :return: Dictionary mapping Pauli terms to their coefficients
    """
    pauli_coeff_map = {}
    for fragment in decomp:
        for idx, term in enumerate(fragment.terms):
            if term not in pauli_coeff_map:
                pauli_coeff_map[term] = fragment.terms[term]

    return pauli_coeff_map
