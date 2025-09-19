# Purpose: Sparse matrix versions of functions for calculating ghost pauli decompositions
# This file provides sparse matrix alternatives to the functions in utils_ghost_pauli.py

from itertools import product

import numpy as np
import scipy.sparse as sp
from openfermion import (
    QubitOperator,
    bravyi_kitaev,
    expectation,
)
from openfermion import get_ground_state as ggs
from openfermion import get_sparse_operator as gso
from openfermion import (
    variance,
)

from entities.paulis import PauliString, pauli_ops_to_qop
from symplectic_vector_space.space_F_definition import SpaceFVector, vector_2_pauli


def is_z_string(pauli_product):
    """Check if a Pauli product is a Z-string."""
    for pauli in pauli_product:
        if pauli not in {"Z"}:
            return False
    return True


def multiply_pauli_terms(term1, term2):
    """Multiply two Pauli terms and return the result."""
    result = {}
    phase = 1  # Track the phase (1, -1, i, -i)

    # Merge the two terms
    all_qubits = set(term1.keys()).union(set(term2.keys()))
    for qubit in all_qubits:
        op1 = term1.get(qubit, "I")
        op2 = term2.get(qubit, "I")

        # Multiply the Pauli operators
        if op1 == "I":
            result[qubit] = op2
        elif op2 == "I":
            result[qubit] = op1
        elif op1 == op2:
            result[qubit] = "I"
        elif op1 == "X" and op2 == "Y":
            result[qubit] = "Z"
            phase *= 1j
        elif op1 == "Y" and op2 == "X":
            result[qubit] = "Z"
            phase *= -1j
        elif op1 == "X" and op2 == "Z":
            result[qubit] = "Y"
            phase *= -1j
        elif op1 == "Z" and op2 == "X":
            result[qubit] = "Y"
            phase *= 1j
        elif op1 == "Y" and op2 == "Z":
            result[qubit] = "X"
            phase *= 1j
        elif op1 == "Z" and op2 == "Y":
            result[qubit] = "X"
            phase *= -1j

    return result, phase


def find_z_string_combination(pauli_pool: list[QubitOperator], pauli):
    """Find a Pauli product in `fragment` such that when multiplied by `pauli`, the result is a Z-string."""
    for fragment in pauli_pool:
        term, coeff = list(fragment.terms.items())[0]
        # Multiply the term with `pauli`
        product_term, phase = multiply_pauli_terms(
            dict(term), dict(list(pauli.terms.items())[0][0])
        )

        # Check if the result is a Z-string
        if is_z_string(product_term):
            return True
    return False


def matrix_J(n):
    # Create an n x n zero matrix
    zeros = np.zeros((n, n), dtype=int)
    # Create an n x n identity matrix
    identity = np.eye(n, dtype=int)

    # Combine the blocks into a 2n x 2n matrix
    top = np.hstack((zeros, identity))
    bottom = np.hstack((identity, zeros))
    matrix = np.vstack((top, bottom))

    return matrix


def exclude_paulis(pauli_pool: list[QubitOperator], pauli: QubitOperator):
    """
    If the fragment contains pauli or another pauli product that multiplied together to get z-string, we exclude it.
    :param fragment:
    :param pauli:
    :return: bool
    """
    key_list = [list(fragment.terms.keys())[0] for fragment in pauli_pool]
    pauli_key = list(pauli.terms.keys())[0]
    # We'll not add pauli if it already exists in
    if pauli_key in key_list:
        return True

    # Check if pauli makes a z-string by multiplying any of the existing paulis
    if find_z_string_combination(pauli_pool, pauli):
        return True

    return False


def check_commutativity(frag: QubitOperator, pauli: QubitOperator):
    """
    Check if a Pauli product is a commuting set.
    :param frag:
    :param pauli:
    :return:
    """
    first_word = PauliString(list(frag.terms.keys())[0])
    second_word = PauliString(list(pauli.terms.keys())[0])
    return first_word.qubit_wise_commute(second_word)


def get_variance_reduction(c, d_of_pauli, var_of_pauli):
    """
    Get the amount of variance reduction as a result of adding pauli with coefficient c
    :param fragment:
    :param new_fragment:
    :param psi:
    :return:
    """
    first_term = 2 * c * d_of_pauli
    second_term = c**2 * var_of_pauli
    return first_term - second_term


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

    return expectation_op_squared - expectation_op**2


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


def select_paulis_sparse(frag_combs, original_decomp, N, psi_sparse):
    """
    Given pairs of Hamiltonian fragments, this code finds the Ghost Pauli
    as well as the corresponding coefficients to be added to each fragment
    This is the sparse matrix version of select_paulis.

    :param frag_combs: The indices pairs of the Hamiltonian fragments
    :param original_decomp: The decomposition of the input Hamiltonian
    :param N: The number of qubits for operator creation (should be Nqubits // 2 for tapered operators)
    :param psi_sparse: The quantum state as a sparse matrix
    :return: [(c, pauli, frag_a, frag_b), ...]
    """

    if not isinstance(psi_sparse, sp.spmatrix):
        raise ValueError("psi_sparse must be a sparse matrix")

    variance_sum = 0
    for fragment in original_decomp:
        frag_op = gso(fragment, N)
        variance_sum += np.sqrt(sparse_variance(frag_op, psi_sparse))

    pauli_added_combs = []
    counter = 0
    for combination in frag_combs:
        counter += 1
        print(f"Started {counter}-th fragment combination")
        (index_a, index_b) = combination
        frag_a = original_decomp[index_a]
        frag_b = original_decomp[index_b]

        # Get the matrix in the linear symplectic vector space F
        matrix_M = [
            SpaceFVector(PauliString(term), N).get_vector() for term in frag_a.terms
        ]
        matrix_M += [
            SpaceFVector(PauliString(term), N).get_vector() for term in frag_b.terms
        ]

        exe_matrix_M = np.array(matrix_M)
        n_cols = exe_matrix_M.shape[1]

        # Filter binary vectors in the null space of exe_matrix_M
        null_space_vectors = [
            np.array(vec)
            for vec in product([1, 0], repeat=n_cols)
            if np.all(np.dot(exe_matrix_M, vec) % 2 == 0)
        ]

        counter = 1
        for vec in null_space_vectors:
            pauli_op = vector_2_pauli(matrix_J(N) @ vec, N)
            qubit_op = pauli_ops_to_qop(pauli_op.get_pauli_ops())
            qubit_op_sparse = gso(qubit_op, N)
            var_psi_pauli = sparse_variance(qubit_op_sparse, psi_sparse)

            if var_psi_pauli > 0.9:
                frag_a_sparse = gso(frag_a, N)
                frag_b_sparse = gso(frag_b, N)
                var_psi_ha = np.sqrt(sparse_variance(frag_a_sparse, psi_sparse))
                var_psi_hb = np.sqrt(sparse_variance(frag_b_sparse, psi_sparse))
                m_a, m_b = var_psi_ha / variance_sum, var_psi_hb / variance_sum
                mu = m_a * m_b / (m_a + m_b)

                cov_a_pauli = sparse_expectation(
                    frag_a_sparse @ qubit_op_sparse, psi_sparse
                ) - sparse_expectation(frag_a_sparse, psi_sparse) * sparse_expectation(
                    qubit_op_sparse, psi_sparse
                )
                cov_b_pauli = sparse_expectation(
                    frag_b_sparse @ qubit_op_sparse, psi_sparse
                ) - sparse_expectation(frag_b_sparse, psi_sparse) * sparse_expectation(
                    qubit_op_sparse, psi_sparse
                )

                d_of_pauli = (m_a * cov_b_pauli - m_b * cov_a_pauli) / (m_a + m_b)
                c = d_of_pauli / var_psi_pauli
                variance_reduction = (
                    get_variance_reduction(c, d_of_pauli, var_psi_pauli) / mu
                )

                if variance_reduction > 1e-5:
                    print(f"Variance reduction: {variance_reduction}")
                    pauli_added_combs.append((c, qubit_op, index_a, index_b))

        del (
            qubit_op,
            pauli_op,
            frag_a,
            frag_b,
            matrix_M,
            exe_matrix_M,
            null_space_vectors,
        )

    return pauli_added_combs


def select_combs_sparse(psi_sparse, N, original_decomp):
    """
    Given a decomposition of the input Hamiltonian, select the set of fragment combinations
    This is the sparse matrix version of select_combs.

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
        for b in range(a + 1, n_frag):
            var_a = vars[a]
            var_b = vars[b]

            # The score is the upper-bound of the variance metric reduction
            # Eq (15) of the ghost Pauli paper.
            score = p * (np.sqrt(var_a * var_b)) / (np.sqrt(var_a) + np.sqrt(var_b))
            score_board[(a, b)] = score

    sorted_items = sorted(score_board.items(), key=lambda item: item[1], reverse=True)

    top_50_count = len(score_board) // 4 + (1 if len(score_board) % 2 != 0 else 0)

    top_keys = [key for key, value in sorted_items[:50]]
    print(top_keys)

    return top_keys


def update_decomp_w_ghost_paulis_sparse(psi_sparse, N, original_decomp):
    """
    Update the commuting decomposition by introducing ghost paulis into some commuting sets.
    This is the sparse matrix version of update_decomp_w_ghost_paulis.

    :param psi_sparse: The quantum state as a sparse matrix
    :param N: The number of qubits for operator creation (should be Nqubits // 2 for tapered operators)
    :param original_decomp: The original Pauli decomposition of the input Hamiltonian
    :return: The updated Pauli decomposition with ghost paulis in each set
    """

    if not isinstance(psi_sparse, sp.spmatrix):
        raise ValueError("psi_sparse must be a sparse matrix")

    new_decomp = original_decomp.copy()

    frag_combs = select_combs_sparse(psi_sparse, N, original_decomp)

    pauli_added_combs = select_paulis_sparse(frag_combs, original_decomp, N, psi_sparse)

    pauli_list = []

    for combination in pauli_added_combs[:1]:
        (c, qubit_op, index_a, index_b) = combination
        pauli_is_excluded = exclude_paulis(pauli_list, qubit_op)
        pauli_list.append(qubit_op)
        if not pauli_is_excluded:
            new_decomp[index_a] += c.real * qubit_op
            new_decomp[index_b] -= c.real * qubit_op

    return new_decomp


def commutator_variance_sparse(psi_sparse, decomp, N):
    """
    Computes the variance of the [H, G] - K using sparse matrices.
    This is the sparse matrix version of commutator_variance.

    :param psi_sparse: The quantum state as a sparse matrix
    :param decomp: The decomposition of the Hamiltonian
    :param N: The number of sites
    :return: The variance metric
    """

    if not isinstance(psi_sparse, sp.spmatrix):
        raise ValueError("psi_sparse must be a sparse matrix")

    total_var = 0
    for i, frag in enumerate(decomp):
        frag_op = gso(frag, N)
        total_var += np.sqrt(sparse_variance(frag_op, psi_sparse))

    vars = np.zeros(len(decomp), dtype=np.complex128)
    for i, frag in enumerate(decomp):
        frag_op = gso(frag, N)
        m_a = np.sqrt(sparse_variance(frag_op, psi_sparse)) / total_var
        vars[i] = sparse_variance(frag_op, psi_sparse) / m_a

    return np.sum(vars)


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

    return np.sum((vars) ** (1 / 2)) ** 2


def abs_of_dict_value(x):
    return np.abs(x[1])


def copy_hamiltonian(H):
    H_copy = QubitOperator().zero()

    for t, s in H.terms.items():
        H_copy += s * QubitOperator(t)

    assert (H - H_copy) == QubitOperator().zero()
    return H_copy
