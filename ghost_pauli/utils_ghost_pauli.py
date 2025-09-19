from itertools import product

import numpy as np
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
            True
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


def update_decomp_w_ghost_paulis(psi, N, original_decomp):
    """
    Update the commuting decomposition by introducing ghost paulis into some commuting sets.
    :param original_decomp: The original Pauli decomposition of the input Hamiltonian
    :return: The updated Pauli decomposition with ghost paulis in each set
    """
    new_decomp = original_decomp.copy()

    # Selecting pairs of combination
    frag_combs = select_combs(psi, N, original_decomp)

    pauli_added_combs = select_paulis(frag_combs, original_decomp, N, psi)

    pauli_list = []

    for combination in pauli_added_combs[:1]:
        (c, qubit_op, index_a, index_b) = combination
        pauli_is_excluded = exclude_paulis(pauli_list, qubit_op)
        pauli_list.append(qubit_op)
        if not pauli_is_excluded:
            new_decomp[index_a] += c.real * qubit_op
            new_decomp[index_b] -= c.real * qubit_op

    return new_decomp


def check_commutativity(frag: QubitOperator, pauli: QubitOperator):
    """
    Check if a Pauli product is a commuting set.
    :param frag:
    :param pauli:
    :return:
    """
    first_word = PauliString(frag.terms.keys()[0])
    second_word = PauliString(pauli.terms.keys()[0])
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


def select_paulis(frag_combs, original_decomp, N, psi):
    """
    Given pairs of Hamiltonian fragments, this code finds the Ghost Pauli
    as well as the corresponding coefficients to be added to each fragment
    :param frag_combs: The indices pairs of the Hamiltonian fragments
    :param original_decomp: The decomposition of the input Hamiltonian
    :param N: The number of sites in the Hamiltonian
    :param psi: The quantum state found by CISD/ For experiments, we use the exact groundstate.
    :return: [(c, pauli, frag_a, frag_b), ...]
    """

    variance_sum = 0
    for fragement in original_decomp:
        variance_sum += np.sqrt(variance(gso(fragement, N), psi))

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
            var_psi_pauli = variance(gso(qubit_op, N), psi)

            if var_psi_pauli > 0.9:
                var_psi_ha = np.sqrt(variance(gso(frag_a, N), psi))
                var_psi_hb = np.sqrt(variance(gso(frag_b, N), psi))
                m_a, m_b = var_psi_ha / variance_sum, var_psi_hb / variance_sum
                mu = m_a * m_b / (m_a + m_b)

                cov_a_pauli = expectation(
                    gso(frag_a, N) * gso(qubit_op, N), psi
                ) - expectation(gso(frag_a, N), psi) * expectation(
                    gso(qubit_op, N), psi
                )
                cov_b_pauli = expectation(
                    gso(frag_b, N) * gso(qubit_op, N), psi
                ) - expectation(gso(frag_b, N), psi) * expectation(
                    gso(qubit_op, N), psi
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


def select_combs(psi, N, original_decomp):
    """
    Given a decomposition of the input Hamiltonian, select the set of fragment combinations
    :param original_decomp:
    :return: A list of fragment combinations as indices in the decomposition: [(a, b)]
    """

    n_frag = len(original_decomp)
    vars = np.zeros(n_frag, dtype=np.complex128)
    for i, frag in enumerate(original_decomp):
        vars[i] = variance(gso(frag, N), psi)

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

    top_keys = [key for key, value in sorted_items[:30]]
    print(top_keys)

    return top_keys


def commutator_variance(psi, decomp, N):
    """
    Computes the variance of the [H, G] - K.
    :param H: The Hamiltonian to compute the variance of.
    :param decomp: The decomposition of the Hamiltonian.
    :param G: The fermion operator to define the gradient
    :param K: The Killer operator applied to the whole [H, G]
    :param N: The number of sites
    :return: The variance metric
    """
    total_var = 0
    for i, frag in enumerate(decomp):
        total_var += np.sqrt(variance(gso(frag, N), psi))

    vars = np.zeros(len(decomp), dtype=np.complex128)
    for i, frag in enumerate(decomp):
        m_a = np.sqrt(variance(gso(frag, N), psi)) / total_var
        vars[i] = variance(gso(frag, N), psi) / m_a
    # return np.sum((vars) ** (1 / 2)) ** 2
    return np.sum(vars)


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


def load_hamiltonian(moltag):
    filename = f"../SolvableQubitHamiltonians/ham_lib/h4_sto-3g.pkl"
    with open(filename, "rb") as f:
        Hfer = pickle.load(f)
    Hqub = bravyi_kitaev(Hfer)
    Hqub -= Hqub.constant
    Hqub.compress()
    Hqub.terms = dict(sorted(Hqub.terms.items(), key=abs_of_dict_value, reverse=True))
    return Hqub
