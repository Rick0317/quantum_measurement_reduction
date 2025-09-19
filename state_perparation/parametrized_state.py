"""
This file defines parametrized states that starts from Hartree-Fock

I think the orders of occupied and virtuals are flipped by the bk transformation
"""

import numpy as np
from openfermion import (
    FermionOperator,
    bravyi_kitaev,
    expectation,
)
from openfermion import get_sparse_operator as gso

from state_perparation.hartree_fock import get_bk_hf_state
from unitary_preparation.generalized_fermionic_generator import one_body_unitary


def get_single_unitary_parametered_state(j: int, b: int, n_qubits, n_occ):
    """
    Given the number of qubits = spin-orbitals and the number of occupations,
    return the Hartree-Fock state as 2^n_qubits state.
    :param n_qubits:
    :param n_occ:
    :return:
    """
    hf_state = get_bk_hf_state(n_qubits, n_occ)
    unitary_ia_matrix = one_body_unitary(j, b, n_qubits)
    v = unitary_ia_matrix @ hf_state

    # Compute the norm
    norm_v = np.linalg.norm(v)
    return v / norm_v


if __name__ == "__main__":
    N = 4
    Ne = 2
    hf_state = get_bk_hf_state(N, Ne)
    parametrized = get_single_unitary_parametered_state(2, 0, N, Ne)
    number_operator = FermionOperator()
    for mode in range(N):
        number_operator += FermionOperator(f"{mode}^ {mode}")  # a_i† a_i

    # Convert to qubit operator using Jordan-Wigner transformation
    number_operator_qubit = bravyi_kitaev(number_operator)

    print(
        f"Particle number Qubit: {expectation(gso(number_operator_qubit), parametrized)}"
    )

    print(parametrized)
    print(f"HF: {hf_state}")
    cos_theta = parametrized[2]
    sin_theta = parametrized[12]

    print(f"Cos Theta: {cos_theta}")
    print(f"Sin Theta: {sin_theta}")

    killer = (cos_theta / sin_theta) * FermionOperator("2^ 0") + (
        sin_theta / cos_theta
    ) * FermionOperator("0^ 2")
    print(f"Killer: {killer}")
    killer_matrix = gso(bravyi_kitaev(killer), N).toarray() - np.eye(2**N)
    applied = killer_matrix @ parametrized
    print(f"Killer matrix: {applied}")

    term_1 = gso(bravyi_kitaev(FermionOperator("2^ 0")), N).toarray()
    term_2 = gso(bravyi_kitaev(FermionOperator("0^ 2")), N).toarray()

    applied_1 = term_1 @ parametrized

    applied_2 = term_2 @ parametrized
    print((cos_theta / sin_theta) * applied_1)
    print((sin_theta / cos_theta) * applied_2)
