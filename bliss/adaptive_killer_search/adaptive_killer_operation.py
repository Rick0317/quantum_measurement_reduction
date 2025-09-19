from typing import List
from openfermion import FermionOperator, normal_ordered


def get_adaptive_killer(anti_hermitian_list: List, previous_killers: List, n_qubits: int):
    """
    Given the list of anti-Hermitian generators that have been used
    find the killer operator that can be used at this step.
    :param anti_unitary_list: The list of anti-Hermitian generators being used.
    :param previous_killers: The list of killers used in the previous step
    :param n_qubits: The number of qubits
    :return:
    """
    # First, find the 'untouched' qubit registers and apply the corresponding
    # (ne - occ), a^_i, and a_i killers.
    pass

    # Then, we apply killers to the 'touched' qubit registers.
    # There are two steps:
    # 1. Know which (SDs)_q are generated from HF.

    # 2. Find the coefficients in front of each SD_q


if __name__ == '__main__':
    test = FermionOperator("3^ 2") - FermionOperator("2^ 3")
    for term, coeff in test.terms.items():
        print(term[-1])
