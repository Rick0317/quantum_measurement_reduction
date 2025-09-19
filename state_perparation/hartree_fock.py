from openfermion import FermionOperator, bravyi_kitaev
from openfermion import get_sparse_operator as gso

from state_perparation.reference_state_utils import *


def get_bk_hf_state(n_qubits, n_occ):
    """
    Given the number of qubits = spin-orbitals and the number of occupations,
    return the Hartree-Fock state as 2^n_qubits state.
    :param n_qubits:
    :param n_occ:
    :return:
    """
    state_string = ""
    for i in range(n_qubits - n_occ):
        state_string += "0"
    for i in range(n_occ):
        state_string += "1"
    bk_basis_state = get_bk_basis_states(state_string, n_qubits)
    index = find_index(bk_basis_state)
    wfs = np.zeros(2**n_qubits)
    wfs[index] = 1
    return wfs


if __name__ == "__main__":
    n_qubits = 4
    n_occ = 2
    hartree_fock_state = get_bk_hf_state(n_qubits, n_occ)

    op_part = bravyi_kitaev(FermionOperator(((2, 1), (2, 0))))
    op_matrix = gso(op_part, n_qubits).toarray()
    identity = np.eye(2**n_qubits)
    zeros = np.zeros((2**n_qubits, 2**n_qubits))
    killer = op_matrix - identity

    print(killer @ hartree_fock_state)
