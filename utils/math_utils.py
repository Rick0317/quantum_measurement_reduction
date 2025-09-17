import numpy as np
from openfermion import (expectation, QubitOperator,
                         get_sparse_operator as gso,
                         get_ground_state as ggs,
                         FermionOperator, variance)
import openfermion as of

def cov_frag_pauli(frag: QubitOperator, pauli_w: QubitOperator, psi):
    """
    Compute the covariance of the fragment and the Pauli word.
    :param frag:
    :param pauli_w:
    :param psi:
    :return:
    """
    return expectation(frag * pauli_w, psi) - expectation(frag, psi) * expectation(pauli_w, psi)


def cov_frag_pauli_iterative(frag: QubitOperator, pauli_w: QubitOperator, psi, n_qubits, alpha):
    """
    Compute the covariance of the fragment and the Pauli word.
    :param frag:
    :param pauli_w:
    :param psi:
    :return:
    """
    pauli_w_term = list(pauli_w.terms.keys())[0]
    sum_of_cov = 0.0
    for term, coeff in frag.terms.items():
        pauli_v = coeff * QubitOperator(term=term)
        sum_of_cov += (1. - alpha) * cov_pauli_pauli(gso(pauli_v, n_qubits), gso(pauli_w, n_qubits), psi)
        if term == pauli_w_term:
            sum_of_cov += alpha * coeff * var_avg(
                n_qubits)
    return sum_of_cov


def var_avg(n_qubits):
    return 1. - 1. / ( 2 ** n_qubits  + 1 )


def cov_frag_sum_pauli(term_list: list, pauli_w: QubitOperator, psi, n_qubits):
    """
    Compute the covariance of the fragment and the Pauli word.
    :param frag:
    :param pauli_w:
    :param psi:
    :return:
    """
    cov_sum = 0
    for term in term_list:
        cov_sum += cov_pauli_pauli(gso(QubitOperator(term), n_qubits=n_qubits), pauli_w, psi)
    return cov_sum



def cov_pauli_pauli(pauli_w_1, pauli_w_2, psi):
    """
    Compute the covariance of the two Pauli words.
    :param pauli_w_1:
    :param pauli_w_2:
    :param psi:
    :return:
    """
    return expectation(pauli_w_1 * pauli_w_2, psi) - expectation(pauli_w_1, psi) * expectation(pauli_w_2, psi)


def get_measurement_variance_simple(groupings, wfs, n_qubits, ev_dict=None):
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
        var_val = variance_of_group(group, wfs, n_qubits, ev_dict)
        var.append(var_val)

    sqrt_var = np.sqrt(np.abs(np.real_if_close(np.array(var))))
    variance = np.sum(sqrt_var)**2
    return var, variance, ev_dict



def variance_of_group(fragment: QubitOperator, wfs, n_qubits):
   return variance(gso(fragment, n_qubits), wfs)


def commutator_variance(H: FermionOperator, decomp, N, psi):
    """
    Computes the variance of the [H, G] - K.
    :param H: The Hamiltonian to compute the variance of.
    :param decomp: The decomposition of the Hamiltonian.
    :param G: The fermion operator to define the gradient
    :param K: The Killer operator applied to the whole [H, G]
    :param N: The number of sites
    :return: The variance metric
    """

    vars = np.zeros(len(decomp), dtype=np.complex128)
    for i, frag in enumerate(decomp):
        vars[i] = variance(gso(frag, N), psi)
    return np.sum((vars) ** (1 / 2)) ** 2

def get_pauli_word_coefficient(P: QubitOperator, ghosts=None):
    """Given a single pauli word P, extract its coefficient.
    """
    if ghosts is not None:
       if P in ghosts:
          coeffs = [0.0]
       else:
          coeffs = list(P.terms.values())
    else:
       coeffs = list(P.terms.values())
    return coeffs[0]

def get_pauli_word_tuple(P: QubitOperator):
    """Given a single pauli word P, extract the tuple representing the word.
    """
    words = list(P.terms.keys())
    if len(words) != 1:
        raise(ValueError("P given is not a single pauli word"))
    return words[0]


if __name__ == "__main__":

    pass
