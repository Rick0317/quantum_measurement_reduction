from openfermion import QubitOperator
from typing import List


def does_term_frag_commute(term, frag: QubitOperator):
    return (term * frag - frag * term)


def get_pauli_words_lst(decomposition: List[QubitOperator]):
    pauli_words_lst = []
    for fragment in decomposition:
        for _, pauli in enumerate(fragment.terms):
            pauli_words_lst.append(pauli)


def get_pauli_words_decomp(decomposition: List[QubitOperator]):
    pauli_words_decomp = []
    for fragment in decomposition:
        fragment_list = []
        for _, pauli in enumerate(fragment.terms):
            fragment_list.append(QubitOperator(term=pauli))

        pauli_words_decomp.append(fragment_list)
    return pauli_words_decomp
