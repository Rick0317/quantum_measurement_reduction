from openfermion import FermionOperator, normal_ordered, bravyi_kitaev
import numpy as np


def chemist_ordered(fermion_operator):
    r"""Puts a two-body fermion operator in chemist ordering.

    The normal ordering convention for chemists is different.
    Rather than ordering the two-body term as physicists do, as
    $a^\dagger a^\dagger a a$
    the chemist ordering of the two-body term is
    $a^\dagger a a^\dagger a$

    Args:
        fermion_operator (FermionOperator): a fermion operator guaranteed to
            have number conserving one- and two-body fermion terms only.
    Returns:
        chemist_ordered_operator (FermionOperator): the input operator
            ordered in the chemistry convention.
    Raises:
        OperatorSpecificationError: Operator is not two-body number conserving.
    """

    # Normal order and begin looping.
    normal_ordered_input = normal_ordered(fermion_operator)
    chemist_ordered_operator = FermionOperator()
    for term, coefficient in normal_ordered_input.terms.items():
        if len(term) == 2 or not len(term):
            chemist_ordered_operator += FermionOperator(term, coefficient)
        elif len(term) == 4:
            # Possibly add new one-body term.
            if term[1][0] == term[2][0]:
                new_one_body_term = (term[0], term[3])
                chemist_ordered_operator += FermionOperator(new_one_body_term, coefficient)
            # Reorder two-body term.
            new_two_body_term = (term[0], term[2], term[1], term[3])
            chemist_ordered_operator += FermionOperator(new_two_body_term, -coefficient)
        elif len(term) == 6:
            if term[2][0] == term[3][0]:
                new_two_body_term1 = (term[0], term[1], term[4], term[5])
                chemist_ordered_operator += chemist_ordered(FermionOperator(new_two_body_term1, coefficient))
            if term[1][0] == term[3][0]:
                new_two_body_term2 = (term[0], term[2], term[4], term[5])
                chemist_ordered_operator += chemist_ordered(FermionOperator(new_two_body_term2, -coefficient))
            if term[2][0] == term[4][0]:
                new_two_body_term3 = (term[0], term[3], term[1], term[5])
                chemist_ordered_operator += FermionOperator(new_two_body_term3, coefficient)
            new_three_body_term = (term[0], term[3], term[1], term[4], term[2], term[5])
            chemist_ordered_operator += FermionOperator(new_three_body_term, -coefficient)

        elif len(term) == 8:
            if term[3][0] == term[4][0]:
                new_three_body_term1 = (term[0], term[1], term[2], term[5], term[6], term[7])
                chemist_ordered_operator += chemist_ordered(FermionOperator(new_three_body_term1, coefficient))
            if term[2][0] == term[4][0]:
                new_three_body_term2 = (term[0], term[1], term[3], term[5], term[6], term[7])
                chemist_ordered_operator += chemist_ordered(FermionOperator(new_three_body_term2, -coefficient))
            if term[1][0] == term[4][0]:
                new_three_body_term3 = (term[0], term[2], term[3], term[5], term[6], term[7])
                chemist_ordered_operator += chemist_ordered(FermionOperator(new_three_body_term3, coefficient))

            prefix_term = (term[0], term[4])
            tale_op = FermionOperator((term[1], term[2], term[3], term[5], term[6], term[7]), -coefficient)
            chemist_ordered_operator += FermionOperator(prefix_term, 1) * chemist_ordered(tale_op)

    return chemist_ordered_operator


def ferm_op_to_tensors(fermion_operator: FermionOperator, N):
    h1e = np.zeros((N, N))
    g2e = np.zeros((N, N, N, N))
    t3e = np.zeros((N, N, N, N, N, N))
    q4e = np.zeros((N, N, N, N, N, N, N, N))
    for term, coefficient in fermion_operator.terms.items():
        if len(term) == 2:
            h1e[term[0][0], term[1][0]] = coefficient
        elif len(term) == 4:
            g2e[term[0][0], term[1][0], term[2][0], term[3][0]] = coefficient
        elif len(term) == 6:
            t3e[term[0][0], term[1][0], term[2][0], term[3][0], term[4][0],
            term[5][0]] = coefficient
        elif len(term) == 8:
            q4e[term[0][0], term[1][0], term[2][0], term[3][0], term[4][0],
            term[5][0], term[6][0], term[7][0]] = coefficient

    return h1e, g2e, t3e, q4e


def abs_of_dict_value(x):
    return np.abs(x[1])

def ferm_to_qubit(H: FermionOperator):
    """
    Converts a FermionOperator to a qubit with Bravyi Kitaev
    :param H:
    :return:
    """
    Hqub = bravyi_kitaev(H)
    Hqub -= Hqub.constant
    Hqub.compress()
    Hqub.terms = dict(
    sorted(Hqub.terms.items(), key=abs_of_dict_value, reverse=True))
    return Hqub
