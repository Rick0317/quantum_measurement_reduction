from openfermion import FermionOperator as FO, get_sparse_operator as gso, bravyi_kitaev, normal_ordered
from scipy.linalg import expm
import random
import numpy as np


def one_body_unitary(i, a, N):
    """
    Return the one-body fermionic operator based unitary as a matrix
    :param i:
    :param a:
    :param N:
    :return:
    """
    generator = FO(f"{i}^ {a}") - FO(f"{a}^ {i}")
    qubit_generator = bravyi_kitaev(generator)
    t_ia = random.random()
    print(f"parameter t_ia is {t_ia}")
    generator_matrix = gso(qubit_generator, N).toarray()
    unitary_ia = expm(t_ia * generator_matrix)
    return unitary_ia
