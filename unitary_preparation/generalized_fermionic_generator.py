import random

import numpy as np
from openfermion import FermionOperator as FO
from openfermion import (
    bravyi_kitaev,
)
from openfermion import get_sparse_operator as gso
from openfermion import (
    normal_ordered,
)
from scipy.linalg import expm


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
