"""
This file checks the commutation relation between a Killer that annihilates
the reference state (HF) and the product of unitary operators that were chosen
prior to the current step.
"""
from openfermion import FermionOperator as FO, get_sparse_operator as gso, bravyi_kitaev, normal_ordered
from scipy.linalg import expm
import random
import numpy as np
from state_perparation.parametrized_state import get_single_unitary_parametered_state


def one_body_unitary(i, a, N):
    generator = FO(f"{i}^ {a}") - FO(f"{a}^ {i}")
    qubit_generator = generator
    t_ia = random.random()
    generator_matrix = gso(qubit_generator, N).toarray()
    print(generator_matrix.shape)
    unitary_ia = expm(t_ia * generator_matrix)
    return unitary_ia


def is_unitary(U, tol=1e-10):
    """Check if a matrix U is unitary: U†U = I and UU† = I"""
    U_dag = np.conjugate(U.T)
    identity = np.eye(U.shape[0])

    # Check if U†U ≈ I and UU† ≈ I
    return np.allclose(U_dag @ U, identity, atol=tol) and np.allclose(U @ U_dag, identity, atol=tol)


if __name__ == "__main__":
    N = 4
    Ne = 2
    tol = 1e-10
    unitary_ia = one_body_unitary(2, 0, N)
    i = 2
    a = 0
    generator = FO(f"{i}^ {a}") - FO(f"{a}^ {i}")
    generator_matrix = gso(generator, N).toarray()
    G3 = generator * generator * generator

    j = 3
    b = 1
    parametrized_state = get_single_unitary_parametered_state(i, a, N, Ne)
    killer = FO(f"{j}^ {b}")
    generator_matrix = gso(bravyi_kitaev(killer), N).toarray()
    print(generator_matrix @ parametrized_state)

