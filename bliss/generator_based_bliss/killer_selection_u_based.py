"""
In this file, we will select the killer based on the previously selected
parameterization unitary operators.
"""
from typing import List
from openfermion import FermionOperator


class Killer:
    def __init__(self, N, Ne):
        self.N = N
        self.Ne = Ne


class HFKiller(Killer):
    """
    I want to return

    """
    def __init__(self, N, Ne):
        super().__init__(N, Ne)

    def get_parameterized_killer(self, t_ferm_op_list):
        param_op = FermionOperator()
        for occupied in range(self.N - self.Ne):
            param_op += t_ferm_op_list[occupied] * (
                FermionOperator(((occupied, 1), (occupied, 0))))

        for empty in range(self.N - self.Ne, self.N):
            param_op += t_ferm_op_list[empty] * (
                        FermionOperator(((empty, 1), (empty, 0))) - 1)

        return param_op


def get_killers(N, Ne, u_list: List[tuple]):
    """
    Get the killer operators to use based on the previously selected unitary
    operators. We focus on the generators that have been selected.
    :param u_list: List of anti-Hermitian generator indices. The order must be
    index 0 => The first unitary chosen at step 1.
    :return:
    """
    if not u_list:
        killer = HFKiller(N, Ne)
        return killer.get_parameterized_killer()
    else:
        used_indices = set()
        for anti_herm in u_list:
            for value in anti_herm:
                used_indices.add(value)




