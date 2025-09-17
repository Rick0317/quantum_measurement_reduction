from scipy import linprog


def custom_linear_programming(params):
    """
    Custom Linear Programming implementation for BLISS when the params changes
    for different Hamiltonians
    :param params:
    :return:
    """
    coeffs = ...
    a_ub = ...
    b_ub = ...

    result = linprog()
