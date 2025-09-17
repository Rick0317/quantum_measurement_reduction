"""
Ghost Pauli module for quantum measurement reduction.

This module provides functionality for implementing the Ghost Pauli technique,
which reduces measurement variance in quantum algorithms by introducing
additional Pauli operators into commuting sets.
"""

from .utils_ghost_pauli import (
    is_z_string,
    multiply_pauli_terms,
    find_z_string_combination,
    matrix_J,
    exclude_paulis,
    check_commutativity,
    get_variance_reduction,
    select_paulis,
    select_combs,
    update_decomp_w_ghost_paulis,
    commutator_variance,
    variance_metric,
    copy_hamiltonian
)

from .utils_ghost_pauli_sparse import (
    sparse_variance,
    sparse_expectation,
    select_paulis_sparse,
    select_combs_sparse,
    update_decomp_w_ghost_paulis_sparse,
    commutator_variance_sparse,
    variance_metric_sparse
)

__all__ = [
    # Core functions
    'is_z_string',
    'multiply_pauli_terms',
    'find_z_string_combination',
    'matrix_J',
    'exclude_paulis',
    'check_commutativity',
    'get_variance_reduction',
    
    # Main algorithms (dense)
    'select_paulis',
    'select_combs',
    'update_decomp_w_ghost_paulis',
    'commutator_variance',
    'variance_metric',
    'copy_hamiltonian',
    
    # Sparse matrix versions
    'sparse_variance',
    'sparse_expectation',
    'select_paulis_sparse',
    'select_combs_sparse',
    'update_decomp_w_ghost_paulis_sparse',
    'commutator_variance_sparse',
    'variance_metric_sparse'
]
