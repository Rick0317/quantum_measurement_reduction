"""
Shared Pauli module for quantum measurement reduction.

This module provides functionality for implementing the Shared Pauli technique,
which reduces measurement variance by sharing Pauli operators between
commuting sets in quantum algorithms.
"""

from .shared_paulis import (
    update_decomp_w_shared_paulis,
    select_sharable_paulis,
    select_combs,
    get_sharable_paulis,
    get_share_pauli_only_decomp,
    get_pw_grp_idxes_no_fix_len,
    get_overlapping_decomp,
    get_sharable_only_decomp,
    get_coefficient_orderings,
    get_all_pw_indices,
    get_pauli_coeff_map,
    qubit_op_to_list
)

from .coefficient_optimizer import (
    variance_metric,
    get_pauli_word_tuple,
    optimize_coeffs,
    process_fragment,
    optimize_coeffs_parallel,
    get_pauli_word_coefficient,
    get_split_measurement_variance_unconstrained,
    get_meas_alloc
)

from .shared_pauli_package import (
    apply_shared_pauli
)

from .utils_shared_pauli_sparse import (
    sparse_variance,
    sparse_expectation,
    select_paulis_sparse,
    select_combs_sparse,
    update_decomp_w_shared_paulis_sparse,
    commutator_variance_sparse,
    variance_metric_sparse
)

__all__ = [
    # Main algorithms
    'update_decomp_w_shared_paulis',
    'select_sharable_paulis',
    'select_combs',
    'apply_shared_pauli',
    
    # Utility functions
    'get_sharable_paulis',
    'get_share_pauli_only_decomp',
    'get_pw_grp_idxes_no_fix_len',
    'get_overlapping_decomp',
    'get_sharable_only_decomp',
    'get_coefficient_orderings',
    'get_all_pw_indices',
    'get_pauli_coeff_map',
    'qubit_op_to_list',
    
    # Coefficient optimization
    'variance_metric',
    'get_pauli_word_tuple',
    'optimize_coeffs',
    'process_fragment',
    'optimize_coeffs_parallel',
    'get_pauli_word_coefficient',
    'get_split_measurement_variance_unconstrained',
    'get_meas_alloc',
    
    # Sparse matrix versions
    'sparse_variance',
    'sparse_expectation',
    'select_paulis_sparse',
    'select_combs_sparse',
    'update_decomp_w_shared_paulis_sparse',
    'commutator_variance_sparse',
    'variance_metric_sparse'
]
