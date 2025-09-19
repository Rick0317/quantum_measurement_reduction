"""
Quantum Measurement Reduction Package

A comprehensive Python package for implementing various quantum measurement reduction
techniques including Ghost Pauli, Shared Pauli, Virial relations, and other methods
to reduce measurement variance in quantum algorithms.

This package provides:
- Ghost Pauli techniques for variance reduction
- Shared Pauli methods for measurement optimization
- Virial relation implementations
- Symplectic vector space utilities
- Core quantum data structures and utilities

Example usage:
    from quantum_measurement_reduction import ghost_pauli
    from quantum_measurement_reduction.ghost_pauli import update_decomp_w_ghost_paulis_sparse

    # Use the sparse ghost pauli function
    updated_decomp = update_decomp_w_ghost_paulis_sparse(psi_sparse, N, original_decomp)
"""

# Import main modules
from . import ghost_pauli
from . import shared_pauli
from . import entities
from . import symplectic_vector_space
from . import utils
from . import tapering
from . import virial
from . import bliss

# Import key functions for easy access
from .ghost_pauli import (
    update_decomp_w_ghost_paulis_sparse,
    sparse_variance,
    sparse_expectation
)

from .shared_pauli import (
    apply_shared_pauli,
    optimize_coeffs
)

from .entities import (
    PauliString,
    PauliOp
)

from .symplectic_vector_space import (
    SpaceFVector,
    vector_2_pauli
)

from .bliss import (
    bliss_two_body,
    bliss_three_body_indices_filtered
)

# Package metadata
__version__ = "0.1.0"
__author__ = "Quantum Measurement Reduction Team"
__email__ = "quantum@example.com"
__description__ = "A package for quantum measurement reduction techniques"

__all__ = [
    # Main modules
    'ghost_pauli',
    'shared_pauli',
    'entities',
    'SymplecticVectorSpace',
    'utils',
    'tapering',
    'virial',
    'bliss',

    # Key functions
    'update_decomp_w_ghost_paulis_sparse',
    'sparse_variance',
    'sparse_expectation',
    'apply_shared_pauli',
    'optimize_coeffs',
    'PauliString',
    'PauliOp',
    'SpaceFVector',
    'vector_2_pauli',
    'bliss_two_body',
    'bliss_three_body_indices_filtered',

    # Metadata
    '__version__',
    '__author__',
    '__email__',
    '__description__'
]

