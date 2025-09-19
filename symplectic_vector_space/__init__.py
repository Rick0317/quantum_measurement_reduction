"""
Symplectic Vector Space module for quantum measurement reduction.

This module provides functionality for working with symplectic vector spaces
over binary fields, which are used in the mathematical framework of
quantum measurement reduction techniques.
"""

from .space_F_definition import (
    SpaceFVector,
    _pauli_2_vector,
    binary_2_pauli,
    pauli_2_binary,
    vector_2_pauli,
)

__all__ = [
    "pauli_2_binary",
    "binary_2_pauli",
    "_pauli_2_vector",
    "vector_2_pauli",
    "SpaceFVector",
]
