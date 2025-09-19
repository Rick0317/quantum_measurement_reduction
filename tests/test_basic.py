"""
Basic tests for the quantum_measurement_reduction package.
"""

import pytest
from quantum_measurement_reduction import bliss_two_body, bliss_three_body_indices_filtered
from quantum_measurement_reduction.entities import PauliString, PauliOp


def test_package_imports():
    """Test that main package imports work correctly."""
    # Test bliss functions can be imported
    assert bliss_two_body is not None
    assert bliss_three_body_indices_filtered is not None
    
    # Test entities can be imported
    assert PauliString is not None
    assert PauliOp is not None


def test_pauli_string_creation():
    """Test basic PauliString creation."""
    pauli = PauliString("X", 0)
    assert pauli.operator == "X"
    assert pauli.qubit == 0


def test_pauli_op_creation():
    """Test basic PauliOp creation."""
    pauli_op = PauliOp([PauliString("X", 0), PauliString("Y", 1)])
    assert len(pauli_op.paulis) == 2
    assert pauli_op.paulis[0].operator == "X"
    assert pauli_op.paulis[1].operator == "Y"


if __name__ == "__main__":
    pytest.main([__file__])
