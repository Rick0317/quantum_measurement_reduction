"""
Qubit Space BLISS Implementation with Dual Killer Operators

This module provides BLISS functionality in the qubit space using Pauli operators
with both O(N̂ - Ne) and (N̂ - Ne)O killer operators.
"""

import numpy as np
from scipy.optimize import minimize
from openfermion import QubitOperator, bravyi_kitaev, jordan_wigner
from openfermion import FermionOperator

from entities.paulis import PauliString, pauli_ops_to_qop
from utils.ferm_utils import ferm_to_qubit


def _copy_qubit_hamiltonian(H: QubitOperator):
    """Create a deep copy of a QubitOperator."""
    H_copy = QubitOperator()
    for term, coeff in H.terms.items():
        H_copy += QubitOperator(term, coeff)
    return H_copy


def fermion_to_qubit_bliss(H_fermion: FermionOperator, mapping='bravyi_kitaev'):
    """
    Convert fermionic Hamiltonian to qubit space for BLISS.
    
    Args:
        H_fermion: FermionOperator representing the Hamiltonian
        mapping: 'bravyi_kitaev' or 'jordan_wigner'
    
    Returns:
        QubitOperator in qubit space
    """
    if mapping == 'bravyi_kitaev':
        H_qubit = bravyi_kitaev(H_fermion)
    elif mapping == 'jordan_wigner':
        H_qubit = jordan_wigner(H_fermion)
    else:
        raise ValueError("Mapping must be 'bravyi_kitaev' or 'jordan_wigner'")
    
    # Remove constant term and compress
    H_qubit -= H_qubit.constant
    H_qubit.compress()
    
    return H_qubit


def construct_number_operator_qubit(N, Ne, mapping='bravyi_kitaev'):
    """
    Construct the number operator (N̂ - Ne) in qubit space.
    
    Args:
        N: Number of qubits/spin-orbitals
        Ne: Number of electrons
        mapping: Fermion-to-qubit mapping used
    
    Returns:
        QubitOperator representing (N̂ - Ne)
    """
    # Create fermionic number operator
    number_operator_fermion = FermionOperator()
    for mode in range(N):
        number_operator_fermion += FermionOperator(((mode, 1), (mode, 0)))
    
    # Convert to qubit space
    number_operator_qubit = fermion_to_qubit_bliss(number_operator_fermion, mapping)
    
    # Subtract Ne
    number_operator_qubit -= QubitOperator((), Ne)
    
    return number_operator_qubit


def construct_number_squared_operator_qubit(N, Ne, mapping='bravyi_kitaev'):
    """
    Construct the squared number operator (N̂² - Ne²) in qubit space.
    
    Args:
        N: Number of qubits/spin-orbitals
        Ne: Number of electrons
        mapping: Fermion-to-qubit mapping used
    
    Returns:
        QubitOperator representing (N̂² - Ne²)
    """
    # Create fermionic number operator
    number_operator_fermion = FermionOperator()
    for mode in range(N):
        number_operator_fermion += FermionOperator(((mode, 1), (mode, 0)))
    
    # Square it
    number_squared_fermion = number_operator_fermion * number_operator_fermion
    
    # Convert to qubit space
    number_squared_qubit = fermion_to_qubit_bliss(number_squared_fermion, mapping)
    
    # Subtract Ne²
    number_squared_qubit -= QubitOperator((), Ne**2)
    
    return number_squared_qubit


def construct_one_body_operator_qubit(params, N, mapping='bravyi_kitaev'):
    """
    Construct a one-body operator from parameters in qubit space.
    
    Args:
        params: Upper triangular matrix parameters (N*(N+1)/2 elements)
        N: Number of qubits/spin-orbitals
        mapping: Fermion-to-qubit mapping used
    
    Returns:
        QubitOperator representing the one-body operator
    """
    # Create symmetric matrix from upper triangular parameters
    sym_matrix = np.zeros((N, N))
    upper_tri_indices = np.triu_indices(N)
    sym_matrix[upper_tri_indices] = params
    sym_matrix = sym_matrix + sym_matrix.T - np.diag(np.diag(sym_matrix))
    
    # Create fermionic one-body operator
    one_body_fermion = FermionOperator()
    for i in range(N):
        for j in range(N):
            one_body_fermion += FermionOperator(f"{i}^ {j}", sym_matrix[i, j])
    
    # Convert to qubit space
    one_body_qubit = fermion_to_qubit_bliss(one_body_fermion, mapping)
    
    return one_body_qubit


def construct_H_bliss_qubit_dual_killer(H_qubit, params, N, Ne, mapping='bravyi_kitaev'):
    """
    Construct BLISS Hamiltonian H - K in qubit space with dual killer operators.
    
    Args:
        H_qubit: Original Hamiltonian in qubit space
        params: Optimization parameters [mu1, mu2, o1_params, o2_params...]
        N: Number of qubits/spin-orbitals
        Ne: Number of electrons
        mapping: Fermion-to-qubit mapping used
    
    Returns:
        QubitOperator representing H - K
    """
    result = _copy_qubit_hamiltonian(H_qubit)
    
    # Extract parameters
    mu1 = params[0]
    mu2 = params[1]
    o1_params = params[2:2 + int(N * (N + 1) // 2)]
    o2_params = params[2 + int(N * (N + 1) // 2):2 + 2 * int(N * (N + 1) // 2)]
    
    # Construct symmetry operators in qubit space
    number_op = construct_number_operator_qubit(N, Ne, mapping)
    number_squared_op = construct_number_squared_operator_qubit(N, Ne, mapping)
    one_body_op1 = construct_one_body_operator_qubit(o1_params, N, mapping)
    one_body_op2 = construct_one_body_operator_qubit(o2_params, N, mapping)
    
    # Apply BLISS transformation: H - K
    # K = μ₁(N̂ - Ne) + μ₂(N̂² - Ne²) + O₁(N̂ - Ne) + (N̂ - Ne)O₂
    result -= mu1 * number_op
    result -= mu2 * number_squared_op
    result -= one_body_op1 * number_op  # O₁(N̂ - Ne)
    result -= number_op * one_body_op2  # (N̂ - Ne)O₂
    
    return result


def compute_one_norm_qubit(H_qubit):
    """
    Compute the 1-norm of a QubitOperator.
    
    Args:
        H_qubit: QubitOperator
    
    Returns:
        float: 1-norm of the operator
    """
    return sum(abs(coeff) for coeff in H_qubit.terms.values())


def optimization_wrapper_dual_killer_qubit(H_qubit, N, Ne, mapping='bravyi_kitaev'):
    """
    Create optimization wrapper for dual killer qubit space BLISS.
    
    Args:
        H_qubit: Original Hamiltonian in qubit space
        N: Number of qubits/spin-orbitals
        Ne: Number of electrons
        mapping: Fermion-to-qubit mapping used
    
    Returns:
        tuple: (optimization_function, initial_guess)
    """
    def optimization_function(params):
        H_bliss = construct_H_bliss_qubit_dual_killer(H_qubit, params, N, Ne, mapping)
        return compute_one_norm_qubit(H_bliss)
    
    # Initial guess: [mu1, mu2, o1_params, o2_params...]
    mu1_init = 0.0
    mu2_init = 0.0
    o1_init = np.random.random(int(N * (N + 1) // 2)) * 0.01
    o2_init = np.random.random(int(N * (N + 1) // 2)) * 0.01
    
    initial_guess = np.concatenate([np.array([mu1_init, mu2_init]), o1_init, o2_init])
    
    return optimization_function, initial_guess


def bliss_qubit_dual_killer_two_body(H_fermion, N, Ne, mapping='bravyi_kitaev'):
    """
    Apply BLISS with dual killer operators to two-body Hamiltonian in qubit space.
    
    Args:
        H_fermion: FermionOperator representing the Hamiltonian
        N: Number of qubits/spin-orbitals
        Ne: Number of electrons
        mapping: Fermion-to-qubit mapping ('bravyi_kitaev' or 'jordan_wigner')
    
    Returns:
        QubitOperator: BLISS-optimized Hamiltonian in qubit space
    """
    # Convert to qubit space
    H_qubit = fermion_to_qubit_bliss(H_fermion, mapping)
    
    # Create optimization wrapper
    optimization_wrapper, initial_guess = optimization_wrapper_dual_killer_qubit(
        H_qubit, N, Ne, mapping
    )
    
    # Optimize
    res = minimize(
        optimization_wrapper,
        initial_guess,
        method="Powell",
        options={"disp": True, "maxiter": 100000},
    )
    
    # Construct final BLISS Hamiltonian
    H_bliss_qubit = construct_H_bliss_qubit_dual_killer(
        H_qubit, res.x, N, Ne, mapping
    )
    
    return H_bliss_qubit


def compare_dual_killer_bliss_results(H_fermion, N, Ne, mapping='bravyi_kitaev'):
    """
    Compare original and dual killer BLISS results.
    
    Args:
        H_fermion: FermionOperator representing the Hamiltonian
        N: Number of qubits/spin-orbitals
        Ne: Number of electrons
        mapping: Fermion-to-qubit mapping used
    
    Returns:
        dict: Comparison results
    """
    # Original qubit Hamiltonian
    H_qubit_orig = fermion_to_qubit_bliss(H_fermion, mapping)
    orig_norm = compute_one_norm_qubit(H_qubit_orig)
    
    # Dual killer BLISS-optimized qubit Hamiltonian
    H_bliss_qubit = bliss_qubit_dual_killer_two_body(H_fermion, N, Ne, mapping)
    bliss_norm = compute_one_norm_qubit(H_bliss_qubit)
    
    reduction = (orig_norm - bliss_norm) / orig_norm * 100
    
    return {
        'original_norm': orig_norm,
        'bliss_norm': bliss_norm,
        'reduction_percent': reduction,
        'original_terms': len(H_qubit_orig.terms),
        'bliss_terms': len(H_bliss_qubit.terms)
    }


def analyze_dual_killer_parameters(params, N):
    """
    Analyze the optimized parameters for dual killer operators.
    
    Args:
        params: Optimized parameters
        N: Number of qubits/spin-orbitals
    
    Returns:
        dict: Parameter analysis
    """
    mu1 = params[0]
    mu2 = params[1]
    o1_params = params[2:2 + int(N * (N + 1) // 2)]
    o2_params = params[2 + int(N * (N + 1) // 2):2 + 2 * int(N * (N + 1) // 2)]
    
    return {
        'mu1': mu1,
        'mu2': mu2,
        'o1_norm': np.linalg.norm(o1_params),
        'o2_norm': np.linalg.norm(o2_params),
        'o1_max': np.max(np.abs(o1_params)),
        'o2_max': np.max(np.abs(o2_params)),
        'o1_mean': np.mean(np.abs(o1_params)),
        'o2_mean': np.mean(np.abs(o2_params))
    }
