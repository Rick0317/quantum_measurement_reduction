"""
Killer Operator BLISS Implementation

This module implements BLISS using killer operators of the form (O a)_q where:
- O is a parameterized operator
- a is an annihilation operator in fermionic space
- The subscript q indicates conversion to qubit space
"""

import numpy as np
from scipy.optimize import minimize
from openfermion import QubitOperator, FermionOperator, bravyi_kitaev, jordan_wigner
from openfermion import get_fermion_operator

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


def construct_parameterized_operator_qubit(params, N, mapping='bravyi_kitaev'):
    """
    Construct a parameterized operator O in qubit space.
    
    Args:
        params: Parameters for the operator (N*(N+1)/2 elements for symmetric matrix)
        N: Number of qubits/spin-orbitals
        mapping: Fermion-to-qubit mapping used
    
    Returns:
        QubitOperator representing the parameterized operator O
    """
    # Create symmetric matrix from upper triangular parameters
    sym_matrix = np.zeros((N, N))
    upper_tri_indices = np.triu_indices(N)
    sym_matrix[upper_tri_indices] = params
    sym_matrix = sym_matrix + sym_matrix.T - np.diag(np.diag(sym_matrix))
    
    # Create fermionic parameterized operator
    param_fermion = FermionOperator()
    for i in range(N):
        for j in range(N):
            param_fermion += FermionOperator(f"{i}^ {j}", sym_matrix[i, j])
    
    # Convert to qubit space
    param_qubit = fermion_to_qubit_bliss(param_fermion, mapping)
    
    return param_qubit


def construct_annihilation_operator_qubit(mode, N, mapping='bravyi_kitaev'):
    """
    Construct an annihilation operator a_i in qubit space.
    
    Args:
        mode: The mode index to annihilate
        N: Number of qubits/spin-orbitals
        mapping: Fermion-to-qubit mapping used
    
    Returns:
        QubitOperator representing the annihilation operator a_i
    """
    # Create fermionic annihilation operator
    annihilate_fermion = FermionOperator(f"{mode}")
    
    # Convert to qubit space
    annihilate_qubit = fermion_to_qubit_bliss(annihilate_fermion, mapping)
    
    return annihilate_qubit


def construct_killer_operator_qubit(params, mode, N, mapping='bravyi_kitaev'):
    """
    Construct the killer operator (O a_i)_q in qubit space.
    
    Args:
        params: Parameters for the operator O
        mode: The mode index for the annihilation operator
        N: Number of qubits/spin-orbitals
        mapping: Fermion-to-qubit mapping used
    
    Returns:
        QubitOperator representing (O a_i)_q
    """
    # Construct parameterized operator O
    O_qubit = construct_parameterized_operator_qubit(params, N, mapping)
    
    # Construct annihilation operator a_i
    a_qubit = construct_annihilation_operator_qubit(mode, N, mapping)
    
    # Multiply to get (O a_i)_q
    killer_qubit = O_qubit * a_qubit
    
    return killer_qubit


def construct_H_bliss_killer_qubit(H_qubit, params, mode, N, Ne, mapping='bravyi_kitaev'):
    """
    Construct BLISS Hamiltonian H - K using killer operator (O a_i)_q.
    
    Args:
        H_qubit: Original Hamiltonian in qubit space
        params: Optimization parameters for the killer operator
        mode: The mode index for the annihilation operator
        N: Number of qubits/spin-orbitals
        Ne: Number of electrons
        mapping: Fermion-to-qubit mapping used
    
    Returns:
        QubitOperator representing H - K
    """
    result = _copy_qubit_hamiltonian(H_qubit)
    
    # Construct killer operator (O a_i)_q
    killer_qubit = construct_killer_operator_qubit(params, mode, N, mapping)
    
    # Apply BLISS transformation: H - K
    result -= killer_qubit
    
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


def optimization_wrapper_killer_qubit(H_qubit, mode, N, Ne, mapping='bravyi_kitaev'):
    """
    Create optimization wrapper for killer operator BLISS.
    
    Args:
        H_qubit: Original Hamiltonian in qubit space
        mode: The mode index for the annihilation operator
        N: Number of qubits/spin-orbitals
        Ne: Number of electrons
        mapping: Fermion-to-qubit mapping used
    
    Returns:
        tuple: (optimization_function, initial_guess)
    """
    def optimization_function(params):
        H_bliss = construct_H_bliss_killer_qubit(H_qubit, params, mode, N, Ne, mapping)
        return compute_one_norm_qubit(H_bliss)
    
    # Initial guess: parameters for the parameterized operator O
    # Use N*(N+1)/2 parameters for symmetric matrix
    initial_guess = np.random.random(int(N * (N + 1) // 2)) * 0.01
    
    return optimization_function, initial_guess


def bliss_killer_operator_qubit(H_fermion, mode, N, Ne, mapping='bravyi_kitaev'):
    """
    Apply BLISS using killer operator (O a_i)_q to a fermionic Hamiltonian.
    
    Args:
        H_fermion: FermionOperator representing the Hamiltonian
        mode: The mode index for the annihilation operator
        N: Number of qubits/spin-orbitals
        Ne: Number of electrons
        mapping: Fermion-to-qubit mapping ('bravyi_kitaev' or 'jordan_wigner')
    
    Returns:
        QubitOperator: BLISS-optimized Hamiltonian in qubit space
    """
    # Convert to qubit space
    H_qubit = fermion_to_qubit_bliss(H_fermion, mapping)
    
    # Create optimization wrapper
    optimization_wrapper, initial_guess = optimization_wrapper_killer_qubit(
        H_qubit, mode, N, Ne, mapping
    )
    
    # Optimize
    res = minimize(
        optimization_wrapper,
        initial_guess,
        method="Powell",
        options={"disp": True, "maxiter": 100000},
    )
    
    # Construct final BLISS Hamiltonian
    H_bliss_qubit = construct_H_bliss_killer_qubit(
        H_qubit, res.x, mode, N, Ne, mapping
    )
    
    return H_bliss_qubit


def compare_killer_bliss_results(H_fermion, mode, N, Ne, mapping='bravyi_kitaev'):
    """
    Compare original and killer operator BLISS results.
    
    Args:
        H_fermion: FermionOperator representing the Hamiltonian
        mode: The mode index for the annihilation operator
        N: Number of qubits/spin-orbitals
        Ne: Number of electrons
        mapping: Fermion-to-qubit mapping used
    
    Returns:
        dict: Comparison results
    """
    # Original qubit Hamiltonian
    H_qubit_orig = fermion_to_qubit_bliss(H_fermion, mapping)
    orig_norm = compute_one_norm_qubit(H_qubit_orig)
    
    # Killer operator BLISS-optimized qubit Hamiltonian
    H_bliss_qubit = bliss_killer_operator_qubit(H_fermion, mode, N, Ne, mapping)
    bliss_norm = compute_one_norm_qubit(H_bliss_qubit)
    
    reduction = (orig_norm - bliss_norm) / orig_norm * 100
    
    return {
        'original_norm': orig_norm,
        'bliss_norm': bliss_norm,
        'reduction_percent': reduction,
        'original_terms': len(H_qubit_orig.terms),
        'bliss_terms': len(H_bliss_qubit.terms),
        'mode': mode
    }


def multi_mode_killer_bliss(H_fermion, modes, N, Ne, mapping='bravyi_kitaev'):
    """
    Apply BLISS using multiple killer operators (O a_i)_q for different modes.
    
    Args:
        H_fermion: FermionOperator representing the Hamiltonian
        modes: List of mode indices for the annihilation operators
        N: Number of qubits/spin-orbitals
        Ne: Number of electrons
        mapping: Fermion-to-qubit mapping used
    
    Returns:
        dict: Results for each mode
    """
    results = {}
    
    for mode in modes:
        print(f"\n--- Testing killer operator (O a_{mode})_q ---")
        try:
            result = compare_killer_bliss_results(H_fermion, mode, N, Ne, mapping)
            results[mode] = result
            print(f"Mode {mode}: {result['reduction_percent']:.2f}% reduction")
        except Exception as e:
            print(f"Error with mode {mode}: {e}")
            results[mode] = None
    
    return results


def analyze_killer_operator_structure(H_qubit, params, mode, N, mapping='bravyi_kitaev'):
    """
    Analyze the structure of the killer operator (O a_i)_q.
    
    Args:
        H_qubit: Original Hamiltonian in qubit space
        params: Parameters for the killer operator
        mode: The mode index for the annihilation operator
        N: Number of qubits/spin-orbitals
        mapping: Fermion-to-qubit mapping used
    
    Returns:
        dict: Analysis results
    """
    # Construct components
    O_qubit = construct_parameterized_operator_qubit(params, N, mapping)
    a_qubit = construct_annihilation_operator_qubit(mode, N, mapping)
    killer_qubit = construct_killer_operator_qubit(params, mode, N, mapping)
    
    return {
        'O_terms': len(O_qubit.terms),
        'a_terms': len(a_qubit.terms),
        'killer_terms': len(killer_qubit.terms),
        'O_norm': compute_one_norm_qubit(O_qubit),
        'a_norm': compute_one_norm_qubit(a_qubit),
        'killer_norm': compute_one_norm_qubit(killer_qubit)
    }
