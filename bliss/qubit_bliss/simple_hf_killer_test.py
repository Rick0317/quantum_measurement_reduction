#!/usr/bin/env python3
"""
Simple test for Hartree-Fock killer operator BLISS with a small system.
"""

import sys
import os
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openfermion import FermionOperator
from hartree_fock_killer_bliss import (
    construct_killer_operator_qubit,
    get_hartree_fock_state,
    evaluate_killer_operator_effectiveness,
    construct_parameterized_operator_qubit,
    construct_annihilation_operator_qubit
)


def create_simple_hamiltonian(N=4):
    """Create a simple fermionic Hamiltonian for testing."""
    H = FermionOperator()
    
    # Add some one-body terms
    for i in range(N):
        H += FermionOperator(f"{i}^ {i}", 0.5)
    
    # Add some two-body terms
    for i in range(N-1):
        for j in range(i+1, N):
            H += FermionOperator(f"{i}^ {j}^ {j} {i}", 0.1)
    
    return H


def test_killer_operator_components(N=4, Ne=2, mode=0, mapping='bravyi_kitaev'):
    """Test the components of the killer operator."""
    print(f"Testing killer operator components:")
    print(f"  N={N}, Ne={Ne}, mode={mode}, mapping={mapping}")
    
    # Create random parameters
    params = np.random.random(int(N * (N + 1) // 2)) * 0.01
    
    # Construct components
    O_qubit = construct_parameterized_operator_qubit(params, N, mapping)
    a_qubit = construct_annihilation_operator_qubit(mode, N, mapping)
    killer_qubit = construct_killer_operator_qubit(params, mode, N, mapping)
    
    print(f"  Parameterized operator O: {len(O_qubit.terms)} terms")
    print(f"  Annihilation operator a_{mode}: {len(a_qubit.terms)} terms")
    print(f"  Killer operator (O a_{mode})_q: {len(killer_qubit.terms)} terms")
    
    return O_qubit, a_qubit, killer_qubit


def test_hartree_fock_state(N=4, Ne=2, mapping='bravyi_kitaev'):
    """Test the Hartree-Fock state."""
    print(f"\nTesting Hartree-Fock state:")
    print(f"  N={N}, Ne={Ne}, mapping={mapping}")
    
    hf_state = get_hartree_fock_state(N, Ne, mapping)
    
    print(f"  State dimension: {len(hf_state)}")
    print(f"  Non-zero elements: {np.count_nonzero(hf_state)}")
    print(f"  Norm: {np.linalg.norm(hf_state):.6f}")
    
    # Find the non-zero element
    non_zero_indices = np.where(hf_state != 0)[0]
    if len(non_zero_indices) > 0:
        print(f"  Non-zero index: {non_zero_indices[0]}")
        print(f"  Value: {hf_state[non_zero_indices[0]]:.6f}")
    
    return hf_state


def test_killer_effectiveness_small(N=4, Ne=2, mode=0, mapping='bravyi_kitaev'):
    """Test killer operator effectiveness with a small system."""
    print(f"\nTesting killer operator effectiveness:")
    print(f"  N={N}, Ne={Ne}, mode={mode}, mapping={mapping}")
    
    # Get Hartree-Fock state
    hf_state = get_hartree_fock_state(N, Ne, mapping)
    
    # Test different parameter sets
    print(f"  Testing different parameter scales:")
    
    for scale in [0.01, 0.1, 1.0]:
        # Create random parameters
        params = np.random.random(int(N * (N + 1) // 2)) * scale
        
        # Construct killer operator
        killer_qubit = construct_killer_operator_qubit(params, mode, N, mapping)
        
        try:
            # Evaluate effectiveness
            effectiveness = evaluate_killer_operator_effectiveness(killer_qubit, hf_state)
            print(f"    Scale {scale}: effectiveness = {effectiveness:.6f}, "
                  f"terms = {len(killer_qubit.terms)}")
        except Exception as e:
            print(f"    Scale {scale}: Error - {e}")


def test_simple_bliss_optimization(N=4, Ne=2, mode=0, mapping='bravyi_kitaev'):
    """Test simple BLISS optimization with a small system."""
    print(f"\nTesting simple BLISS optimization:")
    print(f"  N={N}, Ne={Ne}, mode={mode}, mapping={mapping}")
    
    # Create simple Hamiltonian
    H_fermion = create_simple_hamiltonian(N)
    print(f"  Original Hamiltonian: {len(H_fermion.terms)} terms")
    
    # Convert to qubit space
    from hartree_fock_killer_bliss import fermion_to_qubit_bliss, compute_one_norm_qubit
    H_qubit = fermion_to_qubit_bliss(H_fermion, mapping)
    orig_norm = compute_one_norm_qubit(H_qubit)
    print(f"  Original 1-norm: {orig_norm:.6f}")
    
    # Test optimization with different initial guesses
    from scipy.optimize import minimize
    
    def optimization_function(params):
        killer_qubit = construct_killer_operator_qubit(params, mode, N, mapping)
        H_bliss = H_qubit - killer_qubit
        return compute_one_norm_qubit(H_bliss)
    
    # Initial guess
    initial_guess = np.random.random(int(N * (N + 1) // 2)) * 0.01
    
    try:
        res = minimize(
            optimization_function,
            initial_guess,
            method="Powell",
            options={"disp": False, "maxiter": 1000},
        )
        
        if res.success:
            final_norm = res.fun
            reduction = (orig_norm - final_norm) / orig_norm * 100
            print(f"  Optimization successful!")
            print(f"  Final 1-norm: {final_norm:.6f}")
            print(f"  Reduction: {reduction:.2f}%")
        else:
            print(f"  Optimization failed: {res.message}")
            
    except Exception as e:
        print(f"  Optimization error: {e}")


def main():
    print("=== Simple Hartree-Fock Killer Operator BLISS Test ===")
    print("Testing with small systems to avoid memory issues\n")
    
    # Test with small system
    N = 4  # 4 qubits
    Ne = 2  # 2 electrons
    mode = 0  # Test annihilation on mode 0
    
    # Test components
    print("1. Testing killer operator components")
    O_qubit, a_qubit, killer_qubit = test_killer_operator_components(N, Ne, mode)
    
    # Test Hartree-Fock state
    print("\n2. Testing Hartree-Fock state")
    hf_state = test_hartree_fock_state(N, Ne)
    
    # Test effectiveness
    print("\n3. Testing killer operator effectiveness")
    test_killer_effectiveness_small(N, Ne, mode)
    
    # Test simple optimization
    print("\n4. Testing simple BLISS optimization")
    test_simple_bliss_optimization(N, Ne, mode)
    
    print(f"\n✅ Simple Hartree-Fock killer operator BLISS test completed!")


if __name__ == "__main__":
    main()
