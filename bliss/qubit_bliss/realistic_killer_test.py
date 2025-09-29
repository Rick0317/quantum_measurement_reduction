#!/usr/bin/env python3
"""
Realistic test for killer operator BLISS with proper virtual orbital targeting.
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
    construct_annihilation_operator_qubit,
    fermion_to_qubit_bliss,
    compute_one_norm_qubit
)
from scipy.optimize import minimize


def create_realistic_hamiltonian(N=4):
    """Create a more realistic fermionic Hamiltonian."""
    H = FermionOperator()
    
    # One-body terms (kinetic + potential)
    for i in range(N):
        H += FermionOperator(f"{i}^ {i}", -0.5 + 0.1 * i)  # Different on-site energies
    
    # Two-body terms (Coulomb repulsion)
    for i in range(N):
        for j in range(i+1, N):
            H += FermionOperator(f"{i}^ {j}^ {j} {i}", 0.2)
    
    # Hopping terms
    for i in range(N-1):
        H += FermionOperator(f"{i}^ {i+1}", 0.3)
        H += FermionOperator(f"{i+1}^ {i}", 0.3)
    
    return H


def test_virtual_orbital_killer(N=4, Ne=2, mapping='bravyi_kitaev'):
    """Test killer operators on virtual (unoccupied) orbitals."""
    print(f"Testing virtual orbital killer operators:")
    print(f"  N={N}, Ne={Ne}, mapping={mapping}")
    
    # Get Hartree-Fock state
    hf_state = get_hartree_fock_state(N, Ne, mapping)
    
    # In HF, first (N-Ne) orbitals are virtual, last Ne are occupied
    virtual_modes = list(range(N - Ne))  # Virtual orbitals
    occupied_modes = list(range(N - Ne, N))  # Occupied orbitals
    
    print(f"  Virtual orbitals: {virtual_modes}")
    print(f"  Occupied orbitals: {occupied_modes}")
    
    # Test killer operators on virtual orbitals
    print(f"\n  Testing killer operators on virtual orbitals:")
    for mode in virtual_modes:
        # Create random parameters
        params = np.random.random(int(N * (N + 1) // 2)) * 0.1
        
        # Construct killer operator
        killer_qubit = construct_killer_operator_qubit(params, mode, N, mapping)
        
        # Evaluate effectiveness
        effectiveness = evaluate_killer_operator_effectiveness(killer_qubit, hf_state)
        
        print(f"    Mode {mode} (virtual): effectiveness = {effectiveness:.6f}, "
              f"terms = {len(killer_qubit.terms)}")
    
    # Test killer operators on occupied orbitals (should be zero)
    print(f"\n  Testing killer operators on occupied orbitals:")
    for mode in occupied_modes:
        # Create random parameters
        params = np.random.random(int(N * (N + 1) // 2)) * 0.1
        
        # Construct killer operator
        killer_qubit = construct_killer_operator_qubit(params, mode, N, mapping)
        
        # Evaluate effectiveness
        effectiveness = evaluate_killer_operator_effectiveness(killer_qubit, hf_state)
        
        print(f"    Mode {mode} (occupied): effectiveness = {effectiveness:.6f}, "
              f"terms = {len(killer_qubit.terms)}")


def test_optimization_with_virtual_killer(N=4, Ne=2, mode=0, mapping='bravyi_kitaev'):
    """Test BLISS optimization using virtual orbital killer operators."""
    print(f"\nTesting BLISS optimization with virtual orbital killer:")
    print(f"  N={N}, Ne={Ne}, mode={mode} (virtual), mapping={mapping}")
    
    # Create realistic Hamiltonian
    H_fermion = create_realistic_hamiltonian(N)
    print(f"  Original Hamiltonian: {len(H_fermion.terms)} terms")
    
    # Convert to qubit space
    H_qubit = fermion_to_qubit_bliss(H_fermion, mapping)
    orig_norm = compute_one_norm_qubit(H_qubit)
    print(f"  Original 1-norm: {orig_norm:.6f}")
    
    # Get Hartree-Fock state
    hf_state = get_hartree_fock_state(N, Ne, mapping)
    
    def optimization_function(params):
        # Construct killer operator
        killer_qubit = construct_killer_operator_qubit(params, mode, N, mapping)
        
        # Evaluate effectiveness with Hartree-Fock state
        effectiveness = evaluate_killer_operator_effectiveness(killer_qubit, hf_state)
        
        # Construct BLISS Hamiltonian
        H_bliss = H_qubit - killer_qubit
        
        # Compute 1-norm
        one_norm = compute_one_norm_qubit(H_bliss)
        
        # Objective: minimize 1-norm, but also consider effectiveness
        # We want the killer to be effective (large |effectiveness|) and reduce 1-norm
        objective = one_norm - 0.1 * abs(effectiveness)
        
        return objective
    
    # Initial guess
    initial_guess = np.random.random(int(N * (N + 1) // 2)) * 0.1
    
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
            
            # Evaluate final killer operator
            final_killer = construct_killer_operator_qubit(res.x, mode, N, mapping)
            final_effectiveness = evaluate_killer_operator_effectiveness(final_killer, hf_state)
            
            print(f"  Optimization successful!")
            print(f"  Final 1-norm: {final_norm:.6f}")
            print(f"  Reduction: {reduction:.2f}%")
            print(f"  Final effectiveness: {final_effectiveness:.6f}")
            print(f"  Final killer terms: {len(final_killer.terms)}")
        else:
            print(f"  Optimization failed: {res.message}")
            
    except Exception as e:
        print(f"  Optimization error: {e}")


def test_different_mappings(N=4, Ne=2, mode=0):
    """Test killer operators with different fermion-to-qubit mappings."""
    print(f"\nTesting different mappings:")
    print(f"  N={N}, Ne={Ne}, mode={mode}")
    
    mappings = ['bravyi_kitaev', 'jordan_wigner']
    
    for mapping in mappings:
        print(f"\n  Mapping: {mapping.upper()}")
        
        # Get Hartree-Fock state
        hf_state = get_hartree_fock_state(N, Ne, mapping)
        
        # Create random parameters
        params = np.random.random(int(N * (N + 1) // 2)) * 0.1
        
        # Construct killer operator
        killer_qubit = construct_killer_operator_qubit(params, mode, N, mapping)
        
        # Evaluate effectiveness
        effectiveness = evaluate_killer_operator_effectiveness(killer_qubit, hf_state)
        
        print(f"    Effectiveness: {effectiveness:.6f}")
        print(f"    Killer terms: {len(killer_qubit.terms)}")


def main():
    print("=== Realistic Killer Operator BLISS Test ===")
    print("Testing with virtual orbital targeting and realistic Hamiltonians\n")
    
    # Test with small system
    N = 4  # 4 qubits
    Ne = 2  # 2 electrons
    mode = 0  # Test annihilation on mode 0 (virtual)
    
    # Test virtual orbital killers
    print("1. Testing virtual orbital killer operators")
    test_virtual_orbital_killer(N, Ne)
    
    # Test optimization with virtual killer
    print("\n2. Testing BLISS optimization with virtual orbital killer")
    test_optimization_with_virtual_killer(N, Ne, mode)
    
    # Test different mappings
    print("\n3. Testing different mappings")
    test_different_mappings(N, Ne, mode)
    
    print(f"\n✅ Realistic killer operator BLISS test completed!")


if __name__ == "__main__":
    main()
