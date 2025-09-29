"""
Example usage of Qubit Space BLISS

This script demonstrates how to use the qubit space BLISS implementation
and compares it with the original fermionic approach.
"""

import numpy as np
from openfermion import FermionOperator

from bliss.qubit_bliss import bliss_qubit_two_body, compare_fermion_qubit_bliss
from bliss.bliss_main import bliss_two_body


def create_simple_hamiltonian(N=4):
    """
    Create a simple fermionic Hamiltonian for testing.
    
    Args:
        N: Number of spin-orbitals
    
    Returns:
        FermionOperator: Simple test Hamiltonian
    """
    H = FermionOperator()
    
    # Add some one-body terms
    for i in range(N):
        H += FermionOperator(f"{i}^ {i}", 0.5)
    
    # Add some two-body terms
    for i in range(N-1):
        for j in range(i+1, N):
            H += FermionOperator(f"{i}^ {j}^ {j} {i}", 0.1)
    
    return H


def demonstrate_qubit_bliss():
    """Demonstrate qubit space BLISS functionality."""
    print("=== Qubit Space BLISS Demonstration ===\n")
    
    # Parameters
    N = 4  # Number of spin-orbitals
    Ne = 2  # Number of electrons
    
    # Create test Hamiltonian
    H_fermion = create_simple_hamiltonian(N)
    print(f"Created test Hamiltonian with {len(H_fermion.terms)} terms")
    
    # Test both mappings
    for mapping in ['bravyi_kitaev', 'jordan_wigner']:
        print(f"\n--- Testing {mapping.upper()} mapping ---")
        
        try:
            # Apply qubit space BLISS
            H_bliss_qubit = bliss_qubit_two_body(H_fermion, N, Ne, mapping)
            
            # Compare results
            comparison = compare_fermion_qubit_bliss(H_fermion, N, Ne, mapping)
            
            print(f"Original 1-norm: {comparison['original_norm']:.6f}")
            print(f"BLISS 1-norm: {comparison['bliss_norm']:.6f}")
            print(f"Reduction: {comparison['reduction_percent']:.2f}%")
            print(f"Original terms: {comparison['original_terms']}")
            print(f"BLISS terms: {comparison['bliss_terms']}")
            
        except Exception as e:
            print(f"Error with {mapping}: {e}")
    
    # Compare with original fermionic BLISS
    print(f"\n--- Comparing with Original Fermionic BLISS ---")
    try:
        H_bliss_fermion = bliss_two_body(H_fermion, N, Ne)
        print(f"Fermionic BLISS completed with {len(H_bliss_fermion.terms)} terms")
        
        # Convert to qubit for comparison
        from bliss.qubit_bliss import fermion_to_qubit_bliss, compute_one_norm_qubit
        H_bliss_fermion_qubit = fermion_to_qubit_bliss(H_bliss_fermion, 'bravyi_kitaev')
        fermion_norm = compute_one_norm_qubit(H_bliss_fermion_qubit)
        print(f"Fermionic BLISS 1-norm (in qubit space): {fermion_norm:.6f}")
        
    except Exception as e:
        print(f"Error with fermionic BLISS: {e}")


def analyze_pauli_terms(H_qubit, max_terms=10):
    """
    Analyze the structure of Pauli terms in a QubitOperator.
    
    Args:
        H_qubit: QubitOperator to analyze
        max_terms: Maximum number of terms to display
    """
    print(f"\n--- Pauli Term Analysis ---")
    print(f"Total terms: {len(H_qubit.terms)}")
    
    # Sort terms by coefficient magnitude
    sorted_terms = sorted(H_qubit.terms.items(), 
                         key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\nTop {min(max_terms, len(sorted_terms))} terms by coefficient magnitude:")
    for i, (term, coeff) in enumerate(sorted_terms[:max_terms]):
        term_str = str(term) if term else "Identity"
        print(f"{i+1:2d}. {term_str:20s} : {coeff:12.6f}")


if __name__ == "__main__":
    demonstrate_qubit_bliss()
    
    # Additional analysis
    print("\n" + "="*50)
    print("Additional Analysis")
    print("="*50)
    
    # Create Hamiltonian and analyze
    H_fermion = create_simple_hamiltonian(4)
    from bliss.qubit_bliss import fermion_to_qubit_bliss
    H_qubit = fermion_to_qubit_bliss(H_fermion, 'bravyi_kitaev')
    
    analyze_pauli_terms(H_qubit)
    
    # Apply BLISS and analyze
    H_bliss = bliss_qubit_two_body(H_fermion, 4, 2, 'bravyi_kitaev')
    analyze_pauli_terms(H_bliss)
