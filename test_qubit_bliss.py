#!/usr/bin/env python3
"""
Simple test script for qubit space BLISS functionality.
Run this from the project root directory.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openfermion import FermionOperator
from bliss.qubit_bliss import bliss_qubit_two_body, compare_fermion_qubit_bliss

def main():
    print("=== Qubit Space BLISS Test ===\n")
    
    # Create a simple test Hamiltonian
    H = FermionOperator("0^ 0", 0.5)  # One-body term
    H += FermionOperator("0^ 1^ 1 0", 0.2)  # Two-body term
    H += FermionOperator("1^ 1", 0.3)  # Another one-body term
    
    print(f"Original Hamiltonian has {len(H.terms)} terms")
    
    # Test parameters
    N = 4  # Number of spin-orbitals
    Ne = 2  # Number of electrons
    
    # Apply qubit space BLISS
    print("\nApplying qubit space BLISS...")
    H_bliss = bliss_qubit_two_body(H, N, Ne, mapping='bravyi_kitaev')
    
    # Compare results
    comparison = compare_fermion_qubit_bliss(H, N, Ne, mapping='bravyi_kitaev')
    
    print(f"\nResults:")
    print(f"  Original 1-norm: {comparison['original_norm']:.6f}")
    print(f"  BLISS 1-norm: {comparison['bliss_norm']:.6f}")
    print(f"  Reduction: {comparison['reduction_percent']:.2f}%")
    print(f"  Original terms: {comparison['original_terms']}")
    print(f"  BLISS terms: {comparison['bliss_terms']}")
    
    print("\n✅ Qubit space BLISS test completed successfully!")

if __name__ == "__main__":
    main()
