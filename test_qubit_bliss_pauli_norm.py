#!/usr/bin/env python3
"""
Test qubit_bliss_main with Pauli 1-norm calculations before and after BLISS.
"""

import sys
import os
import pickle
import time

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openfermion import FermionOperator, InteractionOperator, get_fermion_operator
from bliss.qubit_bliss import (
    bliss_qubit_two_body,
    compare_fermion_qubit_bliss,
    compute_one_norm_qubit,
    fermion_to_qubit_bliss
)


def load_molecular_hamiltonian(filename):
    """Load a molecular Hamiltonian from a .bin file and convert to FermionOperator."""
    filepath = os.path.join("ham_lib", filename)
    with open(filepath, "rb") as f:
        interaction_op = pickle.load(f)
    
    # Convert InteractionOperator to FermionOperator
    if isinstance(interaction_op, InteractionOperator):
        return get_fermion_operator(interaction_op)
    elif isinstance(interaction_op, FermionOperator):
        return interaction_op
    else:
        raise ValueError(f"Unexpected type: {type(interaction_op)}")


def get_molecule_info(filename):
    """Get molecule information based on filename."""
    molecule_info = {
        "beh2_fer.bin": {"name": "BeH2", "N": 14, "Ne": 6},
        "h4_fer.bin": {"name": "H4", "N": 8, "Ne": 4},
        "lih_fer.bin": {"name": "LiH", "N": 12, "Ne": 4},
        "n2_fer.bin": {"name": "N2", "N": 20, "Ne": 10},
    }
    return molecule_info.get(filename, {"name": "Unknown", "N": 4, "Ne": 2})


def test_qubit_bliss_pauli_norm(filename, mapping='bravyi_kitaev'):
    """Test qubit BLISS with Pauli 1-norm calculations."""
    print(f"\n{'='*60}")
    print(f"Testing {filename} with qubit BLISS - Pauli 1-norm analysis")
    print(f"Mapping: {mapping.upper()}")
    print(f"{'='*60}")
    
    # Load molecular data
    try:
        H_fermion = load_molecular_hamiltonian(filename)
        molecule = get_molecule_info(filename)
        print(f"Molecule: {molecule['name']}")
        print(f"Spin-orbitals: {molecule['N']}, Electrons: {molecule['Ne']}")
        print(f"Original Hamiltonian terms: {len(H_fermion.terms)}")
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None
    
    # Convert to qubit space
    print(f"\n--- Converting to Qubit Space ---")
    H_qubit_orig = fermion_to_qubit_bliss(H_fermion, mapping)
    orig_pauli_norm = compute_one_norm_qubit(H_qubit_orig)
    print(f"Original Pauli 1-norm: {orig_pauli_norm:.6f}")
    print(f"Original Pauli terms: {len(H_qubit_orig.terms)}")
    
    # Apply qubit BLISS
    print(f"\n--- Applying Qubit BLISS ---")
    start_time = time.time()
    try:
        H_bliss_qubit = bliss_qubit_two_body(
            H_fermion, 
            N=molecule['N'], 
            Ne=molecule['Ne'], 
            mapping=mapping
        )
        optimization_time = time.time() - start_time
        
        # Compute BLISS Pauli 1-norm
        bliss_pauli_norm = compute_one_norm_qubit(H_bliss_qubit)
        pauli_reduction = (orig_pauli_norm - bliss_pauli_norm) / orig_pauli_norm * 100
        
        print(f"Optimization time: {optimization_time:.2f} seconds")
        print(f"BLISS Pauli 1-norm: {bliss_pauli_norm:.6f}")
        print(f"Pauli reduction: {pauli_reduction:.2f}%")
        print(f"BLISS Pauli terms: {len(H_bliss_qubit.terms)}")
        
        return {
            'molecule': molecule,
            'mapping': mapping,
            'original_pauli_norm': orig_pauli_norm,
            'bliss_pauli_norm': bliss_pauli_norm,
            'pauli_reduction_percent': pauli_reduction,
            'original_pauli_terms': len(H_qubit_orig.terms),
            'bliss_pauli_terms': len(H_bliss_qubit.terms),
            'optimization_time': optimization_time
        }
        
    except Exception as e:
        print(f"Error in qubit BLISS optimization: {e}")
        return None


def analyze_pauli_terms(H_qubit, max_terms=10):
    """Analyze the structure of Pauli terms."""
    print(f"\n--- Pauli Term Analysis ---")
    print(f"Total terms: {len(H_qubit.terms)}")
    
    # Sort terms by coefficient magnitude
    sorted_terms = sorted(H_qubit.terms.items(), 
                         key=lambda x: abs(x[1]), reverse=True)
    
    print(f"Top {min(max_terms, len(sorted_terms))} terms by coefficient magnitude:")
    for i, (term, coeff) in enumerate(sorted_terms[:max_terms]):
        term_str = str(term) if term else "Identity"
        print(f"  {i+1:2d}. {term_str:30s} : {coeff:12.6f}")


def test_all_molecules_qubit_bliss():
    """Test qubit BLISS on all available molecular Hamiltonians."""
    print("=== Qubit BLISS Testing with Pauli 1-Norm Analysis ===")
    print("Testing qubit space BLISS on real molecular data\n")
    
    # List of molecular files to test
    molecular_files = [
        "h4_fer.bin",    # H4 (smallest)
        "lih_fer.bin",   # LiH
        "beh2_fer.bin",  # BeH2
    ]
    
    # Test both mappings
    mappings = ['bravyi_kitaev', 'jordan_wigner']
    
    all_results = {}
    
    for filename in molecular_files:
        if not os.path.exists(os.path.join("ham_lib", filename)):
            print(f"Warning: {filename} not found, skipping...")
            continue
        
        all_results[filename] = {}
        
        for mapping in mappings:
            result = test_qubit_bliss_pauli_norm(filename, mapping)
            if result:
                all_results[filename][mapping] = result
                
                # Analyze terms for smaller molecules
                if result['molecule']['N'] <= 8:  # Only for smaller systems
                    try:
                        H_fermion = load_molecular_hamiltonian(filename)
                        H_bliss_qubit = bliss_qubit_two_body(
                            H_fermion, 
                            N=result['molecule']['N'], 
                            Ne=result['molecule']['Ne'], 
                            mapping=mapping
                        )
                        analyze_pauli_terms(H_bliss_qubit)
                    except Exception as e:
                        print(f"Error in term analysis: {e}")
    
    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    
    if all_results:
        print(f"{'Molecule':<10} {'Mapping':<15} {'N':<3} {'Ne':<3} {'Pauli%':<10} {'Time(s)':<8} {'Terms':<15}")
        print("-" * 100)
        
        for filename, molecule_results in all_results.items():
            molecule_name = get_molecule_info(filename)['name']
            for mapping, result in molecule_results.items():
                print(f"{molecule_name:<10} {mapping.upper():<15} {result['molecule']['N']:<3} {result['molecule']['Ne']:<3} "
                      f"{result['pauli_reduction_percent']:<10.2f} {result['optimization_time']:<8.2f} "
                      f"{result['original_pauli_terms']}→{result['bliss_pauli_terms']}")
        
        # Find best performing combination
        best_result = None
        best_reduction = -float('inf')
        
        for filename, molecule_results in all_results.items():
            for mapping, result in molecule_results.items():
                if result['pauli_reduction_percent'] > best_reduction:
                    best_reduction = result['pauli_reduction_percent']
                    best_result = result
        
        if best_result:
            print(f"\nBest performing combination:")
            print(f"  Molecule: {best_result['molecule']['name']}")
            print(f"  Mapping: {best_result['mapping'].upper()}")
            print(f"  Pauli reduction: {best_result['pauli_reduction_percent']:.2f}%")
            print(f"  Optimization time: {best_result['optimization_time']:.2f}s")
    
    print(f"\n✅ Qubit BLISS testing with Pauli 1-norm analysis completed!")


def main():
    test_all_molecules_qubit_bliss()


if __name__ == "__main__":
    main()
