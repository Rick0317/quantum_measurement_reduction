#!/usr/bin/env python3
"""
Test script for killer operator BLISS implementation.
"""

import sys
import os
import pickle
import time

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openfermion import FermionOperator, InteractionOperator, get_fermion_operator
from killer_operator_bliss import (
    bliss_killer_operator_qubit,
    compare_killer_bliss_results,
    multi_mode_killer_bliss,
    analyze_killer_operator_structure
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
    }
    return molecule_info.get(filename, {"name": "Unknown", "N": 4, "Ne": 2})


def test_single_killer_operator(filename, mode, mapping='bravyi_kitaev'):
    """Test killer operator BLISS on a single mode."""
    print(f"\n{'='*60}")
    print(f"Testing {filename} with killer operator (O a_{mode})_q")
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
    
    # Test killer operator BLISS
    print(f"\n--- Killer Operator BLISS ---")
    start_time = time.time()
    try:
        result = compare_killer_bliss_results(
            H_fermion, 
            mode, 
            N=molecule['N'], 
            Ne=molecule['Ne'], 
            mapping=mapping
        )
        optimization_time = time.time() - start_time
        
        print(f"Optimization time: {optimization_time:.2f} seconds")
        print(f"Original 1-norm: {result['original_norm']:.6f}")
        print(f"BLISS 1-norm: {result['bliss_norm']:.6f}")
        print(f"Reduction: {result['reduction_percent']:.2f}%")
        print(f"Original terms: {result['original_terms']}")
        print(f"BLISS terms: {result['bliss_terms']}")
        
        return result
        
    except Exception as e:
        print(f"Error in killer operator BLISS: {e}")
        return None


def test_multiple_modes(filename, modes, mapping='bravyi_kitaev'):
    """Test killer operator BLISS on multiple modes."""
    print(f"\n{'='*60}")
    print(f"Testing {filename} with multiple killer operators")
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
    
    # Test multiple modes
    results = multi_mode_killer_bliss(
        H_fermion, 
        modes, 
        N=molecule['N'], 
        Ne=molecule['Ne'], 
        mapping=mapping
    )
    
    # Summary
    print(f"\n--- Summary ---")
    for mode, result in results.items():
        if result:
            print(f"Mode {mode}: {result['reduction_percent']:.2f}% reduction, "
                  f"{result['original_terms']}→{result['bliss_terms']} terms")
        else:
            print(f"Mode {mode}: Failed")
    
    return results


def analyze_killer_operator_components(filename, mode, mapping='bravyi_kitaev'):
    """Analyze the structure of killer operator components."""
    print(f"\n{'='*60}")
    print(f"Analyzing killer operator components for {filename}")
    print(f"Mode: {mode}, Mapping: {mapping.upper()}")
    print(f"{'='*60}")
    
    # Load molecular data
    try:
        H_fermion = load_molecular_hamiltonian(filename)
        molecule = get_molecule_info(filename)
        
        # Convert to qubit space
        from killer_operator_bliss import fermion_to_qubit_bliss
        H_qubit = fermion_to_qubit_bliss(H_fermion, mapping)
        
        # Create dummy parameters for analysis
        import numpy as np
        params = np.random.random(int(molecule['N'] * (molecule['N'] + 1) // 2)) * 0.01
        
        # Analyze structure
        analysis = analyze_killer_operator_structure(
            H_qubit, params, mode, molecule['N'], mapping
        )
        
        print(f"Parameterized operator O:")
        print(f"  Terms: {analysis['O_terms']}")
        print(f"  1-norm: {analysis['O_norm']:.6f}")
        
        print(f"Annihilation operator a_{mode}:")
        print(f"  Terms: {analysis['a_terms']}")
        print(f"  1-norm: {analysis['a_norm']:.6f}")
        
        print(f"Killer operator (O a_{mode})_q:")
        print(f"  Terms: {analysis['killer_terms']}")
        print(f"  1-norm: {analysis['killer_norm']:.6f}")
        
        return analysis
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        return None


def main():
    print("=== Killer Operator BLISS Testing ===")
    print("Testing BLISS with killer operators of the form (O a_i)_q\n")
    
    # Test with H4 molecule (smaller system)
    filename = "h4_fer.bin"
    molecule = get_molecule_info(filename)
    
    if not os.path.exists(os.path.join("ham_lib", filename)):
        print(f"Warning: {filename} not found, skipping...")
        return
    
    # Test single mode
    print("1. Testing single killer operator")
    result = test_single_killer_operator(filename, mode=0, mapping='bravyi_kitaev')
    
    # Test multiple modes
    print("\n2. Testing multiple killer operators")
    modes = [0, 1, 2, 3]  # Test first 4 modes
    results = test_multiple_modes(filename, modes, mapping='bravyi_kitaev')
    
    # Analyze structure
    print("\n3. Analyzing killer operator structure")
    analysis = analyze_killer_operator_components(filename, mode=0, mapping='bravyi_kitaev')
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if results:
        best_mode = max(results.keys(), key=lambda m: results[m]['reduction_percent'] if results[m] else -1)
        best_reduction = results[best_mode]['reduction_percent'] if results[best_mode] else 0
        print(f"Best performing mode: {best_mode} ({best_reduction:.2f}% reduction)")
    
    print(f"\n✅ Killer operator BLISS testing completed!")


if __name__ == "__main__":
    main()
