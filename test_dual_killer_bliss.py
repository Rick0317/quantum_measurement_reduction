#!/usr/bin/env python3
"""
Test dual killer BLISS implementation.
"""

import sys
import os
import pickle
import time

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openfermion import FermionOperator, InteractionOperator, get_fermion_operator
from bliss.qubit_bliss.qubit_bliss_dual_killer import (
    bliss_qubit_dual_killer_two_body,
    compare_dual_killer_bliss_results,
    analyze_dual_killer_parameters
)
from bliss.qubit_bliss.qubit_bliss_main import (
    bliss_qubit_two_body,
    compare_fermion_qubit_bliss
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


def test_dual_killer_vs_single_killer(filename, mapping='bravyi_kitaev'):
    """Compare dual killer vs single killer BLISS."""
    print(f"\n{'='*80}")
    print(f"Comparing Dual Killer vs Single Killer BLISS for {filename}")
    print(f"Mapping: {mapping.upper()}")
    print(f"{'='*80}")
    
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
    
    # Test single killer BLISS
    print(f"\n--- Single Killer BLISS ---")
    start_time = time.time()
    try:
        single_result = compare_fermion_qubit_bliss(
            H_fermion, 
            N=molecule['N'], 
            Ne=molecule['Ne'], 
            mapping=mapping
        )
        single_time = time.time() - start_time
        
        print(f"Optimization time: {single_time:.2f} seconds")
        print(f"Original 1-norm: {single_result['original_norm']:.6f}")
        print(f"BLISS 1-norm: {single_result['bliss_norm']:.6f}")
        print(f"Reduction: {single_result['reduction_percent']:.2f}%")
        print(f"Original terms: {single_result['original_terms']}")
        print(f"BLISS terms: {single_result['bliss_terms']}")
        
    except Exception as e:
        print(f"Error in single killer BLISS: {e}")
        single_result = None
    
    # Test dual killer BLISS
    print(f"\n--- Dual Killer BLISS ---")
    start_time = time.time()
    try:
        dual_result = compare_dual_killer_bliss_results(
            H_fermion, 
            N=molecule['N'], 
            Ne=molecule['Ne'], 
            mapping=mapping
        )
        dual_time = time.time() - start_time
        
        print(f"Optimization time: {dual_time:.2f} seconds")
        print(f"Original 1-norm: {dual_result['original_norm']:.6f}")
        print(f"BLISS 1-norm: {dual_result['bliss_norm']:.6f}")
        print(f"Reduction: {dual_result['reduction_percent']:.2f}%")
        print(f"Original terms: {dual_result['original_terms']}")
        print(f"BLISS terms: {dual_result['bliss_terms']}")
        
    except Exception as e:
        print(f"Error in dual killer BLISS: {e}")
        dual_result = None
    
    # Compare results
    if single_result and dual_result:
        print(f"\n--- Comparison ---")
        improvement = dual_result['reduction_percent'] - single_result['reduction_percent']
        print(f"Single killer reduction: {single_result['reduction_percent']:.2f}%")
        print(f"Dual killer reduction: {dual_result['reduction_percent']:.2f}%")
        print(f"Improvement: {improvement:.2f} percentage points")
        
        if improvement > 0:
            print("✅ Dual killer performs better!")
        elif improvement < 0:
            print("❌ Single killer performs better!")
        else:
            print("➖ Both perform equally!")
    
    return {
        'single_result': single_result,
        'dual_result': dual_result,
        'improvement': improvement if single_result and dual_result else None
    }


def test_all_molecules_dual_killer():
    """Test dual killer BLISS on all available molecular Hamiltonians."""
    print("=== Dual Killer BLISS Testing ===")
    print("Comparing dual killer vs single killer BLISS on molecular data\n")
    
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
            result = test_dual_killer_vs_single_killer(filename, mapping)
            if result:
                all_results[filename][mapping] = result
    
    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    
    if all_results:
        print(f"{'Molecule':<10} {'Mapping':<15} {'Single%':<10} {'Dual%':<10} {'Improvement':<12}")
        print("-" * 100)
        
        for filename, molecule_results in all_results.items():
            molecule_name = get_molecule_info(filename)['name']
            for mapping, result in molecule_results.items():
                if result['single_result'] and result['dual_result']:
                    single_reduction = result['single_result']['reduction_percent']
                    dual_reduction = result['dual_result']['reduction_percent']
                    improvement = result['improvement']
                    
                    print(f"{molecule_name:<10} {mapping.upper():<15} {single_reduction:<10.2f} "
                          f"{dual_reduction:<10.2f} {improvement:<12.2f}")
        
        # Find best performing combination
        best_improvement = -float('inf')
        best_combination = None
        
        for filename, molecule_results in all_results.items():
            for mapping, result in molecule_results.items():
                if result['improvement'] and result['improvement'] > best_improvement:
                    best_improvement = result['improvement']
                    best_combination = (filename, mapping)
        
        if best_combination:
            filename, mapping = best_combination
            molecule_name = get_molecule_info(filename)['name']
            print(f"\nBest improvement: {molecule_name} with {mapping.upper()} mapping")
            print(f"Improvement: {best_improvement:.2f} percentage points")
    
    print(f"\n✅ Dual killer BLISS testing completed!")


def main():
    test_all_molecules_dual_killer()


if __name__ == "__main__":
    main()
