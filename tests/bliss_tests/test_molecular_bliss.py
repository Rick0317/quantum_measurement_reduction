#!/usr/bin/env python3
"""
Test qubit space BLISS with actual molecular data from ham_lib/ folder.
"""

import sys
import os
import pickle
import time

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openfermion import FermionOperator, InteractionOperator, get_fermion_operator
from bliss.qubit_bliss import bliss_qubit_two_body, compare_fermion_qubit_bliss, compute_one_norm_qubit
from bliss.bliss_main import bliss_two_body


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
        "beh2_fer.bin": {"name": "BeH2", "N": 14, "Ne": 6},  # BeH2: 6 spin-orbitals, 4 electrons
        "h4_fer.bin": {"name": "H4", "N": 8, "Ne": 4},      # H4: 8 spin-orbitals, 4 electrons  
        "lih_fer.bin": {"name": "LiH", "N": 12, "Ne": 4},    # LiH: 6 spin-orbitals, 4 electrons
    }
    return molecule_info.get(filename, {"name": "Unknown", "N": 4, "Ne": 2})


def test_molecular_bliss(filename, mapping='bravyi_kitaev'):
    """Test BLISS on a molecular Hamiltonian."""
    print(f"\n{'='*60}")
    print(f"Testing {filename} with {mapping.upper()} mapping")
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
    
    # Test qubit space BLISS
    print(f"\n--- Qubit Space BLISS ---")
    start_time = time.time()
    try:
        H_bliss_qubit = bliss_qubit_two_body(
            H_fermion, 
            N=molecule['N'], 
            Ne=molecule['Ne'], 
            mapping=mapping
        )
        qubit_time = time.time() - start_time
        
        # Compare results
        comparison = compare_fermion_qubit_bliss(
            H_fermion, 
            N=molecule['N'], 
            Ne=molecule['Ne'], 
            mapping=mapping
        )
        
        print(f"Optimization time: {qubit_time:.2f} seconds")
        print(f"Original 1-norm: {comparison['original_norm']:.6f}")
        print(f"BLISS 1-norm: {comparison['bliss_norm']:.6f}")
        print(f"Reduction: {comparison['reduction_percent']:.2f}%")
        print(f"Original terms: {comparison['original_terms']}")
        print(f"BLISS terms: {comparison['bliss_terms']}")
        
        qubit_results = {
            'time': qubit_time,
            'reduction': comparison['reduction_percent'],
            'original_norm': comparison['original_norm'],
            'bliss_norm': comparison['bliss_norm'],
            'original_terms': comparison['original_terms'],
            'bliss_terms': comparison['bliss_terms']
        }
        
    except Exception as e:
        print(f"Error in qubit BLISS: {e}")
        qubit_results = None
    
    # Test fermionic BLISS for comparison
    print(f"\n--- Fermionic BLISS (for comparison) ---")
    start_time = time.time()
    try:
        H_bliss_fermion = bliss_two_body(
            H_fermion, 
            N=molecule['N'], 
            Ne=molecule['Ne']
        )
        fermion_time = time.time() - start_time
        
        # Convert to qubit space for norm comparison
        from bliss.qubit_bliss import fermion_to_qubit_bliss
        H_bliss_fermion_qubit = fermion_to_qubit_bliss(H_bliss_fermion, mapping)
        fermion_norm = compute_one_norm_qubit(H_bliss_fermion_qubit)
        
        print(f"Optimization time: {fermion_time:.2f} seconds")
        print(f"Fermionic BLISS 1-norm (in qubit space): {fermion_norm:.6f}")
        print(f"Fermionic BLISS terms: {len(H_bliss_fermion.terms)}")
        
        fermion_results = {
            'time': fermion_time,
            'norm': fermion_norm,
            'terms': len(H_bliss_fermion.terms)
        }
        
    except Exception as e:
        print(f"Error in fermionic BLISS: {e}")
        fermion_results = None
    
    return {
        'molecule': molecule,
        'qubit_results': qubit_results,
        'fermion_results': fermion_results
    }


def main():
    print("=== Molecular BLISS Testing ===")
    print("Testing qubit space BLISS with actual molecular data\n")
    
    # List of molecular files to test
    molecular_files = [
        "h4_fer.bin",    # H4
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
            results = test_molecular_bliss(filename, mapping)
            if results:
                all_results[filename][mapping] = results
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    for filename, molecule_results in all_results.items():
        molecule_name = get_molecule_info(filename)['name']
        print(f"\n{molecule_name} ({filename}):")
        
        for mapping, results in molecule_results.items():
            if results and results['qubit_results']:
                qr = results['qubit_results']
                print(f"  {mapping.upper()}: {qr['reduction']:.2f}% reduction, "
                      f"{qr['time']:.2f}s, {qr['original_terms']}→{qr['bliss_terms']} terms")
    
    print(f"\n✅ Molecular BLISS testing completed!")


if __name__ == "__main__":
    main()
