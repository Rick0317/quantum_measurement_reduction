#!/usr/bin/env python3
"""
Test script for Hartree-Fock killer operator BLISS implementation.
"""

import sys
import os
import pickle
import time
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openfermion import FermionOperator, InteractionOperator, get_fermion_operator
from hartree_fock_killer_bliss import (
    bliss_killer_operator_hf_qubit,
    compare_killer_hf_bliss_results,
    multi_mode_killer_hf_bliss,
    get_hartree_fock_state,
    evaluate_killer_operator_effectiveness
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


def test_single_killer_hf(filename, mode, mapping='bravyi_kitaev'):
    """Test killer operator BLISS with Hartree-Fock evaluation on a single mode."""
    print(f"\n{'='*60}")
    print(f"Testing {filename} with killer operator (O a_{mode})_q + HF evaluation")
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
    
    # Test killer operator BLISS with HF evaluation
    print(f"\n--- Killer Operator BLISS with HF Evaluation ---")
    start_time = time.time()
    try:
        result = compare_killer_hf_bliss_results(
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
        print(f"Killer effectiveness: {result['killer_effectiveness']:.6f}")
        print(f"Killer terms: {result['killer_terms']}")
        
        return result
        
    except Exception as e:
        print(f"Error in killer operator BLISS: {e}")
        return None


def test_multiple_modes_hf(filename, modes, mapping='bravyi_kitaev'):
    """Test killer operator BLISS with Hartree-Fock evaluation on multiple modes."""
    print(f"\n{'='*60}")
    print(f"Testing {filename} with multiple killer operators + HF evaluation")
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
    results = multi_mode_killer_hf_bliss(
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
                  f"effectiveness: {result['killer_effectiveness']:.6f}, "
                  f"{result['original_terms']}→{result['bliss_terms']} terms")
        else:
            print(f"Mode {mode}: Failed")
    
    return results


def analyze_hartree_fock_state(filename, mapping='bravyi_kitaev'):
    """Analyze the Hartree-Fock state for the molecule."""
    print(f"\n{'='*60}")
    print(f"Analyzing Hartree-Fock state for {filename}")
    print(f"Mapping: {mapping.upper()}")
    print(f"{'='*60}")
    
    molecule = get_molecule_info(filename)
    
    # Get Hartree-Fock state
    hf_state = get_hartree_fock_state(molecule['N'], molecule['Ne'], mapping)
    
    print(f"Hartree-Fock state:")
    print(f"  Dimension: {len(hf_state)}")
    print(f"  Non-zero elements: {np.count_nonzero(hf_state)}")
    print(f"  Norm: {np.linalg.norm(hf_state):.6f}")
    
    # Find the non-zero element
    non_zero_indices = np.where(hf_state != 0)[0]
    if len(non_zero_indices) > 0:
        print(f"  Non-zero index: {non_zero_indices[0]}")
        print(f"  Value: {hf_state[non_zero_indices[0]]:.6f}")
    
    return hf_state


def test_killer_effectiveness_analysis(filename, mode, mapping='bravyi_kitaev'):
    """Test the effectiveness of killer operators with different parameters."""
    print(f"\n{'='*60}")
    print(f"Testing killer operator effectiveness for {filename}")
    print(f"Mode: {mode}, Mapping: {mapping.upper()}")
    print(f"{'='*60}")
    
    molecule = get_molecule_info(filename)
    
    # Get Hartree-Fock state
    hf_state = get_hartree_fock_state(molecule['N'], molecule['Ne'], mapping)
    
    # Test different parameter sets
    from hartree_fock_killer_bliss import construct_killer_operator_qubit
    import numpy as np
    
    print(f"\nTesting different parameter sets:")
    
    for i, scale in enumerate([0.01, 0.1, 1.0, 10.0]):
        # Create random parameters
        params = np.random.random(int(molecule['N'] * (molecule['N'] + 1) // 2)) * scale
        
        # Construct killer operator
        killer_qubit = construct_killer_operator_qubit(params, mode, molecule['N'], mapping)
        
        # Evaluate effectiveness
        effectiveness = evaluate_killer_operator_effectiveness(killer_qubit, hf_state)
        
        print(f"  Scale {scale}: effectiveness = {effectiveness:.6f}, "
              f"terms = {len(killer_qubit.terms)}")


def main():
    print("=== Hartree-Fock Killer Operator BLISS Testing ===")
    print("Testing BLISS with killer operators (O a_i)_q using Hartree-Fock evaluation\n")
    
    # Test with H4 molecule (smaller system)
    filename = "h4_fer.bin"
    molecule = get_molecule_info(filename)
    
    if not os.path.exists(os.path.join("ham_lib", filename)):
        print(f"Warning: {filename} not found, skipping...")
        return
    
    # Analyze Hartree-Fock state
    print("1. Analyzing Hartree-Fock state")
    hf_state = analyze_hartree_fock_state(filename, mapping='bravyi_kitaev')
    
    # Test single mode
    print("\n2. Testing single killer operator with HF evaluation")
    result = test_single_killer_hf(filename, mode=0, mapping='bravyi_kitaev')
    
    # Test multiple modes
    print("\n3. Testing multiple killer operators with HF evaluation")
    modes = [0, 1, 2, 3]  # Test first 4 modes
    results = test_multiple_modes_hf(filename, modes, mapping='bravyi_kitaev')
    
    # Test effectiveness analysis
    print("\n4. Testing killer operator effectiveness analysis")
    test_killer_effectiveness_analysis(filename, mode=0, mapping='bravyi_kitaev')
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if results:
        best_mode = max(results.keys(), key=lambda m: results[m]['reduction_percent'] if results[m] else -1)
        best_reduction = results[best_mode]['reduction_percent'] if results[best_mode] else 0
        best_effectiveness = results[best_mode]['killer_effectiveness'] if results[best_mode] else 0
        print(f"Best performing mode: {best_mode}")
        print(f"  Reduction: {best_reduction:.2f}%")
        print(f"  Effectiveness: {best_effectiveness:.6f}")
    
    print(f"\n✅ Hartree-Fock killer operator BLISS testing completed!")


if __name__ == "__main__":
    main()
