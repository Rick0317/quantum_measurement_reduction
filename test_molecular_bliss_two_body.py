#!/usr/bin/env python3
"""
Test bliss_two_body function on molecular Hamiltonians from ham_lib/ folder.
"""

import sys
import os
import pickle
import time

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openfermion import FermionOperator, InteractionOperator, get_fermion_operator
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
        "beh2_fer.bin": {"name": "BeH2", "N": 14, "Ne": 6},
        "h4_fer.bin": {"name": "H4", "N": 8, "Ne": 4},
        "lih_fer.bin": {"name": "LiH", "N": 12, "Ne": 4},
        "n2_fer.bin": {"name": "N2", "N": 20, "Ne": 10},
    }
    return molecule_info.get(filename, {"name": "Unknown", "N": 4, "Ne": 2})


def compute_fermion_operator_norm(H_fermion):
    """Compute the 1-norm of a FermionOperator."""
    return sum(abs(coeff) for coeff in H_fermion.terms.values())


def compute_majorana_operator_norm(H_fermion):
    """Compute the 1-norm of a FermionOperator in Majorana basis."""
    from bliss.majorana.custom_majorana_transform import get_custom_majorana_operator
    majorana_op = get_custom_majorana_operator(H_fermion)
    return sum(abs(coeff) for coeff in majorana_op.terms.values())


def test_bliss_two_body_molecule(filename):
    """Test bliss_two_body on a single molecule."""
    print(f"\n{'='*60}")
    print(f"Testing {filename} with bliss_two_body")
    print(f"{'='*60}")
    
    # Load molecular data
    try:
        H_fermion = load_molecular_hamiltonian(filename)
        molecule = get_molecule_info(filename)
        print(f"Molecule: {molecule['name']}")
        print(f"Spin-orbitals: {molecule['N']}, Electrons: {molecule['Ne']}")
        print(f"Original Hamiltonian terms: {len(H_fermion.terms)}")
        
        # Compute original 1-norm (both fermionic and Majorana)
        orig_fermion_norm = compute_fermion_operator_norm(H_fermion)
        orig_majorana_norm = compute_majorana_operator_norm(H_fermion)
        print(f"Original fermionic 1-norm: {orig_fermion_norm:.6f}")
        print(f"Original Majorana 1-norm: {orig_majorana_norm:.6f}")
        
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None
    
    # Apply BLISS
    print(f"\n--- Applying BLISS ---")
    start_time = time.time()
    try:
        H_bliss = bliss_two_body(H_fermion, molecule['N'], molecule['Ne'])
        optimization_time = time.time() - start_time
        
        # Compute BLISS 1-norm (both fermionic and Majorana)
        bliss_fermion_norm = compute_fermion_operator_norm(H_bliss)
        bliss_majorana_norm = compute_majorana_operator_norm(H_bliss)
        
        # Calculate reductions
        fermion_reduction = (orig_fermion_norm - bliss_fermion_norm) / orig_fermion_norm * 100
        majorana_reduction = (orig_majorana_norm - bliss_majorana_norm) / orig_majorana_norm * 100
        
        print(f"Optimization time: {optimization_time:.2f} seconds")
        print(f"BLISS fermionic 1-norm: {bliss_fermion_norm:.6f}")
        print(f"BLISS Majorana 1-norm: {bliss_majorana_norm:.6f}")
        print(f"Fermionic reduction: {fermion_reduction:.2f}%")
        print(f"Majorana reduction: {majorana_reduction:.2f}%")
        print(f"BLISS terms: {len(H_bliss.terms)}")
        
        return {
            'molecule': molecule,
            'original_fermion_norm': orig_fermion_norm,
            'original_majorana_norm': orig_majorana_norm,
            'bliss_fermion_norm': bliss_fermion_norm,
            'bliss_majorana_norm': bliss_majorana_norm,
            'fermion_reduction_percent': fermion_reduction,
            'majorana_reduction_percent': majorana_reduction,
            'original_terms': len(H_fermion.terms),
            'bliss_terms': len(H_bliss.terms),
            'optimization_time': optimization_time
        }
        
    except Exception as e:
        print(f"Error in BLISS optimization: {e}")
        return None


def analyze_bliss_terms(H_fermion, H_bliss, max_terms=10):
    """Analyze the structure of BLISS terms."""
    print(f"\n--- Term Analysis ---")
    
    # Sort original terms by coefficient magnitude
    orig_terms = sorted(H_fermion.terms.items(), key=lambda x: abs(x[1]), reverse=True)
    bliss_terms = sorted(H_bliss.terms.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"Top {min(max_terms, len(orig_terms))} original terms by coefficient magnitude:")
    for i, (term, coeff) in enumerate(orig_terms[:max_terms]):
        term_str = str(term) if term else "Identity"
        print(f"  {i+1:2d}. {term_str:30s} : {coeff:12.6f}")
    
    print(f"\nTop {min(max_terms, len(bliss_terms))} BLISS terms by coefficient magnitude:")
    for i, (term, coeff) in enumerate(bliss_terms[:max_terms]):
        term_str = str(term) if term else "Identity"
        print(f"  {i+1:2d}. {term_str:30s} : {coeff:12.6f}")


def test_all_molecules():
    """Test bliss_two_body on all available molecular Hamiltonians."""
    print("=== BLISS Two-Body Testing on Molecular Hamiltonians ===")
    print("Testing the original bliss_two_body function on real molecular data\n")
    
    # List of molecular files to test
    molecular_files = [
        "h4_fer.bin",    # H4 (smallest)
        "lih_fer.bin",   # LiH
        "beh2_fer.bin",  # BeH2
        "n2_fer.bin",    # N2 (largest)
    ]
    
    results = {}
    
    for filename in molecular_files:
        if not os.path.exists(os.path.join("ham_lib", filename)):
            print(f"Warning: {filename} not found, skipping...")
            continue
        
        result = test_bliss_two_body_molecule(filename)
        if result:
            results[filename] = result
            
            # Analyze terms for smaller molecules
            if result['molecule']['N'] <= 12:  # Only for smaller systems
                try:
                    H_fermion = load_molecular_hamiltonian(filename)
                    H_bliss = bliss_two_body(H_fermion, result['molecule']['N'], result['molecule']['Ne'])
                    analyze_bliss_terms(H_fermion, H_bliss)
                except Exception as e:
                    print(f"Error in term analysis: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if results:
        print(f"{'Molecule':<10} {'N':<3} {'Ne':<3} {'Fermion%':<10} {'Majorana%':<12} {'Time(s)':<8} {'Terms':<15}")
        print("-" * 90)
        
        for filename, result in results.items():
            molecule = result['molecule']
            print(f"{molecule['name']:<10} {molecule['N']:<3} {molecule['Ne']:<3} "
                  f"{result['fermion_reduction_percent']:<10.2f} {result['majorana_reduction_percent']:<12.2f} "
                  f"{result['optimization_time']:<8.2f} {result['original_terms']}→{result['bliss_terms']}")
        
        # Find best performing molecule (based on Majorana reduction)
        best_molecule = max(results.keys(), key=lambda f: results[f]['majorana_reduction_percent'])
        best_majorana_reduction = results[best_molecule]['majorana_reduction_percent']
        best_fermion_reduction = results[best_molecule]['fermion_reduction_percent']
        print(f"\nBest performing molecule: {get_molecule_info(best_molecule)['name']}")
        print(f"  Majorana reduction: {best_majorana_reduction:.2f}%")
        print(f"  Fermionic reduction: {best_fermion_reduction:.2f}%")
    
    print(f"\n✅ BLISS two-body testing completed!")


def main():
    test_all_molecules()


if __name__ == "__main__":
    main()
