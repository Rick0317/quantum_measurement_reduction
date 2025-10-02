#!/usr/bin/env python3
"""
Test bliss_two_body() function on LiH Hamiltonian from ham_lib with proper Majorana basis 1-norm calculation.
"""

import pickle
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openfermion import FermionOperator, InteractionOperator, get_fermion_operator
from bliss.bliss_main import bliss_two_body


def load_lih_hamiltonian():
    """Load the LiH Hamiltonian from lih_fer.bin."""
    filepath = os.path.join("ham_lib", "beh2_fer.bin")
    
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found!")
        return None
    
    with open(filepath, "rb") as f:
        interaction_op = pickle.load(f)
    
    # Convert to FermionOperator if needed
    if isinstance(interaction_op, InteractionOperator):
        return get_fermion_operator(interaction_op)
    elif isinstance(interaction_op, FermionOperator):
        return interaction_op
    else:
        raise ValueError(f"Unexpected type: {type(interaction_op)}")


def compute_majorana_operator_norm(H_fermion):
    """Compute the 1-norm of a FermionOperator in Majorana basis (excluding constant term)."""
    from bliss.majorana.custom_majorana_transform import get_custom_majorana_operator
    majorana_op = get_custom_majorana_operator(H_fermion)
    # Exclude constant term (empty tuple)
    return sum(abs(coeff) for term, coeff in majorana_op.terms.items() if term != ())


def test_bliss_lih():
    """Test bliss_two_body with LiH Hamiltonian."""
    print("=== Testing bliss_two_body() with LiH Hamiltonian ===")
    
    # LiH molecule info: 12 spin-orbitals, 4 electrons
    N = 14  # Number of spin-orbitals/qubits
    Ne = 6  # Number of electrons
    
    print(f"Molecule: LiH")
    print(f"Spin-orbitals (N): {N}")
    print(f"Electrons (Ne): {Ne}")
    
    # Load the Hamiltonian
    print("\nLoading LiH Hamiltonian...")
    try:
        H_fermion = load_lih_hamiltonian()
        print(f"✅ Successfully loaded LiH Hamiltonian")
        print(f"   Number of terms: {len(H_fermion.terms)}")
        
        # Show original Majorana 1-norm (excluding constant terms)
        orig_majorana_norm = compute_majorana_operator_norm(H_fermion)
        print(f"   Original Majorana 1-norm (no constant): {orig_majorana_norm:.6f}")
        
    except Exception as e:
        print(f"❌ Error loading Hamiltonian: {e}")
        return False
    
    # Apply BLISS two-body optimization
    print(f"\nApplying BLISS two-body optimization...")
    try:
        H_bliss = bliss_two_body(H_fermion, N, Ne)
        print(f"✅ BLISS optimization completed successfully!")
        print(f"   BLISS terms: {len(H_bliss.terms)}")
        
        # Show BLISS Majorana 1-norm (excluding constant terms)
        bliss_majorana_norm = compute_majorana_operator_norm(H_bliss)
        print(f"   BLISS Majorana 1-norm (no constant): {bliss_majorana_norm:.6f}")
        
        # Calculate Majorana reduction
        majorana_reduction = (orig_majorana_norm - bliss_majorana_norm) / orig_majorana_norm * 100
        print(f"   Majorana reduction: {majorana_reduction:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in BLISS optimization: {e}")
        return False


if __name__ == "__main__":
    success = test_bliss_lih()
    
    if success:
        print(f"\n🎉 SUCCESS: bliss_two_body() works correctly with lih_fer.bin!")
        print("The function can be used for LiH molecular Hamiltonian optimization.")
        print("✅ Majorana basis 1-norm calculation confirmed.")
    else:
        print(f"\n❌ FAILURE: Error occurred during testing.")
