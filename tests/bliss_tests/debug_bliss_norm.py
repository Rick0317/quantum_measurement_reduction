#!/usr/bin/env python3
"""
Debug script to understand why BLISS is increasing the 1-norm.
"""

import sys
import os
import pickle
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openfermion import FermionOperator, InteractionOperator, get_fermion_operator
from bliss.bliss_main import bliss_two_body
from bliss.normal_bliss.bliss_package import construct_H_bliss_m12_o1, optimization_bliss_mu12_o1


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


def compute_fermion_operator_norm(H_fermion):
    """Compute the 1-norm of a FermionOperator."""
    return sum(abs(coeff) for coeff in H_fermion.terms.values())


def debug_bliss_optimization(filename="h4_fer.bin"):
    """Debug the BLISS optimization process step by step."""
    print(f"=== Debugging BLISS Optimization for {filename} ===")
    
    # Load molecular data
    H_fermion = load_molecular_hamiltonian(filename)
    N = 8  # H4
    Ne = 4
    
    print(f"Original Hamiltonian: {len(H_fermion.terms)} terms")
    orig_norm = compute_fermion_operator_norm(H_fermion)
    print(f"Original 1-norm: {orig_norm:.6f}")
    
    # Get optimization wrapper and initial guess
    optimization_wrapper, initial_guess = optimization_bliss_mu12_o1(H_fermion, N, Ne)
    
    print(f"Initial guess length: {len(initial_guess)}")
    print(f"Initial guess: {initial_guess[:5]}...")  # Show first 5 elements
    
    # Test the optimization wrapper with initial guess
    initial_objective = optimization_wrapper(initial_guess)
    print(f"Initial objective value: {initial_objective:.6f}")
    
    # Test with zero parameters
    zero_params = np.zeros_like(initial_guess)
    zero_objective = optimization_wrapper(zero_params)
    print(f"Zero parameters objective: {zero_objective:.6f}")
    
    # Test with small random parameters
    small_params = np.random.random(len(initial_guess)) * 0.001
    small_objective = optimization_wrapper(small_params)
    print(f"Small random parameters objective: {small_objective:.6f}")
    
    # Construct BLISS Hamiltonian with initial guess
    H_bliss_initial = construct_H_bliss_m12_o1(H_fermion, initial_guess, N, Ne)
    bliss_initial_norm = compute_fermion_operator_norm(H_bliss_initial)
    print(f"BLISS with initial guess 1-norm: {bliss_initial_norm:.6f}")
    
    # Construct BLISS Hamiltonian with zero parameters
    H_bliss_zero = construct_H_bliss_m12_o1(H_fermion, zero_params, N, Ne)
    bliss_zero_norm = compute_fermion_operator_norm(H_bliss_zero)
    print(f"BLISS with zero parameters 1-norm: {bliss_zero_norm:.6f}")
    
    # Check if zero parameters should give original Hamiltonian
    print(f"Zero params BLISS == Original: {np.isclose(bliss_zero_norm, orig_norm)}")
    
    # Analyze the killer operator
    print(f"\n--- Analyzing Killer Operator ---")
    
    # The killer operator is: μ₁(N̂ - Ne) + μ₂(N̂² - Ne²) + Ô(N̂ - Ne)
    # where μ₁, μ₂ are scalars and Ô is a one-body operator
    
    mu1 = initial_guess[0]
    mu2 = initial_guess[1]
    o_params = initial_guess[2:]
    
    print(f"μ₁ (mu1): {mu1:.6f}")
    print(f"μ₂ (mu2): {mu2:.6f}")
    print(f"O parameters: {o_params[:5]}...")  # Show first 5
    
    # Check if the killer operator is too large
    killer_norm = abs(mu1) + abs(mu2) + np.sum(np.abs(o_params))
    print(f"Killer operator parameter norm: {killer_norm:.6f}")
    
    return {
        'original_norm': orig_norm,
        'initial_objective': initial_objective,
        'zero_objective': zero_objective,
        'bliss_initial_norm': bliss_initial_norm,
        'bliss_zero_norm': bliss_zero_norm,
        'killer_norm': killer_norm
    }


def test_different_initial_guesses(filename="h4_fer.bin"):
    """Test BLISS with different initial guesses."""
    print(f"\n=== Testing Different Initial Guesses for {filename} ===")
    
    H_fermion = load_molecular_hamiltonian(filename)
    N = 8
    Ne = 4
    
    orig_norm = compute_fermion_operator_norm(H_fermion)
    print(f"Original 1-norm: {orig_norm:.6f}")
    
    # Get optimization wrapper
    optimization_wrapper, _ = optimization_bliss_mu12_o1(H_fermion, N, Ne)
    
    # Test different initial guesses
    initial_guesses = [
        ("Zero", np.zeros(2 + int(N * (N + 1) // 2))),
        ("Small", np.random.random(2 + int(N * (N + 1) // 2)) * 0.001),
        ("Medium", np.random.random(2 + int(N * (N + 1) // 2)) * 0.01),
        ("Large", np.random.random(2 + int(N * (N + 1) // 2)) * 0.1),
    ]
    
    results = {}
    
    for name, guess in initial_guesses:
        objective = optimization_wrapper(guess)
        H_bliss = construct_H_bliss_m12_o1(H_fermion, guess, N, Ne)
        bliss_norm = compute_fermion_operator_norm(H_bliss)
        reduction = (orig_norm - bliss_norm) / orig_norm * 100
        
        results[name] = {
            'objective': objective,
            'bliss_norm': bliss_norm,
            'reduction': reduction
        }
        
        print(f"{name:6s}: objective={objective:.6f}, BLISS norm={bliss_norm:.6f}, "
              f"reduction={reduction:.2f}%")
    
    return results


def main():
    print("=== BLISS 1-Norm Increase Debug ===")
    print("Investigating why BLISS is increasing the 1-norm instead of decreasing it\n")
    
    # Debug the optimization process
    debug_results = debug_bliss_optimization("h4_fer.bin")
    
    # Test different initial guesses
    guess_results = test_different_initial_guesses("h4_fer.bin")
    
    # Analysis
    print(f"\n--- Analysis ---")
    print(f"Original norm: {debug_results['original_norm']:.6f}")
    print(f"Initial objective: {debug_results['initial_objective']:.6f}")
    print(f"Zero objective: {debug_results['zero_objective']:.6f}")
    
    if debug_results['zero_objective'] > debug_results['original_norm']:
        print("⚠️  Zero parameters give higher objective than original norm!")
        print("   This suggests the optimization is not working correctly.")
    
    if debug_results['bliss_zero_norm'] != debug_results['original_norm']:
        print("⚠️  Zero parameters should give original Hamiltonian!")
        print("   This suggests a bug in the BLISS construction.")
    
    print(f"\n✅ Debug completed!")


if __name__ == "__main__":
    main()
