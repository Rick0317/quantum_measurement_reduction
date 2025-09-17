#!/usr/bin/env python3
"""
Simple example demonstrating Ghost Pauli method usage.
This script shows how to use the Ghost Pauli functionality in a real quantum computing scenario.
"""

import numpy as np
import scipy.sparse as sp
from openfermion import QubitOperator, get_sparse_operator, get_ground_state

def simple_ghost_pauli_example():
    """
    Simple example of using Ghost Pauli method for quantum measurement reduction.
    """
    
    print("=" * 60)
    print("Ghost Pauli Method - Simple Example")
    print("=" * 60)
    
    try:
        # Import Ghost Pauli functions
        from quantum_measurement_reduction.ghost_pauli import update_decomp_w_ghost_paulis_sparse
        from quantum_measurement_reduction.ghost_pauli import sparse_variance, sparse_expectation
        
        print("✓ Ghost Pauli functions imported successfully")
        
        # Step 1: Create a simple quantum Hamiltonian
        print("\n1. Creating quantum Hamiltonian...")
        H = QubitOperator('X0 Y1') + QubitOperator('Z0 X1') + QubitOperator('Y0 Z1')
        print(f"   Hamiltonian: {H}")
        
        # Step 2: Get the ground state
        print("\n2. Computing ground state...")
        H_sparse = get_sparse_operator(H, n_qubits=2)
        psi_sparse = get_ground_state(H_sparse)[1]
        psi_sparse = sp.csr_matrix(psi_sparse.reshape(1, -1))
        print(f"   Ground state computed (shape: {psi_sparse.shape})")
        
        # Step 3: Define original Pauli decomposition
        print("\n3. Setting up original decomposition...")
        original_decomp = [
            QubitOperator('X0 Y1'),
            QubitOperator('Z0 X1'),
            QubitOperator('Y0 Z1')
        ]
        
        print("   Original decomposition:")
        for i, frag in enumerate(original_decomp):
            print(f"     Fragment {i}: {frag}")
        
        # Step 4: Compute original variance
        print("\n4. Computing original variance...")
        total_variance = 0
        for i, frag in enumerate(original_decomp):
            frag_op = get_sparse_operator(frag, n_qubits=2)
            var = sparse_variance(frag_op, psi_sparse)
            total_variance += np.sqrt(var)
            print(f"     Fragment {i} variance: {var:.6f}")
        
        print(f"   Total variance: {total_variance:.6f}")
        
        # Step 5: Apply Ghost Pauli method
        print("\n5. Applying Ghost Pauli method...")
        try:
            updated_decomp = update_decomp_w_ghost_paulis_sparse(
                psi_sparse, N=2, original_decomp=original_decomp
            )
            
            print("   ✓ Ghost Pauli method applied successfully!")
            
            # Step 6: Compute new variance
            print("\n6. Computing updated variance...")
            new_total_variance = 0
            for i, frag in enumerate(updated_decomp):
                frag_op = get_sparse_operator(frag, n_qubits=2)
                var = sparse_variance(frag_op, psi_sparse)
                new_total_variance += np.sqrt(var)
                print(f"     Fragment {i} variance: {var:.6f}")
            
            print(f"   New total variance: {new_total_variance:.6f}")
            
            # Step 7: Show results
            print("\n7. Results:")
            print("   Updated decomposition:")
            for i, frag in enumerate(updated_decomp):
                print(f"     Fragment {i}: {frag}")
            
            variance_reduction = total_variance - new_total_variance
            print(f"\n   Variance reduction: {variance_reduction:.6f}")
            
            if variance_reduction > 0:
                print("   🎉 Ghost Pauli method successfully reduced variance!")
            else:
                print("   ℹ️  No significant variance reduction in this example")
            
            return updated_decomp
            
        except Exception as e:
            print(f"   ❌ Error applying Ghost Pauli method: {e}")
            print("   This might be due to the specific Hamiltonian or decomposition")
            return None
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure the package is installed correctly:")
        print("pip install git+https://github.com/Rick0317/quantum_measurement_reduction.git")
        return None

def test_pauli_string_operations():
    """
    Test basic Pauli string operations.
    """
    
    print("\n" + "=" * 60)
    print("Testing Pauli String Operations")
    print("=" * 60)
    
    try:
        from quantum_measurement_reduction.entities import PauliString
        from quantum_measurement_reduction.SymplecticVectorSpace import SpaceFVector
        
        # Create Pauli strings
        pauli1 = PauliString(((0, 'X'), (1, 'Y')))
        pauli2 = PauliString(((0, 'X'), (1, 'Z')))
        
        print(f"Pauli 1: {pauli1}")
        print(f"Pauli 2: {pauli2}")
        
        # Test commutativity
        commute = pauli1.qubit_wise_commute(pauli2)
        print(f"Pauli strings commute: {commute}")
        
        # Test symplectic vector space
        vec1 = SpaceFVector(pauli1, n=2)
        vec2 = SpaceFVector(pauli2, n=2)
        
        product = vec1 * vec2
        print(f"Symplectic product: {product}")
        
        print("✓ Pauli string operations work correctly!")
        
    except Exception as e:
        print(f"❌ Pauli string test failed: {e}")

def main():
    """
    Main function to run the example.
    """
    
    # Run Ghost Pauli example
    result = simple_ghost_pauli_example()
    
    # Test Pauli string operations
    test_pauli_string_operations()
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    
    if result is not None:
        print("✓ Ghost Pauli method executed successfully")
        print("✓ You can now integrate this into your quantum computing projects")
    else:
        print("❌ Example encountered errors")
        print("Please check your installation and dependencies")
    
    print("\nFor more examples, see the INSTALLATION_GUIDE.md file")
    print("GitHub: https://github.com/Rick0317/quantum_measurement_reduction")

if __name__ == "__main__":
    main()
