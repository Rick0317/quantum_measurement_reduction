# Installation and Usage Guide for Quantum Measurement Reduction Package

This guide will help you install the Quantum Measurement Reduction package from GitHub and use the Ghost Pauli functionality in your Python projects.

## Prerequisites

Before installing, make sure you have:
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation Methods

### Method 1: Direct Installation from GitHub (Recommended)

Install the package directly from GitHub without cloning:

```bash
pip install git+https://github.com/Rick0317/quantum_measurement_reduction.git
```

### Method 2: Clone and Install Locally

1. Clone the repository:
```bash
git clone https://github.com/Rick0317/quantum_measurement_reduction.git
cd quantum_measurement_reduction
```

2. Install the package in development mode:
```bash
pip install -e .
```

### Method 3: Install with Development Dependencies

If you want to contribute to the project or run tests:

```bash
git clone https://github.com/Rick0317/quantum_measurement_reduction.git
cd quantum_measurement_reduction
pip install -e ".[dev]"
```

## Installing Dependencies

The package requires several dependencies. Install them using:

```bash
pip install numpy scipy openfermion qiskit cirq-core matplotlib
```

Or install from the requirements file:
```bash
pip install -r requirements.txt
```

## Using Ghost Pauli in Your Python Project

### Basic Import

```python
from quantum_measurement_reduction.ghost_pauli import update_decomp_w_ghost_paulis_sparse
from quantum_measurement_reduction.ghost_pauli import sparse_variance, sparse_expectation
```

### Complete Example: Ghost Pauli Method

Here's a complete example showing how to use the Ghost Pauli method:

```python
import numpy as np
import scipy.sparse as sp
from openfermion import QubitOperator, get_sparse_operator, get_ground_state
from quantum_measurement_reduction.ghost_pauli import update_decomp_w_ghost_paulis_sparse

def ghost_pauli_example():
    """
    Example demonstrating the Ghost Pauli method for quantum measurement reduction.
    """
    
    # Step 1: Create a quantum Hamiltonian
    print("Creating quantum Hamiltonian...")
    H = QubitOperator('X0 Y1 Z2') + QubitOperator('Z0 X1 Y2') + QubitOperator('Y0 Z1 X2')
    print(f"Hamiltonian: {H}")
    
    # Step 2: Get the ground state as a sparse matrix
    print("Computing ground state...")
    H_sparse = get_sparse_operator(H, n_qubits=3)
    psi_sparse = get_ground_state(H_sparse)[1]
    psi_sparse = sp.csr_matrix(psi_sparse.reshape(1, -1))
    print(f"Ground state shape: {psi_sparse.shape}")
    
    # Step 3: Define original Pauli decomposition
    print("Setting up original decomposition...")
    original_decomp = [
        QubitOperator('X0 Y1'),
        QubitOperator('Z0 X1'),
        QubitOperator('Y0 Z1')
    ]
    
    print("Original decomposition:")
    for i, frag in enumerate(original_decomp):
        print(f"  Fragment {i}: {frag}")
    
    # Step 4: Apply Ghost Pauli method
    print("\nApplying Ghost Pauli method...")
    try:
        updated_decomp = update_decomp_w_ghost_paulis_sparse(
            psi_sparse, N=3, original_decomp=original_decomp
        )
        
        print("Updated decomposition with Ghost Paulis:")
        for i, frag in enumerate(updated_decomp):
            print(f"  Fragment {i}: {frag}")
            
        return updated_decomp
        
    except Exception as e:
        print(f"Error applying Ghost Pauli method: {e}")
        return None

# Run the example
if __name__ == "__main__":
    result = ghost_pauli_example()
```

### Advanced Usage: Variance Computation

```python
from quantum_measurement_reduction.ghost_pauli import sparse_variance, sparse_expectation
from openfermion import get_sparse_operator

def compute_variance_reduction():
    """
    Example showing how to compute variance reduction using sparse matrices.
    """
    
    # Create operators
    H = QubitOperator('X0 Y1') + QubitOperator('Z0 X1')
    H_sparse = get_sparse_operator(H, n_qubits=2)
    
    # Get ground state
    psi_sparse = get_ground_state(H_sparse)[1]
    psi_sparse = sp.csr_matrix(psi_sparse.reshape(1, -1))
    
    # Compute variance and expectation
    variance = sparse_variance(H_sparse, psi_sparse)
    expectation = sparse_expectation(H_sparse, psi_sparse)
    
    print(f"Variance: {variance}")
    print(f"Expectation: {expectation}")
    
    return variance, expectation
```

### Working with Pauli Strings

```python
from quantum_measurement_reduction.entities import PauliString, PauliOp
from quantum_measurement_reduction.SymplecticVectorSpace import SpaceFVector

def pauli_string_example():
    """
    Example showing how to work with Pauli strings and symplectic vector spaces.
    """
    
    # Create Pauli strings
    pauli1 = PauliString(((0, 'X'), (1, 'Y'), (2, 'Z')))
    pauli2 = PauliString(((0, 'X'), (1, 'Z'), (2, 'Y')))
    
    print(f"Pauli 1: {pauli1}")
    print(f"Pauli 2: {pauli2}")
    
    # Check commutativity
    commute = pauli1.qubit_wise_commute(pauli2)
    print(f"Pauli strings commute: {commute}")
    
    # Work with symplectic vector space
    vec1 = SpaceFVector(pauli1, n=3)
    vec2 = SpaceFVector(pauli2, n=3)
    
    # Compute symplectic product
    product = vec1 * vec2
    print(f"Symplectic product: {product}")
    
    return pauli1, pauli2, commute, product
```

## Integration in Your Project

### Project Structure Example

```
your_project/
├── main.py
├── requirements.txt
├── README.md
└── examples/
    ├── ghost_pauli_example.py
    └── variance_computation.py
```

### requirements.txt for Your Project

```
git+https://github.com/Rick0317/quantum_measurement_reduction.git
numpy>=2.2.6
scipy>=1.15.3
openfermion>=1.7.1
qiskit>=2.1.2
cirq-core>=1.5.0
matplotlib>=3.10.6
```

### main.py Example

```python
"""
Main script demonstrating Ghost Pauli usage in your project.
"""

from quantum_measurement_reduction.ghost_pauli import update_decomp_w_ghost_paulis_sparse
from quantum_measurement_reduction.entities import PauliString
import numpy as np
import scipy.sparse as sp
from openfermion import QubitOperator, get_sparse_operator, get_ground_state

def main():
    """
    Main function demonstrating Ghost Pauli integration.
    """
    
    # Your quantum system setup
    n_qubits = 4
    H = QubitOperator('X0 Y1 Z2') + QubitOperator('Z0 X1 Y2') + QubitOperator('Y0 Z1 X2')
    
    # Get quantum state
    H_sparse = get_sparse_operator(H, n_qubits=n_qubits)
    psi_sparse = get_ground_state(H_sparse)[1]
    psi_sparse = sp.csr_matrix(psi_sparse.reshape(1, -1))
    
    # Your original decomposition
    original_decomp = [
        QubitOperator('X0 Y1'),
        QubitOperator('Z0 X1'),
        QubitOperator('Y0 Z1'),
        QubitOperator('X2 Y3')
    ]
    
    # Apply Ghost Pauli method
    print("Applying Ghost Pauli method...")
    updated_decomp = update_decomp_w_ghost_paulis_sparse(
        psi_sparse, N=n_qubits, original_decomp=original_decomp
    )
    
    # Use the results in your project
    print("Ghost Pauli optimization completed!")
    print(f"Number of fragments: {len(updated_decomp)}")
    
    return updated_decomp

if __name__ == "__main__":
    result = main()
```

## Troubleshooting

### Common Issues and Solutions

1. **ImportError: No module named 'openfermion'**
   ```bash
   pip install openfermion
   ```

2. **ImportError: No module named 'quantum_measurement_reduction'**
   - Make sure you installed the package correctly
   - Check if you're in the right Python environment
   - Try: `pip list | grep quantum`

3. **Git installation issues**
   ```bash
   # Make sure git is installed
   git --version
   
   # If not installed, install git first
   # On Ubuntu/Debian: sudo apt install git
   # On macOS: brew install git
   # On Windows: Download from https://git-scm.com/
   ```

4. **Permission errors during installation**
   ```bash
   # Use user installation
   pip install --user git+https://github.com/Rick0317/quantum_measurement_reduction.git
   
   # Or use virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install git+https://github.com/Rick0317/quantum_measurement_reduction.git
   ```

### Verification

Test your installation:

```python
# Test script
try:
    from quantum_measurement_reduction.ghost_pauli import update_decomp_w_ghost_paulis_sparse
    print("✓ Ghost Pauli module imported successfully!")
    
    from quantum_measurement_reduction.entities import PauliString
    print("✓ Entities module imported successfully!")
    
    from quantum_measurement_reduction.SymplecticVectorSpace import SpaceFVector
    print("✓ SymplecticVectorSpace module imported successfully!")
    
    print("\n🎉 Package installation successful!")
    print("You can now use Ghost Pauli methods in your project.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please check your installation.")
```

## Getting Help

- **GitHub Issues**: [https://github.com/Rick0317/quantum_measurement_reduction/issues](https://github.com/Rick0317/quantum_measurement_reduction/issues)
- **Documentation**: Check the README.md in the repository
- **Examples**: Look at the code examples in this guide

## Next Steps

1. Install the package using one of the methods above
2. Run the verification script
3. Try the basic example
4. Integrate Ghost Pauli into your quantum computing project
5. Explore other modules (shared_pauli, entities, etc.)

Happy coding with quantum measurement reduction! 🚀

