# Quantum Measurement Reduction Package

A comprehensive Python package for implementing various quantum measurement reduction techniques including Ghost Pauli, Shared Pauli, Virial relations, and other methods to reduce measurement variance in quantum algorithms.

## Features

- **Ghost Pauli Techniques**: Implement variance reduction through ghost Pauli operators
- **Shared Pauli Methods**: Optimize measurements by sharing Pauli operators between commuting sets
- **Virial Relations**: Utilize virial relations for measurement cost reduction
- **Symplectic Vector Spaces**: Work with symplectic vector spaces over binary fields
- **Sparse Matrix Support**: Efficient sparse matrix implementations for large-scale problems
- **Core Quantum Utilities**: Essential data structures and utilities for quantum computing

## Installation

### Quick Install (Recommended)

```bash
pip install git+https://github.com/Rick0317/quantum_measurement_reduction.git
```

### From Source

```bash
git clone https://github.com/Rick0317/quantum_measurement_reduction.git
cd quantum_measurement_reduction
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/Rick0317/quantum_measurement_reduction.git
cd quantum_measurement_reduction
pip install -e ".[dev]"
```

### Verify Installation

After installation, run the verification script:

```bash
python verify_installation.py
```

### Quick Example

Run the Ghost Pauli example:

```bash
python ghost_pauli_example.py
```

## Quick Start

### Ghost Pauli Method

```python
import numpy as np
import scipy.sparse as sp
from openfermion import QubitOperator, get_sparse_operator, get_ground_state
from quantum_measurement_reduction.ghost_pauli import update_decomp_w_ghost_paulis_sparse

# Create a sample Hamiltonian
H = QubitOperator('X0 Y1 Z2') + QubitOperator('Z0 X1 Y2') + QubitOperator('Y0 Z1 X2')

# Get ground state as sparse matrix
H_sparse = get_sparse_operator(H, n_qubits=3)
psi_sparse = get_ground_state(H_sparse)[1]
psi_sparse = sp.csr_matrix(psi_sparse.reshape(1, -1))

# Original decomposition (example)
original_decomp = [QubitOperator('X0 Y1'), QubitOperator('Z0 X1'), QubitOperator('Y0 Z1')]

# Apply ghost Pauli method
updated_decomp = update_decomp_w_ghost_paulis_sparse(psi_sparse, N=3, original_decomp=original_decomp)

print("Updated decomposition with ghost Paulis:")
for i, frag in enumerate(updated_decomp):
    print(f"Fragment {i}: {frag}")
```

### Shared Pauli Method

```python
from quantum_measurement_reduction.shared_pauli import apply_shared_pauli, optimize_coeffs

# Apply shared Pauli method
result = apply_shared_pauli(H_q=H, decomposition=original_decomp, N=3, Ne=2, state=psi_sparse)

# Optimize coefficients
optimized_coeffs = optimize_coeffs(
    pw_grp_idxes_fix=[],
    pw_grp_idxes_no_fix_len=[],
    meas_alloc={},
    shara_pauli_only_decomp=[],
    shara_pauli_only_no_fixed_decomp=[],
    pw_indices_dict={},
    psi=psi_sparse,
    n_qubits=3,
    original_decomp=original_decomp,
    alpha=0.1
)
```

### Working with Pauli Strings

```python
from quantum_measurement_reduction.entities import PauliString, PauliOp
from quantum_measurement_reduction.SymplecticVectorSpace import SpaceFVector, vector_2_pauli

# Create Pauli strings
pauli1 = PauliString(((0, 'X'), (1, 'Y'), (2, 'Z')))
pauli2 = PauliString(((0, 'X'), (1, 'Z'), (2, 'Y')))

# Check commutativity
print(f"Pauli strings commute: {pauli1.qubit_wise_commute(pauli2)}")

# Work with symplectic vector space
vec1 = SpaceFVector(pauli1, n=3)
vec2 = SpaceFVector(pauli2, n=3)

# Compute symplectic product
product = vec1 * vec2
print(f"Symplectic product: {product}")
```

## Package Structure

```
quantum_measurement_reduction/
├── ghost_pauli/           # Ghost Pauli techniques
│   ├── utils_ghost_pauli.py
│   └── utils_ghost_pauli_sparse.py
├── shared_pauli/          # Shared Pauli methods
│   ├── shared_paulis.py
│   ├── coefficient_optimizer.py
│   ├── shared_pauli_package.py
│   └── utils_shared_pauli_sparse.py
├── entities/              # Core data structures
│   └── paulis.py
├── SymplecticVectorSpace/ # Symplectic vector space utilities
│   └── space_F_definition.py
├── utils/                 # General utilities
│   ├── basic_utils.py
│   ├── ferm_utils.py
│   ├── frag_utils.py
│   ├── math_utils.py
│   └── ...
├── tapering/              # Tapering techniques
├── virial/                # Virial relations
└── bliss/                 # Bliss techniques
```

## Key Functions

### Ghost Pauli Module

- `update_decomp_w_ghost_paulis_sparse()`: Main function for applying ghost Pauli method with sparse matrices
- `sparse_variance()`: Compute variance using sparse matrices
- `sparse_expectation()`: Compute expectation values using sparse matrices
- `select_paulis_sparse()`: Select optimal Pauli operators for ghost Pauli method

### Shared Pauli Module

- `apply_shared_pauli()`: Apply shared Pauli method to decomposition
- `optimize_coeffs()`: Optimize coefficients for shared Pauli method
- `get_sharable_paulis()`: Identify sharable Pauli operators

### Core Utilities

- `PauliString`: Core class for representing Pauli strings
- `SpaceFVector`: Symplectic vector space operations
- `vector_2_pauli()`: Convert vectors to Pauli strings

## Dependencies

The package requires the following key dependencies:

- `numpy>=2.2.6`
- `scipy>=1.15.3`
- `openfermion>=1.7.1`
- `qiskit>=2.1.2`
- `cirq-core>=1.5.0`
- `matplotlib>=3.10.6`

See `requirements.txt` for the complete list of dependencies.

## Contributing

We welcome contributions! Please see our contributing guidelines for details on how to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{quantum_measurement_reduction,
  title={Quantum Measurement Reduction Package},
  author={Quantum Measurement Reduction Team},
  year={2024},
  url={https://github.com/yourusername/quantum_measurement_reduction}
}
```

## Getting Started

### Installation Files

- **`INSTALLATION_GUIDE.md`**: Comprehensive installation and usage guide
- **`verify_installation.py`**: Script to verify your installation
- **`ghost_pauli_example.py`**: Complete example showing Ghost Pauli usage

### Quick Start Steps

1. **Install the package:**
   ```bash
   pip install git+https://github.com/Rick0317/quantum_measurement_reduction.git
   ```

2. **Verify installation:**
   ```bash
   python verify_installation.py
   ```

3. **Run example:**
   ```bash
   python ghost_pauli_example.py
   ```

4. **Use in your project:**
   ```python
   from quantum_measurement_reduction.ghost_pauli import update_decomp_w_ghost_paulis_sparse
   ```

## Support

For questions, bug reports, or feature requests, please:

1. Check the [documentation](https://quantum-measurement-reduction.readthedocs.io/)
2. Search existing [issues](https://github.com/Rick0317/quantum_measurement_reduction/issues)
3. Create a new issue if needed

## Changelog

### Version 0.1.0
- Initial release
- Ghost Pauli implementation with sparse matrix support
- Shared Pauli methods
- Core quantum data structures
- Symplectic vector space utilities
