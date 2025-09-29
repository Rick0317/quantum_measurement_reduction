# Test Suite

This directory contains all test files for the Quantum Measurement Reduction package, organized by functionality.

## Structure

```
tests/
├── __init__.py
├── README.md
├── bliss_tests/           # BLISS algorithm tests
│   ├── __init__.py
│   ├── test_qubit_bliss.py
│   ├── test_molecular_bliss.py
│   ├── test_molecular_bliss_two_body.py
│   ├── test_qubit_bliss_pauli_norm.py
│   ├── test_dual_killer_bliss.py
│   ├── test_hf_killer.py
│   ├── test_killer_operator.py
│   ├── simple_hf_killer_test.py
│   ├── realistic_killer_test.py
│   ├── debug_bliss_norm.py
│   └── example_usage.py
├── test_basic.py          # Basic package tests
├── ghost_pauli_example.py # Ghost Pauli method example
└── verify_installation.py # Installation verification
```

## Running Tests

### Basic Tests
```bash
python3 tests/test_basic.py
```

### BLISS Tests
```bash
# Test qubit space BLISS
python3 tests/bliss_tests/test_qubit_bliss.py

# Test molecular BLISS
python3 tests/bliss_tests/test_molecular_bliss.py

# Test dual killer BLISS
python3 tests/bliss_tests/test_dual_killer_bliss.py
```

### Installation Verification
```bash
python3 tests/verify_installation.py
```

### Examples
```bash
# Ghost Pauli example
python3 tests/ghost_pauli_example.py

# BLISS usage example
python3 tests/bliss_tests/example_usage.py
```

## Test Categories

### BLISS Algorithm Tests
- **Molecular Tests**: Test BLISS on real molecular Hamiltonians from `ham_lib/`
- **Qubit Space Tests**: Test BLISS in qubit space using Pauli operators
- **Killer Operator Tests**: Test different killer operator formulations
- **Hartree-Fock Tests**: Test BLISS with Hartree-Fock reference states
- **Dual Killer Tests**: Test BLISS with multiple killer operators

### Performance Tests
- **1-Norm Analysis**: Compare 1-norm reductions before and after BLISS
- **Optimization Tests**: Test different optimization strategies
- **Mapping Tests**: Compare Bravyi-Kitaev vs Jordan-Wigner mappings

### Debugging Tests
- **Norm Analysis**: Debug why BLISS might increase certain norms
- **Parameter Analysis**: Analyze optimized parameters
- **Term Structure**: Analyze Pauli term structure changes

## Requirements

All tests require:
- OpenFermion
- SciPy
- NumPy
- SymPy

Molecular tests also require molecular data files in `ham_lib/` directory.
