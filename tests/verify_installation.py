#!/usr/bin/env python3
"""
Verification script for Quantum Measurement Reduction Package installation.
Run this script to verify that the package is correctly installed and working.
"""


def test_imports():
    """Test that all main modules can be imported."""
    print("Testing package imports...")

    try:
        # Test main package import
        import quantum_measurement_reduction

        print("✓ Main package imported successfully")

        # Test Ghost Pauli module
        from quantum_measurement_reduction.ghost_pauli import (
            sparse_expectation,
            sparse_variance,
            update_decomp_w_ghost_paulis_sparse,
        )

        print("✓ Ghost Pauli module imported successfully")

        # Test entities module
        from quantum_measurement_reduction.entities import PauliOp, PauliString

        print("✓ Entities module imported successfully")

        # Test SymplecticVectorSpace module
        from quantum_measurement_reduction.SymplecticVectorSpace import (
            SpaceFVector,
            vector_2_pauli,
        )

        print("✓ SymplecticVectorSpace module imported successfully")

        # Test shared_pauli module
        from quantum_measurement_reduction.shared_pauli import (
            apply_shared_pauli,
            optimize_coeffs,
        )

        print("✓ Shared Pauli module imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_dependencies():
    """Test that required dependencies are available."""
    print("\nTesting dependencies...")

    dependencies = [
        ("numpy", "np"),
        ("scipy", "sp"),
        ("openfermion", None),
        ("qiskit", None),
        ("cirq", None),
        ("matplotlib", None),
    ]

    all_available = True

    for dep_name, alias in dependencies:
        try:
            if alias:
                exec(f"import {dep_name} as {alias}")
            else:
                exec(f"import {dep_name}")
            print(f"✓ {dep_name} is available")
        except ImportError:
            print(f"❌ {dep_name} is not available")
            all_available = False

    return all_available


def test_basic_functionality():
    """Test basic functionality of the package."""
    print("\nTesting basic functionality...")

    try:
        from quantum_measurement_reduction.entities import PauliString

        # Test PauliString creation
        pauli = PauliString(((0, "X"), (1, "Y"), (2, "Z")))
        print("✓ PauliString creation works")

        # Test commutativity check
        pauli2 = PauliString(((0, "X"), (1, "Z"), (2, "Y")))
        commute = pauli.qubit_wise_commute(pauli2)
        print(f"✓ Commutativity check works: {commute}")

        # Test SymplecticVectorSpace
        from quantum_measurement_reduction.SymplecticVectorSpace import SpaceFVector

        vec = SpaceFVector(pauli, n=3)
        print("✓ SpaceFVector creation works")

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_ghost_pauli_availability():
    """Test that Ghost Pauli functions are available."""
    print("\nTesting Ghost Pauli function availability...")

    try:
        from quantum_measurement_reduction.ghost_pauli import (
            select_combs_sparse,
            select_paulis_sparse,
            sparse_expectation,
            sparse_variance,
            update_decomp_w_ghost_paulis_sparse,
        )

        print("✓ All Ghost Pauli functions are available:")
        print("  - update_decomp_w_ghost_paulis_sparse")
        print("  - sparse_variance")
        print("  - sparse_expectation")
        print("  - select_paulis_sparse")
        print("  - select_combs_sparse")

        return True

    except ImportError as e:
        print(f"❌ Ghost Pauli functions not available: {e}")
        return False


def main():
    """Main verification function."""
    print("=" * 60)
    print("Quantum Measurement Reduction Package - Installation Verification")
    print("=" * 60)

    # Run all tests
    import_success = test_imports()
    deps_success = test_dependencies()
    func_success = test_basic_functionality()
    ghost_success = test_ghost_pauli_availability()

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    if import_success and deps_success and func_success and ghost_success:
        print("🎉 ALL TESTS PASSED!")
        print("✓ Package is correctly installed")
        print("✓ All dependencies are available")
        print("✓ Basic functionality works")
        print("✓ Ghost Pauli functions are ready to use")
        print("\nYou can now use the Ghost Pauli method in your projects!")
        print("\nExample usage:")
        print(
            "from quantum_measurement_reduction.ghost_pauli import update_decomp_w_ghost_paulis_sparse"
        )

    else:
        print("❌ SOME TESTS FAILED!")
        print("\nPlease check the following:")
        if not import_success:
            print(
                "- Package installation (try: pip install git+https://github.com/Rick0317/quantum_measurement_reduction.git)"
            )
        if not deps_success:
            print(
                "- Dependencies installation (try: pip install numpy scipy openfermion qiskit cirq-core matplotlib)"
            )
        if not func_success:
            print("- Basic functionality (check error messages above)")
        if not ghost_success:
            print("- Ghost Pauli functions (check import errors above)")

    print("\n" + "=" * 60)
    print("For help, visit: https://github.com/Rick0317/quantum_measurement_reduction")
    print("=" * 60)


if __name__ == "__main__":
    main()
