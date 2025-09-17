from openfermion import MolecularData, FermionOperator
from openfermionpyscf import run_pyscf, generate_molecular_hamiltonian
import pickle


def generate_molecule_data(geometry, file_name, basis='sto3g', multiplicity=1, charge=0):

    # Get FermionOperator for the molecular Hamiltonian
    fermion_hamiltonian = generate_molecular_hamiltonian(geometry, basis,
                                                         multiplicity, charge)

    # Save to .bin file using pickle
    with open(f'{file_name}', 'wb') as f:
        pickle.dump(fermion_hamiltonian, f)


if __name__ == '__main__':
    r = 1
    file_name = "beh2_fer.bin"
    geometry = [
        ('H', (0, 0, 0)),
        ('Be', (0, 0, r)),
        ('H', (0, 0, 2 * r)),
    ]
    generate_molecule_data(geometry, file_name)
