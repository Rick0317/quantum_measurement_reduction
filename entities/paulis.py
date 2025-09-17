from openfermion import QubitOperator

class PauliOp:
    def __init__(self, op):
        self.index, self.pauli = op

    def commutes(self, other):
        if self.index != other.index:
            return True  # Different qubits always commute.
        # Commutation logic: same Pauli or identity always commutes.
        if self.pauli == other.pauli or self.pauli == 'I' or other.pauli == 'I':
            return True
        # Anti-commuting pairs: X/Y, Y/Z, Z/X.
        return False  # X vs Y, Y vs Z, Z vs X anti-commute.


class PauliString:
    def __init__(self, pauli_ops: tuple):
        self.pauli_ops = {index: pauli for index, pauli in pauli_ops}

    def qubit_wise_commute(self, other):
        for index in set(self.pauli_ops.keys()).intersection(other.pauli_ops.keys()):
            if not PauliOp((index, self.pauli_ops[index])).commutes(PauliOp((index, other.pauli_ops[index]))):
                return False
        return True

    def fully_commute(self, other):
        if not isinstance(other, PauliString):
            raise TypeError("Can only compare with PauliString objects.")
        qubit_op1 = self.to_qubit_operator()
        qubit_op2 = other.to_qubit_operator()
        commutator = qubit_op1 * qubit_op2 - qubit_op2 * qubit_op1
        return len(commutator.terms) == 0

    def to_qubit_operator(self, coeff=1.0):
        qubit_terms = tuple(
            (index, pauli) for index, pauli in self.pauli_ops.items())
        return QubitOperator(qubit_terms, coeff)

    def get_pauli_ops(self):
        return self.pauli_ops

    def __add__(self, other):
        if not isinstance(other, PauliString):
            raise TypeError("Can only add PauliString objects.")

        # Combine the two QubitOperators.
        result_operator = self.to_qubit_operator() + other.to_qubit_operator()
        return result_operator

    def __str__(self):
        return str(self.pauli_ops)

    def __eq__(self, other):
        if not isinstance(other, PauliString):
            return False
        if len(self.pauli_ops) != len(other.pauli_ops):
            return False
        for index, pauli in self.pauli_ops.items():
            if pauli != other.pauli_ops[index]:
                return False

        return True



def pauli_ops_to_qop(pauli_ops: dict):
    pauli_string = ''
    for key in pauli_ops:
        pauli = pauli_ops[key]
        if pauli != 'I':
            pauli_string += pauli + str(key) + ' '

    return QubitOperator(pauli_string)


if __name__ == '__main__':
    p1 = PauliString(((0, 'X'), (4, 'Z')))
    p2 = PauliString(((0, 'X'), (1, 'Z'), (2, 'Z'), (3, 'Y')))


