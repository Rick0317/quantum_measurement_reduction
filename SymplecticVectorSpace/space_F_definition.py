from entities.paulis import PauliString
pauli_2_binary = {'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1), 'I': (0, 0)}
binary_2_pauli = {(0, 0): 'I', (0, 1): 'Z', (1, 0): 'X', (1, 1): 'Y'}


def _pauli_2_vector(pauli_string: PauliString, n: int):
    """
    Given a Pauli String, this function returns the binary field representation of it in the vector space
    :param pauli_string:
    :return: The list of 0 nad 1s corresponding to the binary field representation of the Pauli String
    """
    first_list = []
    second_list = []
    pauli_ops = pauli_string.get_pauli_ops()
    non_ident_indices = pauli_ops.keys()
    for i in range(n):
        if i not in non_ident_indices:
            first_list.append(0)
            second_list.append(0)
        else:
            pauli = pauli_ops[i]
            binary_pair = pauli_2_binary[pauli]
            first_list.append(binary_pair[0])
            second_list.append(binary_pair[1])

    return first_list + second_list


def vector_2_pauli(vector, n: int):
    """
    Given a vector in F space,
    this function returns the corresponding Pauli String
    :param vector: The binary vector
    :param n:
    :return:
    """
    pauli_list = []
    for i in range(n):
        index_1 = vector[i]
        index_2 = vector[i + n]
        pauli_list.append((i, binary_2_pauli[(index_1, index_2)]))

    result = tuple(pauli_list)

    return PauliString(pauli_ops=result)


class SpaceFVector:
    """
    The Symplectic vector in F over the binary field.

    """
    def __init__(self, pauli_string: PauliString, n):
        self._dim = 2 * n
        self._vector = _pauli_2_vector(pauli_string, n)

    def get_vector(self):
        return self._vector

    def set_vector(self, vector):
        self._vector = vector

    def apply_J(self):
        first_vector = self._vector[:self._dim // 2]
        second_vector = self._vector[self._dim // 2:]
        self.set_vector(second_vector + first_vector)

    def __mul__(self, other):
        self_vector = self.get_vector()
        other_vector = other.get_vector()
        result = 0
        for i in range(self._dim):
            result += (self_vector[i] * other_vector[i])

        return result % 2


if __name__ == '__main__':
    p1 = PauliString(((0, 'X'), (3, 'Z')))
    p2 = PauliString(((0, 'X'), (1, 'Z'), (2, 'Z'), (3, 'Y')))
    dim = 5
    vec_1 = SpaceFVector(p1, dim)
    vec_1_copy = SpaceFVector(p1, dim)
    vec_1_copy.apply_J()
    print(vec_1.get_vector())
    vec_2 = SpaceFVector(p2, dim)
    vec_2.apply_J()
    print(vec_2.get_vector())

    print(vec_1 * vec_1_copy)
