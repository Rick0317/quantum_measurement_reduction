from openfermion import MajoranaOperator
from sympy import Add, Basic, simplify
import sympy as sp
import numpy


class CustomMajoranaOperator(MajoranaOperator):

    def __iadd__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        for term, coefficient in other.terms.items():
            if term in self.terms:
                self.terms[term] += coefficient
            else:
                self.terms[term] = coefficient

        return self

    def __add__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        terms = {}
        terms.update(self.terms)

        for term, coefficient in other.terms.items():
            if term in terms:
                terms[term] += coefficient
            else:
                terms[term] = coefficient

    def __imul__(self, other):
        if not isinstance(other, (type(self), int, float, complex, Add)):
            return NotImplemented

        if isinstance(other, (int, float, complex, Add)):
            for term in self.terms:
                self.terms[term] *= sp.sympify(other)
            return self

        return self * other

    def __rmul__(self, other):
        if not isinstance(other, (int, float, complex, Add, sp.Basic)):
            return NotImplemented
        return self * other

    def __mul__(self, other):
        if not isinstance(other, (type(self), int, float, complex, Basic, Add)):
            return NotImplemented

        if isinstance(other, (int, float, complex, Basic, Add)):
            terms = {term: coefficient * other for term, coefficient in
                     self.terms.items()}
            return CustomMajoranaOperator.from_dict(terms)

        terms = {}
        for left_term, left_coefficient in self.terms.items():
            for right_term, right_coefficient in other.terms.items():
                new_term, parity = _merge_majorana_terms(left_term, right_term)
                coefficient = left_coefficient * right_coefficient * (
                    -1) ** parity
                if new_term in terms:
                    terms[new_term] += coefficient
                else:
                    terms[new_term] = coefficient

        return CustomMajoranaOperator.from_dict(terms)

    def __add__(self, other):
        if isinstance(other, (CustomMajoranaOperator, sp.Basic)):
            terms = self.terms.copy()
            if isinstance(other, CustomMajoranaOperator):
                for term, coeff in other.terms.items():
                    terms[term] = terms.get(term, 0) + coeff
            elif isinstance(other, sp.Basic):  # Handle symbolic addition.
                for term in self.terms:
                    terms[term] += sp.sympify(other)
            return CustomMajoranaOperator.from_dict(terms)
        return NotImplemented

    def __radd__(self, other):
        # Handle reverse addition (Add + MajoranaOperator).
        return self.__add__(other)

    def __str__(self):
        if not self.terms:
            return '0'
        lines = []
        for term, coeff in sorted(self.terms.items()):
            # if isinstance(coeff, Add):
            #     if simplify(coeff) == 0:
            #         continue
            # else:
            #     if numpy.isclose(float(coeff), 0.0):
            #         continue
            lines.append('{} {} +'.format(coeff, term))
        if not lines:
            return '0'
        return '\n'.join(lines)[:-2]

    @staticmethod
    def from_dict(terms):
        """Initialize a CustomMajoranaOperator from a terms dictionary."""
        op = CustomMajoranaOperator()
        op.terms = terms

        return op

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex, Add, sp.Basic)):
            terms = {term: coeff / sp.sympify(other) for term, coeff in self.terms.items()}
            return CustomMajoranaOperator.from_dict(terms)
        return NotImplemented

    def numeric_terms(self):
        """
        Returns numeric coefficients, excluding symbolic terms.
        """
        return {term: coeff for term, coeff in self.terms.items() if sp.asks_finite(coeff)}

def _merge_majorana_terms(left_term, right_term):
    """Merge two Majorana terms.

    Args:
        left_term (Tuple[int]): The left-hand term
        right_term (Tuple[int]): The right-hand term

    Returns:
        Tuple[Tuple[int], int]. The first object returned is a sorted list
        representing the indices acted upon. The second object is the parity
        of the term. A parity of 1 indicates that the term should include
        a minus sign.
    """
    merged_term = []
    parity = 0
    i, j = 0, 0
    while i < len(left_term) and j < len(right_term):
        if left_term[i] < right_term[j]:
            merged_term.append(left_term[i])
            i += 1
        elif left_term[i] > right_term[j]:
            merged_term.append(right_term[j])
            j += 1
            parity += len(left_term) - i
        else:
            parity += len(left_term) - i - 1
            i += 1
            j += 1
    if i == len(left_term):
        merged_term.extend(right_term[j:])
    else:
        merged_term.extend(left_term[i:])
    return tuple(merged_term), parity % 2
