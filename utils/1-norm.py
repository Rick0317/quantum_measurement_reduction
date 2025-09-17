import numpy as np
import h5py
from openfermion import FermionOperator, get_majorana_operator, normal_ordered
from basic_utils import *


def one_body_1norm(h1e, g2e, t3e):
    n = h1e.shape[0]
    one_norm = 0
    for i in range(n):
        for j in range(n):
            term = h1e[i, j]
            term += 2 * sum(g2e[i, j, k, k] for k in range(n))
            term += 3 * sum(t3e[i, j, k, k, m, m] for k in range(n) for m in range(n))
            term += 1/4 * (t3e[i, j, i, j, i, j])
            for k in range(n):
                for l in range(n):
                    term += 2 * t3e[k, l, k, l, i, j]
                    term -= t3e[k, l, k, j, i, l]
                    term += t3e[k, j, k, l, i, l]
                    term -= t3e[k, l, i, l, k, j]
                    term += 3 * t3e[k, l, i, j, k, l]
                    term -= t3e[k, j, i, l, k, l]
                    term += t3e[i, l, k, l, k, j]
                    term -= t3e[i, l, k, j, k, l]
                    term += t3e[i, j, k, l, k, l]
            one_norm += abs(term)
    return one_norm


def two_body_1norm_1(g2e, t3e):
    n = g2e.shape[0]
    one_norm = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    one_norm += abs(0.5 * g2e[i, j, k, l] +
                                    1.5 * sum(t3e[i, j, k, l, m, m]
                                              for m in range(n)))

    return one_norm

def two_body_1norm_2(g2e, t3e):
    n = g2e.shape[0]
    one_norm = 0
    for i in range(n):
        for j in range(n):
            for k in range(i):
                for l in range(j):
                    one_norm += abs(g2e[i, j, k, l] - g2e[i, l, k, j] + 3 * sum(t3e[i, j, k, l, m, m] - t3e[i, l, k, j, m, m] for m in range(n)))

    return one_norm

def two_body_1norm_3(t3e):
    mode = t3e.shape[0]
    one_norm = 0
    for j in range(mode):
        for l in range(j):
            for m in range(mode):
                for n in range(mode):
                    one_norm += abs(1/4 * sum(t3e[i, j, m, n, i, l] +
                                              t3e[i, l, m, n, i, j] +
                                              t3e[i, j, i, l, m, n] -
                                              t3e[i, l, m, n, i, j] -
                                              t3e[i, j, m, n, i, l] -
                                              t3e[i, l, i, j, m, n]
                                              for i in range(mode)))

    return one_norm

def two_body_1norm_4(t3e):
    mode = t3e.shape[0]
    one_norm = 0
    for i in range(mode):
        for k in range(i):
            for m in range(mode):
                for n in range(mode):
                    one_norm += abs(1/4 * sum(t3e[i, j, m, n, k, j] +
                                              t3e[k, j, m, n, i, j] +
                                              t3e[i, j, k, j, m, n] -
                                              t3e[k, j, m, n, i, j] -
                                              t3e[i, j, m, n, k, j] -
                                              t3e[k, j, i, j, m, n]
                                              for j in range(mode)))

    return one_norm


def two_body_1norm_5(t3e):
    mode = t3e.shape[0]
    one_norm = 0
    for j in range(mode):
        for l in range(j):
            for n in range(l):
                for m in range(mode):
                    term = (t3e[m, j, m, l, m, n] -
                            t3e[m, j, m, n, m, l] -
                            t3e[m, l, m, j, m, n] +
                            t3e[m, l, m, n, m, j] +
                            t3e[m, n, m, j, m, l] -
                            t3e[m, n, m, l, m, j])

                    term += sum(t3e[i, j, i, l, m, n] -
                                t3e[i, j, i, n, m, l] -
                                t3e[i, l, i, n, m, l] +
                                t3e[i, l, i, n, m, j] +
                                t3e[i, n, i, j, m, l] -
                                t3e[i, n, i, l, m, j] -
                                t3e[i, j, m, l, i, n] +
                                t3e[i, j, m, n, i, l] +
                                t3e[i, l, m, j, i, n] -
                                t3e[i, l, m, n, i, j] -
                                t3e[i, n, m, j, i, l] +
                                t3e[i, n, m, l, i, j] +
                                t3e[m, j, i, l, i, n] -
                                t3e[m, j, i, n, i, l] -
                                t3e[m, l, i, j, i, n] +
                                t3e[m, l, i, n, i, j] +
                                t3e[m, n, i, j, i, l] -
                                t3e[m, n, i, l, i, j]
                                for i in range(mode))

                    one_norm += abs(term)

    return one_norm


def two_body_1norm_6(t3e):
    mode = t3e.shape[0]
    one_norm = 0
    for i in range(mode):
        for k in range(i):
            for m in range(k):
                for n in range(mode):
                    term = (t3e[i, n, k, n, m, n] -
                            t3e[i, n, m, n, k, n] -
                            t3e[k, n, i, n, m, n] +
                            t3e[k, n, m, n, i, n] +
                            t3e[m, n, i, n, k, n] -
                            t3e[m, n, k, n, i, n])

                    term += sum(t3e[i, j, k, j, m, n] -
                                t3e[i, j, m, j, k, n] -
                                t3e[k, j, i, j, m, n] +
                                t3e[k, j, i, j, m, n] +
                                t3e[m, j, i, j, k, n] -
                                t3e[m, j, k, j, i, n] -
                                t3e[i, j, k, n, m, j] +
                                t3e[i, j, m, n, k, j] +
                                t3e[k, j, i, n, m, j] -
                                t3e[k, j, m, n, i, j] -
                                t3e[m, j, i, n, k, j] +
                                t3e[m, j, k, n, i, j] +
                                t3e[i, n, k, n, m, n] -
                                t3e[i, n, m, n, k, n] -
                                t3e[k, n, i, j, m, j] +
                                t3e[k, n, m, k, i, j] +
                                t3e[m, n, i, j, k, j] -
                                t3e[m, n, k, j, i, j]

                                for j in range(mode))

                    one_norm += abs(term)

    return one_norm

def three_body_1norm_1(t3e):
    mode = t3e.shape[0]
    one_norm = 0
    for i in range(mode):
        for k in range(i):
            for j in range(mode):
                for l in range(j):
                    for m in range(mode):
                        for n in range(mode):
                            term = (2 * t3e[i, j, m, n, k, l] +
                                    2 * t3e[k, l, m, n, i, j] +
                                    t3e[i, j, k, l, m, n] +
                                    t3e[k, l, i, j, m, n] -
                                    2 * t3e[k, j, m, n, i, l] -
                                    2 * t3e[i, l, m, n, k, j] -
                                    t3e[k, j, i, l, m, n] -
                                    t3e[i, l, k, j, m, n])

                            one_norm += abs(term)

    return one_norm

def three_body_1norm_2(t3e):
    mode = t3e.shape[0]
    one_norm = 0
    for i in range(mode):
        for j in range(mode):
            for k in range(i):
                for l in range(j):
                    for m in range(k):
                        for n in range(l):
                            term = 1/4 * (t3e[i, j, k, l, m, n] -
                                          t3e[i, j, k, n, m, l] -
                                          t3e[i, l, k, j, m, n] +
                                          t3e[i, l, k, n, m, j] +
                                          t3e[i, n, k, j, m, l] -
                                          t3e[i, n, k, l, m, j])

                            term += 1/4 * (-t3e[i, j, m, l, k, n] +
                                           t3e[i, j, m, n, k, l] +
                                           t3e[i, l, m, j, k, n] -
                                           t3e[i, l, m, n, k, j] -
                                           t3e[i, n, m, j, k, l] +
                                           t3e[i, n, m, l, k, j])

                            term += 1/4 * (-t3e[k, j, i, l, m, n] +
                                           t3e[k, j, i, n, m, l] +
                                           t3e[k, l, i, j, m, n] -
                                           t3e[k, l, i, n, m , j] -
                                           t3e[k, n, i, j, m, l] +
                                           t3e[k, n, i, l, m, j])

                            term += 1/4 * (t3e[k, j, m, l, i, n] -
                                           t3e[k, j, m, n, i, l] -
                                           t3e[k, l, m, j, i, n] +
                                           t3e[k, l, m, n, i, j] +
                                           t3e[k, n, m, j, i, l] -
                                           t3e[k, n, m, l, i, j])

                            term += 1/4 * (t3e[m, j, i, l, k, n] -
                                           t3e[m, j, i, n, k, l] -
                                           t3e[m, l, i, j, k, n] +
                                           t3e[m , l, i, n, k, j] +
                                           t3e[m, n, i, j, k, l] -
                                           t3e[m , n, i, l, k, j])

                            term += 1/4 * (- t3e[m, j, k, l, i, n] +
                                           t3e[m, j, k, n, i, l] +
                                           t3e[m, l, k, j, i, n] -
                                           t3e[m, l, k, n, i, j] -
                                           t3e[m, n, k, j, i, l] +
                                           t3e[m, n, k, l, i, j])

                            one_norm += abs(term)

    return one_norm


if __name__ == '__main__':
    with h5py.File("h2_commutator_tensors.h5", "r") as fid:

        # Access the "BLISS_HAM" group
        mol_data_group = fid["BLISS_HAM"]

        # Read multi-dimensional data
        h_const = mol_data_group["h_const"]
        h1e = mol_data_group["obt"][:]
        g2e = mol_data_group["tbt"][:]
        t3e = mol_data_group["threebt"][:]
        eta = mol_data_group["eta"][()]

    t3e = np.zeros((2, 2, 2, 2, 2, 2))
    print(g2e)

    one_body_1norm_est = one_body_1norm(h1e, g2e, t3e)
    two_body_1norm_est1 = two_body_1norm_1(g2e, t3e)
    two_body_1norm_est2 = two_body_1norm_2(g2e, t3e)
    two_body_1norm_est3 = two_body_1norm_3(t3e)
    two_body_1norm_est4 = two_body_1norm_4(t3e)
    two_body_1norm_est5 = two_body_1norm_5(t3e)
    two_body_1norm_est6 = two_body_1norm_6(t3e)
    three_body_1norm_est1 = three_body_1norm_1(t3e)
    three_body_1norm_est2 = three_body_1norm_2(t3e)

    estimated_1norm = (one_body_1norm_est + two_body_1norm_est1 +
                       two_body_1norm_est2 + two_body_1norm_est3 +
                       two_body_1norm_est4 + two_body_1norm_est5 +
                       two_body_1norm_est6 + three_body_1norm_est1 +
                       three_body_1norm_est2)
    fermion_operator = FermionOperator()
    fermion_operator = construct_one_body_fermion_operator(fermion_operator,
                                                           h1e)
    fermion_operator = construct_two_body_fermion_operator(fermion_operator,
                                                           g2e)
    fermion_operator = construct_three_body_fermion_operator(fermion_operator,
                                                             t3e)
    majo = get_majorana_operator(fermion_operator)
    one_norm_corr = 0
    two_majo_norm = 0
    four_majo_norm = 0
    four_majo_norm1 = 0
    four_majo_norm2 = 0
    allowed_four_2 = [(0, 1, 2, 3), (2, 3, 4, 5), (0, 1, 6, 7), (0, 2, 5, 7), (1, 2, 4, 7), (0, 3, 5, 6), (1, 3, 4, 6), (4, 5, 6, 7)]
    # print(majo)
    for term, coeff in majo.terms.items():
        if term != ():
            one_norm_corr += abs(coeff)
            if coeff != 0:
                even_indices = sum(1 for i in range(len(term)) if i % 2 == 0)
                odd_indices = sum(1 for i in range(len(term)) if i % 2 != 0)
        if len(term) == 2:
            two_majo_norm += abs(coeff)

        if len(term) == 4:
            if term == (0, 1, 4, 5) or term == (2, 3, 6, 7):
                four_majo_norm2 += abs(coeff)

            elif term in allowed_four_2:
                four_majo_norm1 += abs(coeff)
            elif coeff != 0:
                print(term)
                print(coeff)

    #one_norm_corr += two_majo_norm + four_majo_norm1 + four_majo_norm2

    print(two_majo_norm)
    print(f"2 Majo Norm Estimated: {one_body_1norm_est}, Correct: {two_majo_norm}")
    print(f"4 Majo Norm (1) Estimated: {two_body_1norm_est1 }, Correct: {four_majo_norm1}")
    print(f"4 Majo Norm (2) Estimated: {two_body_1norm_est2}, Correct: {four_majo_norm2}")
    print(f"4 Majo Norm Estimated: {two_body_1norm_est1 + two_body_1norm_est2}, Correct: {four_majo_norm1 + four_majo_norm2}")
    print(f"1-Norm Estimated: {estimated_1norm}, Correct: {one_norm_corr}")
