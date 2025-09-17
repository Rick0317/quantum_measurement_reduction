from openfermion import FermionOperator

def construct_one_body_fermion_operator(ferm_op, obt):

    num_orbitals = obt.shape[0]  # assuming h1e is square
    num_spin_orbs = 2 * num_orbitals
    for p in range(num_orbitals):
        for q in range(num_orbitals):
            coefficient = obt[p, q]
            if coefficient != 0.0:

                term_alpha_alpha = FermionOperator((( 2 *p, 1), ( 2 *q, 0)), coefficient)
                ferm_op += term_alpha_alpha

                # Beta-beta: (2*p+1, 1) and (2*q+1, 0)
                term_beta_beta = FermionOperator((( 2 * p +1, 1), ( 2 * q +1, 0)), coefficient)
                ferm_op += term_beta_beta

    return ferm_op


def construct_two_body_fermion_operator(ferm_op, tbt):

    num_orbitals = tbt.shape[0]  # assuming h1e is square
    for p in range(num_orbitals):
        for q in range(num_orbitals):
            for r in range(num_orbitals):
                for s in range(num_orbitals):
                    coefficient = tbt[p, q, r, s]
                    if coefficient != 0.0:
                        term_1 = FermionOperator \
                            ((( 2 *p, 1), ( 2 *q, 0), ( 2 *r, 1), ( 2 *s, 0)), coefficient)
                        term_2 = FermionOperator(
                                                 (( 2 *p + 1, 1), ( 2 * q +1, 0), ( 2 *r, 1), ( 2 *s, 0)), coefficient)
                        term_3 = FermionOperator((( 2 *p, 1), ( 2 *q, 0), ( 2 *r + 1, 1), ( 2 *s + 1, 0)), coefficient)
                        term_4 = FermionOperator(
                                                 (( 2 * p +1, 1), ( 2 * q +1, 0), ( 2 *r + 1, 1), ( 2 *s + 1, 0)), coefficient)

                        ferm_op += term_1 + term_2 + term_3 + term_4

    return ferm_op

def construct_three_body_fermion_operator(ferm_op, threebt):
    num_orbitals = threebt.shape[0]  # assuming h1e is square
    for p in range(num_orbitals):
        for q in range(num_orbitals):
            for r in range(num_orbitals):
                for s in range(num_orbitals):
                    for m in range(num_orbitals):
                        for n in range(num_orbitals):
                            coefficient = threebt[p, q, r, s, m, n]
                            if coefficient != 0.0:
                                term_1 = FermionOperator(
                                                         (( 2 *p, 1), ( 2 *q, 0), ( 2 *r, 1), ( 2 *s, 0), ( 2 *m, 1), ( 2 *n, 0)), coefficient)
                                term_2 = FermionOperator((( 2 * p +1, 1), ( 2 * q +1, 0), ( 2 *r, 1), ( 2 *s, 0), ( 2 *m, 1), ( 2 *n, 0)), coefficient)
                                term_3 = FermionOperator(
                                                         (( 2 *p, 1), ( 2 *q, 0), ( 2 * r +1, 1), ( 2 * s +1, 0), ( 2 *m, 1), ( 2 *n, 0)), coefficient)
                                term_4 = FermionOperator(
                                                         (( 2 *p, 1), ( 2 *q, 0), ( 2 *r, 1), ( 2 *s, 0), ( 2 * m +1, 1), ( 2 * n +1, 0)), coefficient)
                                term_5 = FermionOperator((( 2 * p +1, 1), ( 2 * q +1, 0), ( 2 * r +1, 1), ( 2 * s +1, 0), ( 2 *m, 1), ( 2 *n, 0)), coefficient)
                                term_6 = FermionOperator((( 2 * p +1, 1), ( 2 * q +1, 0), ( 2 *r, 1), ( 2 *s, 0), ( 2 * m +1, 1), ( 2 * n +1, 0)), coefficient)
                                term_7 = FermionOperator(
                                                         (( 2 *p, 1), ( 2 *q, 0), ( 2 * r +1, 1), ( 2 * s +1, 0), ( 2 * m +1, 1), ( 2 * n +1, 0)), coefficient)
                                term_8 = FermionOperator((( 2 * p +1, 1), ( 2 * q +1, 0), ( 2 * r +1, 1), ( 2 * s +1, 0), ( 2 * m +1, 1), ( 2 * n +1, 0)), coefficient)


                                ferm_op += term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8

    return ferm_op
