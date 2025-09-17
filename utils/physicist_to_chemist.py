from openfermion import FermionOperator, normal_ordered
import numpy as np

def physicist_to_chemist(fermion_op: FermionOperator):
    """
    Transform normal ordered fermion operator into chemist format.
    :param fermion_op:
    :return:
    """
    normal_ordered_op = normal_ordered(fermion_op)
    chemist_op = FermionOperator()

    for term, coeff in normal_ordered_op.terms.items():
        if len(term) == 2 or not len(term):
            chemist_op += FermionOperator(term, coeff)

        elif len(term) == 4:
            if term[1][0] == term[2][0]:
                new_one_body_term = (term[0], term[3])
                chemist_op += FermionOperator(new_one_body_term, coeff)

            new_two_body_term = (term[0], term[2], term[1], term[3])
            chemist_op += FermionOperator(new_two_body_term, -coeff)

        elif len(term) == 6:
            print("Term", term)
            if term[2][0] == term[3][0]:
                if term[1][0] == term[5][0]:
                    new_one_body_term = (term[0], term[4])
                    chemist_op += FermionOperator(new_one_body_term, -coeff)

                new_two_body_term = (term[0], term[5], term[1], term[4])
                chemist_op += FermionOperator(new_two_body_term, coeff)

            if term[3][0] == term[1][0]:
                if term[2][0] == term[4][0]:
                    new_onw_body_term = (term[0], term[5])
                    chemist_op += FermionOperator(new_onw_body_term, -coeff)

                new_two_body_term = (term[0], term[4], term[2], term[5])
                chemist_op += FermionOperator(new_two_body_term, coeff)

            if term[2][0] == term[4][0]:
                new_two_body_term = (term[0], term[3], term[1], term[5])
                chemist_op += FermionOperator(new_two_body_term, coeff)

            new_three_body_term = (term[0], term[3], term[1], term[4], term[2], term[5])
            chemist_op += FermionOperator(new_three_body_term, -coeff)

    return chemist_op


def physicist_to_chemist_tensor(obt, tbt):
    n = obt.shape[0]
    obt_phy = obt.copy()
    for p in range(n):
        for q in range(n):
            obt_phy[p, q] -= 0.5 * sum([tbt.copy()[p, r, r, q] for r in range(n)])

    return obt_phy, 0.5 * tbt.copy()


def chemist_ferm_to_tensor(fermion_op: FermionOperator, n_mode):
    one_body_tensor = np.zeros((n_mode, n_mode))
    two_body_tensor = np.zeros((n_mode, n_mode, n_mode, n_mode))
    three_body_tensor = np.zeros((n_mode, n_mode, n_mode, n_mode, n_mode, n_mode))

    for term, coeff in fermion_op.terms.items():
        if len(term) == 2:
            p = term[0][0]
            q = term[1][0]
            spatial_p = p // 2
            spatial_q = q // 2
            one_body_tensor[spatial_p, spatial_q] = coeff

        if len(term) == 4:
            p = term[0][0]
            q = term[1][0]
            r = term[2][0]
            s = term[3][0]
            spatial_p = p // 2
            spatial_q = q // 2
            spatial_r = r // 2
            spatial_s = s // 2
            two_body_tensor[spatial_p, spatial_q, spatial_r, spatial_s] = coeff

        if len(term) == 6:
            p = term[0][0]
            q = term[1][0]
            r = term[2][0]
            s = term[3][0]
            t = term[4][0]
            u = term[5][0]
            spatial_p = p // 2
            spatial_q = q // 2
            spatial_r = r // 2
            spatial_s = s // 2
            spatial_t = t // 2
            spatial_u = u // 2
            three_body_tensor[spatial_p, spatial_q, spatial_r, spatial_s, spatial_t, spatial_u] = coeff

    return one_body_tensor, two_body_tensor, three_body_tensor
