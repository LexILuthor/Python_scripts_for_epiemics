import numpy as np
from scipy.optimize import brentq

import analyze_data as ad
import myFunctions as myFun


def Rstar_following_pellis_2(nh, betaG, betaH, nu, gamma):
    # Go to function p in fiile myFun.py to change the way q(k) is computed (using Pellis method or using trasition matrix)
    summation = 0
    for k in range(0, nh):
        summation = summation + myFun.mu(nh, 1, nh - 1, k, betaH, nu, gamma)
    return summation * (betaG / gamma)


def betaG_given_r(nh, r, betaG, betaH, nu, gamma, initial_infected=1):
    QH, states_to_id, id_to_states = myFun.get_QH(nh, betaH, nu, gamma, initial_infected)

    number_of_states = len(QH[0])

    matrix = QH - (r * np.identity(number_of_states - nh))
    Q_HI = np.linalg.inv(matrix)

    result = 0
    for i in range(number_of_states - nh):
        result = result + id_to_states[i][2] * (-Q_HI[1][i])
    return 1 / result


def R_0_Household(nh, betaG, betaH, gamma, nu):
    epsilon = 0.0001
    return brentq(myFun.g_nh, 0 + epsilon, 20, args=(nh, betaG, betaH, gamma, nu))


def R0_from_r(algorithm, tot_simulations, nu, gamma):
    r = ad.logistic_regression(algorithm, tot_simulations)[0]
    R0 = 1 + (r * (r + nu + gamma) / (nu * gamma))
    return R0


def Rstar_from_r(nh, betaG, betaH, nu, gamma, a, b, initial_infected):
    r = compute_growth_rate_r(nh, betaG, betaH, nu, gamma, a, b, -1)
    QH, states_to_id, id_to_states = myFun.get_QH(nh, betaH, nu, gamma, initial_infected)
    initial_state = states_to_id[(nh - 1, 0, 1)]
    QH_1 = np.linalg.inv(QH)
    matrix = np.matmul(QH_1, np.matmul(-(np.identity(len(QH)) / r) + QH_1, QH_1))
    Rstar = 1
    for i in range(len(QH)):
        Rstar = Rstar - betaG * id_to_states[i][2] * matrix[initial_state][i]
    return Rstar


def compute_growth_rate_r(nh, betaG, betaH, nu, gamma, a, b, initial_infected=1):
    QH, states_to_id, id_to_states = myFun.get_QH(nh, betaH, nu, gamma, initial_infected)
    # find the solution of the laplace transform
    root = brentq(myFun.laplace_transform_infectious_profile, a, b,
                  args=(nh, betaG, QH, states_to_id, id_to_states, -1))
    return root


def compute_Rstar_following_pellis_markov(nh, betaG, betaH, nu, gamma, initial_infected=1):
    # ------------------------------------------------------------------------------------------------------------------
    '''
    transition_matrix, states_to_id, id_to_states = myFun.get_continuous_transition_matrix(nh, betaH, nu, gamma)
    initial_state = states_to_id[(nh - 1, 0, 1)]
    number_of_states = len(transition_matrix[0])

    Rstar = 0
    for i in range(number_of_states):
        if id_to_states[i][2] != 0:
            func = lambda t: scipy.linalg.expm(transition_matrix * t)[initial_state, i]
            Rstar = Rstar + (scipy.integrate.quad(func, 0, np.inf)[0] * id_to_states[i][2])

    return betaG * Rstar
    '''
    # ------------------------------------------------------------------------------------------------------------------

    transition_matrix, states_to_id, id_to_states = myFun.get_QH(nh, betaH, nu, gamma, initial_infected)
    initial_state = states_to_id[(nh - 1, 0, 1)]

    number_of_states = len(transition_matrix[0])

    Q_1 = np.linalg.inv(transition_matrix)

    Rstar = 0
    for i in range(number_of_states):
        Rstar = Rstar + ((- Q_1[initial_state][i]) * id_to_states[i][2])
    return betaG * Rstar
