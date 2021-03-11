import myFunctions as myFun
import analyze_data as ad
from scipy.optimize import brentq
import numpy as np


def Rstar(nh, betaG, betaH, gamma, nu):
    # Go to function p in fiile myFun.py to change the way q(k) is computed (using Pellis method or using trasition matrix)
    summation = 0
    for k in range(0, nh - 1):
        summation = summation + myFun.mu(nh, 1, nh - 1, k, betaH, gamma, nu)
    return summation * (betaG / gamma)


def R_0_Household(nh, betaG, betaH, gamma, nu):
    epsilon = 0.0001
    return brentq(myFun.g_nh, 0 + epsilon, 20, args=(nh, betaG, betaH, gamma, nu))


def R0_from_r(algorithm, tot_simulations, nu, gamma):
    r = ad.logistic_regression(algorithm, tot_simulations)[0]
    R0 = 1 + (r * (r + nu + gamma) / (nu * gamma))
    return R0


def compute_growth_rate_r(nh, betaG, betaH, nu, gamma, a, b, initial_infected=1):
    transition_matrix, states_to_id, id_to_states = myFun.get_continuous_transition_matrix(nh, betaH, nu, gamma,
                                                                                           initial_infected)
    # subtract to the diagonal of the matrix i_k (the matix delta in the article)
    for i in range(len(transition_matrix[0])):
        current_state = id_to_states[i]
        transition_matrix[i][i] = transition_matrix[i][i] - gamma * current_state[2]

    # find the solution of the laplace transform
    root = brentq(myFun.laplace_transform_infectious_profile, a, b,
                  (nh, betaG, transition_matrix, id_to_states, states_to_id, -1))
    return root


def compute_Rstar(nh, betaG, betaH, nu, gamma, initial_infected=1):
    transition_matrix, states_to_id, id_to_states = myFun.get_continuous_transition_matrix(nh, betaH, nu, gamma,
                                                                                           initial_infected)
    # subtract to the diagonal of the matrix i_k (the matix delta in the article)
    for i in range(len(transition_matrix[0])):
        current_state = id_to_states[i]
        transition_matrix[i][i] = transition_matrix[i][i] - gamma * current_state[2]

    slimmed_transition_matrix, absorbing_states = myFun.slim_transition_matrix(nh, transition_matrix, states_to_id)
    number_of_states = len(transition_matrix[0])
    Q_1 = np.linalg.inv(slimmed_transition_matrix)

    ik = 0
    Rstar = 0
    for i in range(number_of_states - nh):
        if ik in absorbing_states:
            ik = ik + 1
        Rstar = Rstar + (- Q_1[1][i]) * id_to_states[ik][2]
        ik = ik + 1
    return betaG * Rstar
