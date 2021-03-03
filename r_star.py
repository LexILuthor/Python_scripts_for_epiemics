import myFunctions as myFun
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


def states(nh, initial_infected=1):
    S = nh - initial_infected
    E = 0
    I = initial_infected
    R = 0
    state_to_id = {}
    id_to_state = {}
    id_counter = 0
    while S >= 0:
        tmp_E = E
        tmp_SI = I
        while E >= 0:
            tmp_I = I
            while I >= 0:
                state_to_id[(S, E, I)] = id_counter
                id_to_state[id_counter] = (S, E, I)
                id_counter = id_counter + 1
                I = I - 1

            E = E - 1
            I = tmp_I + 1

        S = S - 1
        E = tmp_E + 1
        I = tmp_SI
    return state_to_id, id_to_state, id_counter


def get_transition_matrix(nh, beta, nu, gamma, initial_infected=1):
    states_to_id, id_to_states, number_of_states = states(nh, initial_infected)
    transition_matrix = np.zeros((number_of_states, number_of_states))

    # function in substitution to the map function
    [myFun.initialize_row_of_transition_matrix(x, transition_matrix, id_to_states, states_to_id, beta, nu,
                                               gamma, nh) for x in id_to_states]
    return transition_matrix, states_to_id, id_to_states
