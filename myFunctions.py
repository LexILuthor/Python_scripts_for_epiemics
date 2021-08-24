import math
from functools import cache

import numpy as np
import pandas as pd


def q(nh, k, betaH, gamma):
    return gamma / (gamma + k * betaH)


def p_pellis(a, m, s, nh, betaH, gamma, nu):
    summation = 0

    for k in range(m, s):
        # two options
        # here we use the exact same formula as Pellis to compute q(k)
        summation = summation + pow(-1, k - m) * math.comb(s - m, k - m) * pow(q(nh, k, betaH, gamma), a)
        # here we use the transition matrix
        # summation = summation + pow(-1, k - m) * comb(s - m, k - m) * pow(q_from_matrix[k], a)
    return math.comb(s, m) * summation


def p(a, m, s, nh, betaH, gamma, nu):
    transition_matrix, states_to_id, id_to_states = get_discrete_transition_matrix(nh, betaH, nu, gamma, 1)
    id_target_state = np.zeros(nh, dtype=int)
    for m in range(s + 1):
        id_target_state[m] = int(states_to_id[(m, 0, 0)])

    old_distribution = np.zeros(len(transition_matrix[0, :]))
    current_distribution = np.zeros(len(transition_matrix[0, :]))

    id_initial_state = states_to_id[(s, a, 0)]
    current_distribution[id_initial_state] = 1

    while not ((old_distribution == current_distribution).all()):
        old_distribution = current_distribution
        current_distribution = np.matmul(old_distribution, transition_matrix)

    p = np.zeros(s + 1)
    for k in range(s + 1):
        p[k] = current_distribution[id_target_state[k]]
    return p[m]


def mu(nh, a, s, k, betaH, nu, gamma):
    if s == 0:
        return 0
    if k > s:
        print("Error k>s!")
    if k == 0:
        return a

    summation = 0
    summation_pellis = 0
    for i in range(1, s - k + 2):
        summation = summation + p(a, s - i, s, nh, betaH, gamma, nu) * mu(nh, i, s - i, k - 1, betaH, nu, gamma)
        summation_pellis = summation_pellis + p_pellis(a, s - i, s, nh, betaH, gamma, nu) * mu(nh, int(i), s - int(i),
                                                                                               k - 1, betaH, nu,
                                                                                               gamma)
    return summation_pellis


def get_path_of(algorithm):
    if algorithm.lower() == "gillespie_household_lockdown_new_beta":
        return "../Gillespie_for_Households/InputOutput/gillespie_Household_new_beta__lockdown"
    if algorithm.lower() == "gillespie_household":
        return "../Gillespie_for_Households/InputOutput/gillespie_Household"
    if algorithm.lower() == "test":
        return "test"
    if algorithm.lower() == "gillespie_household_lockdown":
        return "../Gillespie_for_Households/InputOutput/gillespie_Household_lockdown"
    if algorithm.lower() == "gillespie":
        return "../Gillespie_algorithm/OutputFIle/gillespie"
    if algorithm.lower() == "sellke":
        return "../Sellke/OutputFile/sellke"
    if algorithm.lower() == "gillespie_lockdown_gamma":
        return "../Gillespie_for_Households_gamma_distribution/Output/gillespie_Household_lockdown"
    if algorithm.lower() == "gillespie_gamma":
        return "../Gillespie_for_Households_gamma_distribution/Output/gillespie_Household"
    print(
        "error, the possibie choice are: gillespie, sellke, gillespie_household, gillespie_household_lockdown, test,gillespie_gamma,gillespie_lockdown_gamma")
    exit()


def plot_dataset(data, ax, color_susceptible, color_exposed, color_infected, color_recovered, line_width=0.2,
                 log_scale=False):
    S = data[0].values
    E = data[1].values
    I = data[2].values
    R = data[3].values
    time = data[4].values

    if not log_scale:
        ax.plot(time, S, color=color_susceptible, linewidth=line_width, linestyle='-')
        ax.plot(time, R, color=color_recovered, linewidth=line_width, linestyle='-')
    ax.plot(time, E, color=color_exposed, linewidth=line_width, linestyle='-')
    ax.plot(time, I, color=color_infected, linewidth=line_width, linestyle='-')


def print_analysis(algorithm, tot_simulations, infected_peak, infected_peak_time, total_infected, major_outbreak,
                   S_infinity_time):
    outputResults = open("results_" + algorithm + ".txt", "w")
    outputResults.write("the maximum number of infected at the same time is " + str(infected_peak.mean()))
    outputResults.write(" (variance " + str(math.sqrt(infected_peak.var())) + " ) ")
    outputResults.write(" and it is reached at time " + str(infected_peak_time.mean()))
    outputResults.write(" with variance " + str(math.sqrt(infected_peak_time.var())))
    outputResults.write("The total number of infected is " + str(total_infected.mean()))
    outputResults.write(" (variance " + str(math.sqrt(total_infected.var())) + " )\n")
    outputResults.write("we got a major outbreak " + str(100 * major_outbreak / tot_simulations) + "% of the times\n")
    outputResults.write("in mean, S infinity is reached at time: " + str(S_infinity_time.mean()))
    outputResults.write(" (variance: " + str(math.sqrt(S_infinity_time.var())) + " )\n")

    outputResults.close()


def logistic_function(t, r, k, c0=1):
    return (k * c0) / (c0 + (k - c0) * np.exp(-r * t))


def print_estimation(time, real_data, estimation, ax):
    ax.plot(time, real_data, linestyle='-', color='#FF4000', linewidth=0.2)
    ax.plot(time, estimation, linestyle='-', color='#2E64FE', linewidth=0.2)


def g_nh(x, nh, betaG, betaH, gamma, nu):
    summation = 0
    for i in range(nh):
        summation = summation + (betaG / gamma) * mu(nh, 1, nh - 1, int(i), betaH, nu, gamma) / pow(x, int(i) + 1)
    return 1 - summation


def read_lockdown_times(path, iteration=0):
    path = path + str(iteration) + "lock_down_time" + ".txt"
    with open(path) as f:
        content = f.readlines()

    number_of_lockdowns = int(list(map(int, content[0].strip().split()))[0])

    lock_down_start_end = np.zeros((number_of_lockdowns, 2))

    for i in range(0, number_of_lockdowns):
        start, end, = list(map(str, content[int(i) + 1].strip().split()))
        start = float(start)
        end = float(end)
        lock_down_start_end[i][0] = start
        lock_down_start_end[i][1] = end

    return lock_down_start_end


def read_dataset(path):
    data = pd.read_csv(filepath_or_buffer=path, header=None)
    S = data[0].values
    E = data[1].values
    I = data[2].values
    R = data[3].values
    time = data[4].values
    return S, E, I, R, time


def get_data_during_lockdown(path, lockdown_times, lockdown_number):
    S, E, I, R, time = read_dataset(path)
    start_index = np.argmax(time > lockdown_times[lockdown_number][0])
    end_index = np.argmax(time > lockdown_times[lockdown_number][1])
    return S[start_index:end_index], E[start_index:end_index], I[start_index:end_index], R[start_index:end_index], time[
                                                                                                                   start_index:end_index]


# read the dataset


def laplace_transform_infectious_profile(r, nh, betaG, QH, states_to_id, id_to_states, result=-1):
    number_of_states = len(QH[0])
    initial_state = states_to_id[(nh - 1, 1, 0)]
    matrix = QH - (r * np.identity(number_of_states))
    Q_HI = np.linalg.inv(matrix)
    for i in range(number_of_states):
        result = result + (betaG * id_to_states[i][2] * (-Q_HI[initial_state][i]))
    return result


def initialize_row_of_transition_matrix(id_starting_state, transition_matrix, states_to_id, id_to_states, beta, nu,
                                        gamma, nh):
    S, E, I = id_to_states[id_starting_state]
    new_exposed_probability = transfer_probability(beta, nu, gamma, S, E, I, S - 1, E + 1, I, nh)
    new_infected_probability = transfer_probability(beta, nu, gamma, S, E, I, S, E - 1, I + 1, nh)
    new_recovered_probability = transfer_probability(beta, nu, gamma, S, E, I, S, E, I - 1, nh)
    normalization_constant = new_exposed_probability + new_infected_probability + new_recovered_probability
    if normalization_constant == 0:
        transition_matrix[id_starting_state][id_starting_state] = 1
    else:
        new_exposed_probability = new_exposed_probability / normalization_constant
        new_infected_probability = new_infected_probability / normalization_constant
        new_recovered_probability = new_recovered_probability / normalization_constant

        if new_exposed_probability > 0:
            arrival_state = states_to_id[(S - 1, E + 1, I)]
            transition_matrix[id_starting_state][arrival_state] = new_exposed_probability

        if new_infected_probability > 0:
            arrival_state = states_to_id[(S, E - 1, I + 1)]
            transition_matrix[id_starting_state][arrival_state] = new_infected_probability

        if new_recovered_probability > 0:
            arrival_state = states_to_id[(S, E, I - 1)]
            transition_matrix[id_starting_state][arrival_state] = new_recovered_probability


def initialize_row_of_transition_matrix_not_normalized(id_starting_state, transition_matrix, states_to_id, id_to_states,
                                                       beta, nu, gamma, nh):
    S, E, I = id_to_states[id_starting_state]
    new_exposed_probability = transfer_probability(beta, nu, gamma, S, E, I, S - 1, E + 1, I, nh)
    new_infected_probability = transfer_probability(beta, nu, gamma, S, E, I, S, E - 1, I + 1, nh)
    new_recovered_probability = transfer_probability(beta, nu, gamma, S, E, I, S, E, I - 1, nh)

    if new_exposed_probability > 0:
        arrival_state = states_to_id[(S - 1, E + 1, I)]
        transition_matrix[id_starting_state][arrival_state] = new_exposed_probability

    if new_infected_probability > 0:
        arrival_state = states_to_id[(S, E - 1, I + 1)]
        transition_matrix[id_starting_state][arrival_state] = new_infected_probability

    if new_recovered_probability > 0:
        arrival_state = states_to_id[(S, E, I - 1)]
        transition_matrix[id_starting_state][arrival_state] = new_recovered_probability


def states(nh, initial_infected=1):
    S = nh - initial_infected
    E = initial_infected
    I = 0
    R = 0
    state_to_id = {}
    id_to_state = {}
    id_counter = 0

    state_to_id[(S, E, I)] = id_counter
    id_to_state[id_counter] = (S, E, I)
    id_counter = id_counter + 1

    s = nh
    while s >= 0:
        e = nh
        while e >= 0:
            i = nh
            while i >= 0:
                if (s + e < nh) and (((s, e, i) in state_to_id) is False and s + e + i <= nh):
                    state_to_id[(s, e, i)] = id_counter
                    id_to_state[id_counter] = (s, e, i)
                    id_counter = id_counter + 1
                i = i - 1
            e = e - 1
        s = s - 1

    return state_to_id, id_to_state, id_counter


@cache
def get_discrete_transition_matrix(nh, beta, nu, gamma, initial_infected=1):
    states_to_id, id_to_states, number_of_states = states(nh, initial_infected)
    transition_matrix = np.zeros((number_of_states, number_of_states))

    # function in substitution to the map function
    [initialize_row_of_transition_matrix(x, transition_matrix, states_to_id, id_to_states, beta, nu,
                                         gamma, nh) for x in id_to_states]
    return transition_matrix, states_to_id, id_to_states


def get_continuous_transition_matrix(nh, beta, nu, gamma, initial_infected=1):
    states_to_id, id_to_states, number_of_states = states(nh, initial_infected)
    transition_matrix = np.zeros((number_of_states, number_of_states))

    # function in substitution to the map function
    [initialize_row_of_transition_matrix_not_normalized(x, transition_matrix, states_to_id, id_to_states, beta, nu,
                                                        gamma, nh) for x in id_to_states]

    # The following is to put on the diagonal the -(sum) of the row
    for i in range(number_of_states):
        transition_matrix[i][i] = 0
        transition_matrix[i][i] = -sum(transition_matrix[i])
        # transition_matrix[i][i] = transition_matrix[i][i] - gamma*id_to_states[i][2]

    return transition_matrix, states_to_id, id_to_states


def get_QH(nh, beta, nu, gamma, initial_infected=1):
    transition_matrix, states_to_id, id_to_states = get_continuous_transition_matrix(nh, beta, nu, gamma)

    slimmed_transition_matrix, slim_states_to_id, slim_id_to_states = slim_transition_matrix(nh, transition_matrix,
                                                                                             states_to_id, id_to_states)
    return slimmed_transition_matrix, slim_states_to_id, slim_id_to_states


def transfer_probability(beta, nu, gamma, from_S, from_E, from_I, to_S, to_E, to_I, nh):
    if from_S - 1 == to_S:
        return beta * from_S * from_I / (nh)
    if from_E - 1 == to_E:
        return nu * from_E
    if from_I - 1 == to_I:
        return gamma * from_I


def set_to_none(data):
    data = None


def slim_transition_matrix(nh, transition_matrix, states_to_id, id_to_states):
    number_of_states = len(transition_matrix[0])
    slimmed_transition_matrix = np.zeros((number_of_states - nh, number_of_states - nh))
    absorbing_states = []
    slim_id_to_states = {}
    slim_states_to_id = {}
    for i in range(nh):
        absorbing_states.append(states_to_id[(int(i), 0, 0)])

    row_s = 0
    column_s = 0
    row_o = 0
    column_o = 0

    while row_s < (number_of_states - nh):
        while row_o in absorbing_states:
            row_o = row_o + 1

        state = id_to_states[row_o]
        slim_states_to_id[state] = row_s
        slim_id_to_states[row_s] = state

        column_s = 0
        column_o = 0

        while column_s < (number_of_states - nh):
            while column_o in absorbing_states:
                column_o = column_o + 1
            slimmed_transition_matrix[row_s][column_s] = transition_matrix[row_o][column_o]
            column_s = column_s + 1
            column_o = column_o + 1
        row_s = row_s + 1
        row_o = row_o + 1

    return slimmed_transition_matrix, slim_states_to_id, slim_id_to_states


def old_inverse_Qr_SEIR_3_old(r, nh, betaG, betaH, nu, gamma):
    lambd = betaH / (nh)
    result = -1

    Q_0 = 1 / (2 * lambd + gamma + r)
    result = result + Q_0 * betaG

    Q_2 = ((2 * lambd) / (2 * lambd + gamma + r)) * (1 / (lambd + gamma + nu + r))
    result = result + Q_2 * betaG

    Q_3 = ((2 * lambd) / (2 * lambd + gamma + r)) * (gamma / (lambd + gamma + nu + r)) * (1 / (nu + r))
    result = result + Q_3 * betaG * 0

    Q_4 = ((2 * lambd) / (2 * lambd + gamma + r)) * (nu / (lambd + gamma + nu + r)) * (1 / (2 * lambd + 2 * gamma + r))
    result = result + Q_4 * betaG * 2

    Q_5 = (((2 * lambd) / (2 * lambd + gamma + r)) * (nu / (lambd + gamma + nu + r)) * (
            (2 * gamma) / (2 * lambd + 2 * gamma + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                   gamma / (lambd + gamma + nu + r)) * (nu / (nu + r))) * (1 / (lambd + gamma + r))
    result = result + Q_5 * betaG

    Q_7 = ((2 * lambd) / (2 * lambd + gamma + r)) * (lambd / (lambd + gamma + nu + r)) * (1 / (gamma + 2 * nu + r))
    result = result + Q_7 * betaG

    Q_8 = ((2 * lambd) / (2 * lambd + gamma + r)) * (lambd / (lambd + gamma + nu + r)) * (
            gamma / (gamma + 2 * nu + r)) * (1 / (2 * nu + r))
    result = result + Q_8 * betaG * 0

    Q_9 = (((2 * lambd) / (2 * lambd + gamma + r)) * (lambd / (lambd + gamma + nu + r)) * (
            (2 * nu) / (gamma + 2 * nu + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                   nu / (lambd + gamma + nu + r)) * ((2 * lambd) / (2 * lambd + 2 * gamma + r))) * (
                  1 / (2 * gamma + nu + r))
    result = result + Q_9 * betaG * 2

    Q_10 = ((((2 * lambd) / (2 * lambd + gamma + r)) * (lambd / (lambd + gamma + nu + r)) * (
            (2 * nu) / (gamma + 2 * nu + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                     nu / (lambd + gamma + nu + r)) * ((2 * lambd) / (2 * lambd + 2 * gamma + r))) * (
                    (2 * gamma) / (2 * gamma + nu + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                    lambd / (lambd + gamma + nu + r)) * (
                    gamma / (gamma + 2 * nu + r)) * ((2 * nu) / (2 * nu + r)) + (
                    ((2 * lambd) / (2 * lambd + gamma + r)) * (nu / (lambd + gamma + nu + r)) * (
                    (2 * gamma) / (2 * lambd + 2 * gamma + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                            gamma / (lambd + gamma + nu + r)) * (nu / (nu + r))) * (
                    (lambd) / (lambd + gamma + r))) * (1 / (gamma + nu + r))
    result = result + Q_10 * betaG

    Q_11 = ((((2 * lambd) / (2 * lambd + gamma + r)) * (lambd / (lambd + gamma + nu + r)) * (
            (2 * nu) / (gamma + 2 * nu + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                     nu / (lambd + gamma + nu + r)) * ((2 * lambd) / (2 * lambd + 2 * gamma + r))) * (
                    (2 * gamma) / (2 * gamma + nu + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                    lambd / (lambd + gamma + nu + r)) * (
                    gamma / (gamma + 2 * nu + r)) * ((2 * nu) / (2 * nu + r)) + (
                    ((2 * lambd) / (2 * lambd + gamma + r)) * (nu / (lambd + gamma + nu + r)) * (
                    (2 * gamma) / (2 * lambd + 2 * gamma + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                            gamma / (lambd + gamma + nu + r)) * (nu / (nu + r))) * (
                    (lambd) / (lambd + gamma + r))) * ((gamma) / (gamma + nu + r)) * (1 / (nu + r))
    result = result + Q_11 * betaG * 0

    Q_12 = (((2 * lambd) / (2 * lambd + gamma + r)) * (lambd / (lambd + gamma + nu + r)) * (
            (2 * nu) / (gamma + 2 * nu + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                    nu / (lambd + gamma + nu + r)) * ((2 * lambd) / (2 * lambd + 2 * gamma + r))) * (
                   (nu) / (2 * gamma + nu + r)) * (1 / 3 * gamma + r)
    result = result + Q_12 * betaG * 3

    Q_13 = ((((2 * lambd) / (2 * lambd + gamma + r)) * (lambd / (lambd + gamma + nu + r)) * (
            (2 * nu) / (gamma + 2 * nu + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                     nu / (lambd + gamma + nu + r)) * ((2 * lambd) / (2 * lambd + 2 * gamma + r))) * (
                    (nu) / (2 * gamma + nu + r)) * ((3 * gamma) / 3 * gamma + r) + (
                    (((2 * lambd) / (2 * lambd + gamma + r)) * (lambd / (lambd + gamma + nu + r)) * (
                            (2 * nu) / (gamma + 2 * nu + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                             nu / (lambd + gamma + nu + r)) * ((2 * lambd) / (2 * lambd + 2 * gamma + r))) * (
                            (2 * gamma) / (2 * gamma + nu + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                            lambd / (lambd + gamma + nu + r)) * (
                            gamma / (gamma + 2 * nu + r)) * ((2 * nu) / (2 * nu + r)) + (
                            ((2 * lambd) / (2 * lambd + gamma + r)) * (nu / (lambd + gamma + nu + r)) * (
                            (2 * gamma) / (2 * lambd + 2 * gamma + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                                    gamma / (lambd + gamma + nu + r)) * (nu / (nu + r))) * (
                            (lambd) / (lambd + gamma + r))) * ((nu) / (gamma + nu + r))) * (1 / (2 * gamma + r))
    result = result + Q_13 * betaG * 2

    Q_14 = (((((2 * lambd) / (2 * lambd + gamma + r)) * (lambd / (lambd + gamma + nu + r)) * (
            (2 * nu) / (gamma + 2 * nu + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                      nu / (lambd + gamma + nu + r)) * ((2 * lambd) / (2 * lambd + 2 * gamma + r))) * (
                     (nu) / (2 * gamma + nu + r)) * ((3 * gamma) / 3 * gamma + r) + (
                     (((2 * lambd) / (2 * lambd + gamma + r)) * (lambd / (lambd + gamma + nu + r)) * (
                             (2 * nu) / (gamma + 2 * nu + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                              nu / (lambd + gamma + nu + r)) * ((2 * lambd) / (2 * lambd + 2 * gamma + r))) * (
                             (2 * gamma) / (2 * gamma + nu + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                             lambd / (lambd + gamma + nu + r)) * (
                             gamma / (gamma + 2 * nu + r)) * ((2 * nu) / (2 * nu + r)) + (
                             ((2 * lambd) / (2 * lambd + gamma + r)) * (nu / (lambd + gamma + nu + r)) * (
                             (2 * gamma) / (2 * lambd + 2 * gamma + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                                     gamma / (lambd + gamma + nu + r)) * (nu / (nu + r))) * (
                             (lambd) / (lambd + gamma + r))) * ((nu) / (gamma + nu + r))) * (
                    (2 * gamma) / (2 * gamma + r)) + (
                    (((2 * lambd) / (2 * lambd + gamma + r)) * (lambd / (lambd + gamma + nu + r)) * (
                            (2 * nu) / (gamma + 2 * nu + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                             nu / (lambd + gamma + nu + r)) * ((2 * lambd) / (2 * lambd + 2 * gamma + r))) * (
                            (2 * gamma) / (2 * gamma + nu + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                            lambd / (lambd + gamma + nu + r)) * (
                            gamma / (gamma + 2 * nu + r)) * ((2 * nu) / (2 * nu + r)) + (
                            ((2 * lambd) / (2 * lambd + gamma + r)) * (nu / (lambd + gamma + nu + r)) * (
                            (2 * gamma) / (2 * lambd + 2 * gamma + r)) + ((2 * lambd) / (2 * lambd + gamma + r)) * (
                                    gamma / (lambd + gamma + nu + r)) * (nu / (nu + r))) * (
                            (lambd) / (lambd + gamma + r))) * ((gamma) / (gamma + nu + r)) * ((nu) / (nu + r))) * (
                   1 / (gamma + r))
    result = result + Q_14 * betaG

    return result


def inverse_Qr_SEIR_3(r, nh, betaG, betaH, nu, gamma):
    lambd = betaH / (nh)
    result = -1

    Q_0 = 1 / (2 * lambd + gamma + r) * (nu / (nu + r))
    result = result + Q_0 * betaG

    Q_2 = Q_0 * 2 * lambd / (gamma + nu + lambd + r)
    result = result + Q_2 * betaG

    Q_3 = Q_2 * (gamma / (nu + r))
    result = result + Q_3 * betaG * 0

    Q_4 = Q_2 * nu / (2 * lambd + 2 * gamma + r)
    result = result + Q_4 * betaG * 2

    Q_5 = (Q_3 * nu + Q_4 * 2 * gamma) / (lambd + gamma + r)
    result = result + Q_5 * betaG

    Q_7 = Q_2 * lambd / (gamma + 2 * nu + r)
    result = result + Q_7 * betaG

    Q_8 = Q_7 * gamma / (2 * nu + r)
    result = result + Q_8 * betaG * 0

    Q_9 = (Q_4 * 2 * lambd + Q_7 * 2 * nu) / (2 * gamma + nu + r)
    result = result + Q_9 * betaG * 2

    Q_10 = (Q_5 * lambd + Q_9 * 2 * gamma + Q_8 * 2 * nu) / (gamma + nu + r)
    result = result + Q_10 * betaG

    Q_11 = Q_10 * gamma / (nu + r)
    result = result + Q_11 * betaG * 0

    Q_12 = Q_9 * nu / (3 * gamma + r)
    result = result + Q_12 * betaG * 3

    Q_13 = (Q_10 * nu + Q_12 * 3 * gamma) / (2 * gamma + r)
    result = result + Q_13 * betaG * 2

    Q_14 = (Q_11 * nu + Q_13 * 2 * gamma) / (gamma + r)
    result = result + Q_14 * betaG

    return result


def inverse_Qr_SIR(r, nh, betaG, betaH, gamma):
    # return the value of the cell 1,target_state of the matrix -(Q-r)^-1
    lambd = betaH / (nh)

    result = 1 / (2 * lambd + gamma + r)
    target_state1 = betaG * result

    result = (2 * lambd / (2 * lambd + gamma + r)) * (1 / (2 * lambd + 2 * gamma + r))
    target_state3 = 2 * betaG * result

    result = (2 * lambd / (2 * lambd + gamma + r)) * (2 * gamma / (2 * lambd + 2 * gamma + r)) * (
            1 / (lambd + gamma + r))
    target_state4 = betaG * result

    result = (2 * lambd / (2 * lambd + gamma + r)) * (2 * lambd / (2 * lambd + 2 * gamma + r)) * (1 / (3 * gamma + r))
    target_state6 = 3 * betaG * result

    result = (2 * lambd / (2 * lambd + gamma + r)) * (2 * lambd / (2 * lambd + 2 * gamma + r)) * (
            3 * gamma / (3 * gamma + r)) + (
                     (2 * lambd / (2 * lambd + gamma + r)) * (2 * gamma / (2 * lambd + 2 * gamma + r)) * (
                     lambd / (lambd + gamma + r))) * (1 / (2 * gamma + r))
    target_state7 = 2 * betaG * result

    result = (2 * lambd / (2 * lambd + gamma + r)) * (2 * lambd / (2 * lambd + 2 * gamma + r)) * (
            3 * gamma / (3 * gamma + r)) + (
                     (2 * lambd / (2 * lambd + gamma + r)) * (2 * gamma / (2 * lambd + 2 * gamma + r)) * (
                     lambd / (lambd + gamma + r))) * (2 * gamma / (2 * gamma + r)) * (1 / (gamma + r))

    target_state8 = betaG * result

    return target_state1 + target_state3 + target_state4 + target_state6 + target_state7 + target_state8 - 1


def read_region_nr(codice_regione):
    real_data = pd.read_csv('dpc-covid19-ita-regioni.csv')
    region_data = real_data.loc[real_data['codice_regione'] == codice_regione]
    return region_data['totale_positivi'].values


def stable_sigmoid(x):
    sig = np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))
    return sig
