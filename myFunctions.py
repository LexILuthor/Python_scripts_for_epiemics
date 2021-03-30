import math
from functools import cache

import numpy as np
import pandas as pd


def q(nh, k, betaH, gamma):
    return gamma / (gamma + k * betaH)


def p(a, s, m, nh, betaH, gamma, nu):
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

    p = np.zeros(nh)
    for k in range(nh):
        p[k] = current_distribution[id_target_state[k]]
    return p[m]


def mu(nh, a, s, k, betaH, gamma, nu):
    if s == 0:
        return 0
    if k > s:
        print("Error k>s!")
    if k == 0:
        return a

    summation = 0
    for i in range(1, s - k + 1):
        summation = p(a, s - i, s, nh, betaH, gamma, nu) * mu(nh, i, s - i, k - 1, betaH, gamma, nu)
    return summation


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
    print(
        "error, the possibie choice are: gillespie, sellke, gillespie_household, gillespie_household_lockdown, test")
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
        summation = summation + (betaG / gamma) * mu(nh, 1, nh - 1, int(i), betaH, gamma, nu) / pow(x, int(i) + 1)
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


def laplace_transform_infectious_profile(r, nh, betaG, QH, states_to_id, id_to_states, result=0):
    number_of_states = len(QH[0])

    matrix = QH - (r * np.identity(number_of_states))
    Q_HI = np.linalg.inv(matrix)
    for i in range(number_of_states):
        result = result + betaG * id_to_states[i][2] * (-Q_HI[1][i])
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
                                                       beta, nu,
                                                       gamma, nh):
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


@cache
def get_discrete_transition_matrix(nh, beta, nu, gamma, initial_infected=1):
    states_to_id, id_to_states, number_of_states = states(nh, initial_infected)
    transition_matrix = np.zeros((number_of_states, number_of_states))

    # function in substitution to the map function
    [initialize_row_of_transition_matrix(x, transition_matrix, states_to_id, id_to_states, beta, nu,
                                         gamma, nh) for x in id_to_states]
    return transition_matrix, states_to_id, id_to_states


def get_continuous_transition_matrix(nh, beta, nu, gamma):
    states_to_id, id_to_states, number_of_states = states(nh, 1)
    transition_matrix = np.zeros((number_of_states, number_of_states))

    # function in substitution to the map function
    [initialize_row_of_transition_matrix_not_normalized(x, transition_matrix, states_to_id, id_to_states, beta, nu,
                                                        gamma, nh) for x in id_to_states]

    # The following is to put on the diagonal the -(sum) of the row
    for i in range(number_of_states):
        transition_matrix[i][i] = 0
        transition_matrix[i][i] = -sum(transition_matrix[i])

    return transition_matrix, states_to_id, id_to_states


def get_QH(nh, beta, nu, gamma, initial_infected=1):
    transition_matrix, states_to_id, id_to_states = get_continuous_transition_matrix(nh, beta, nu, gamma)

    slimmed_transition_matrix, slim_states_to_id, slim_id_to_states = slim_transition_matrix(nh, transition_matrix,
                                                                                             states_to_id,
                                                                                             id_to_states)
    return slimmed_transition_matrix, slim_states_to_id, slim_id_to_states


def transfer_probability(beta, nu, gamma, from_S, from_E, from_I, to_S, to_E, to_I, nh):
    if from_S - 1 == to_S:
        return beta * from_S * from_I / nh
    if from_E - 1 == to_E:
        return nu * from_E
    if from_I - 1 == to_I:
        return gamma * from_I


def set_to_none(data):
    data = None


def slim_transition_matrix(nh, transition_matrix, states_to_id, id_to_states):
    number_of_states = len(transition_matrix[0])
    slimmed_transition_matrix = np.zeros((number_of_states - nh, number_of_states - nh))
    absorbing_states = [None]
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
