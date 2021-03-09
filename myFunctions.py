from math import comb
import numpy as np
import r_star as r
import matplotlib.pyplot as plt
import math


def q(nh, k, betaH, gamma):
    return gamma / (gamma + k * betaH)


def p(nh, a, m, s, betaH, gamma, nu):
    summation = 0
    q_from_matrix = compute_q_using_transition_matrix(nh, betaH, gamma, nu)
    for k in range(m, s):
        # two options
        # here we use the exact same formula as Pellis to compute q(k)
        # summation = summation + pow(-1, k - m) * comb(s - m, k - m) * pow(q(nh, k, betaH, gamma), a)
        # here we use the transition matrix
        summation = summation + pow(-1, k - m) * comb(s - m, k - m) * pow(q_from_matrix[k], a)
    return comb(a, m) * summation


def mu(nh, a, s, k, betaH, gamma, nu):
    if (s == 0):
        return 0
    if (k > s):
        print("Error k>s!")
    if (k == 0):
        return a

    summation = 0
    for i in range(1, s - k + 1):
        summation = p(nh, a, s - i, s, betaH, gamma, nu) * mu(nh, i, s - i, k - 1, betaH, gamma, nu)
    return summation


def get_path_of(algorithm):
    if (algorithm.lower() == "gillespie_household"):
        return "../Gillespie_for_Households/InputOutput/gillespie_Household"
    if (algorithm.lower() == "test"):
        return "test"
    if (algorithm.lower() == "gillespie_household_lockdown"):
        return "../Gillespie_for_Households/InputOutput/gillespie_Household_lockdown"
    if (algorithm.lower() == "gillespie"):
        return "../Gillespie_algorithm/OutputFIle/gillespie"
    if (algorithm.lower() == "sellke"):
        return "../Sellke/OutputFile/sellke"
    print(
        "error, the possibie choice are: gillespie, sellke, gillespie_household, gillespie_household_lockdown, test")
    exit()


def plot_dataset(data, ax, color_susceptible, color_exposed, color_infected, color_recovered):
    S = data[0].values
    E = data[1].values
    I = data[2].values
    R = data[3].values
    time = data[4].values

    ax.plot(time, S, color=color_susceptible, linewidth=0.2, linestyle='-')
    ax.plot(time, R, color=color_recovered, linewidth=0.2, linestyle='-')
    ax.plot(time, E, color=color_exposed, linewidth=0.2, linestyle='-')
    ax.plot(time, I, color=color_infected, linewidth=0.2, linestyle='-')


def print_analysis(algorithm, tot_simulations, infected_peak, infected_peak_time, total_infected, major_outbreak):
    outputResults = open("results_" + algorithm + ".txt", "w")
    outputResults.write("the maximum number of infected at the same time is " + str(infected_peak.mean()))
    outputResults.write(" (variance " + str(math.sqrt(infected_peak.var())) + " ) ")
    outputResults.write(" and it is reached at time " + str(infected_peak_time.mean()))
    outputResults.write(" with variance " + str(math.sqrt(infected_peak_time.var())))
    outputResults.write("The total number of infected is " + str(total_infected.mean()))
    outputResults.write(" (variance " + str(math.sqrt(total_infected.var())) + " )\n")
    outputResults.write("we got a major outbreak " + str(100 * major_outbreak / tot_simulations) + "% of the times")

    outputResults.close()


def logistic_function(t, r, k, c0=1):
    return (k * c0) / (c0 + (k - c0) * np.exp(-r * t))


def print_simulation(time, cumulative_cases, ax, parameters):
    ax.plot(time, cumulative_cases, linestyle='-', color='#FF4000', linewidth=0.2)
    ax.plot(time, logistic_function(time, *parameters), linestyle='-', color='#2E64FE', linewidth=0.2)


def g_nh(x, nh, betaG, betaH, gamma, nu):
    summation = 0
    for i in range(nh):
        summation = summation + (betaG / gamma) * mu(nh, 1, nh - 1, int(i), betaH, gamma, nu) / pow(x, int(i) + 1)
    return 1 - summation


def initialize_row_of_transition_matrix(id_starting_state, transition_matrix, id_to_states, states_to_id, beta, nu,
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


def get_discrete_transition_matrix(nh, beta, nu, gamma, initial_infected=1):
    states_to_id, id_to_states, number_of_states = states(nh, initial_infected)
    transition_matrix = np.zeros((number_of_states, number_of_states))

    # function in substitution to the map function
    [initialize_row_of_transition_matrix(x, transition_matrix, id_to_states, states_to_id, beta, nu,
                                         gamma, nh) for x in id_to_states]
    return transition_matrix, states_to_id, id_to_states


def get_continuous_transition_matrix(nh, beta, nu, gamma, initial_infected=1):
    states_to_id, id_to_states, number_of_states = states(nh, initial_infected)
    transition_matrix = np.zeros((number_of_states, number_of_states))

    # function in substitution to the map function
    [initialize_row_of_transition_matrix(x, transition_matrix, id_to_states, states_to_id, beta, nu,
                                         gamma, nh) for x in id_to_states]

    for i in range(number_of_states):
        transition_matrix[i][i] = 0
        transition_matrix[i][i] = -sum(transition_matrix[i])

    return transition_matrix, states_to_id, id_to_states


def compute_q_using_transition_matrix(nh, betaH, gamma, nu):
    transition_matrix, states_to_id, id_to_states = get_discrete_transition_matrix(nh, betaH, nu, gamma, 1)
    id_target_state = np.zeros(nh, dtype=int)
    for k in range(nh):
        id_target_state[k] = int(states_to_id[(int(k), 0, 0)])

    old_distribution = np.zeros(len(transition_matrix[0, :]))
    current_distribution = np.zeros(len(transition_matrix[0, :]))

    id_initial_state = states_to_id[(nh - 1, 0, 1)]
    current_distribution[id_initial_state] = 1

    while not ((old_distribution == current_distribution).all()):
        old_distribution = current_distribution
        current_distribution = np.matmul(old_distribution, transition_matrix)

    q = np.zeros(nh)
    for k in range(nh):
        q[k] = current_distribution[id_target_state[k]]
    return q


def transfer_probability(beta, nu, gamma, from_S, from_E, from_I, to_S, to_E, to_I, nh):
    if from_S - 1 == to_S:
        return beta * from_S * from_I / nh
    if from_E - 1 == to_E:
        return nu * from_E
    if from_I - 1 == to_I:
        return gamma * from_I


def set_to_none(data):
    data = None
