import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

import myFunctions as myFun
import r_star as r


def analyze_my_data(algorithm, tot_simulations):
    path = myFun.get_path_of(algorithm)
    infected_peak = []
    infected_peak_time = []
    total_infected = []
    S_infinity_time = []
    major_outbreak = 0
    for i in range(0, tot_simulations):
        data = pd.read_csv(filepath_or_buffer=path + str(i) + ".csv", header=None)
        S = data[0].values
        E = data[1].values
        I = data[2].values
        R = data[3].values
        time = data[4].values
        if S[-1] < S[0] / 2:
            index = np.argmax(I)
            infected_peak.append(I[index])
            infected_peak_time.append(time[index])
            total_infected.append(R[-1])
            S_infinity_time.append(time[-1])
            major_outbreak = major_outbreak + 1

    myFun.print_analysis(algorithm, tot_simulations, np.array(infected_peak), np.array(infected_peak_time),
                         np.array(total_infected), major_outbreak, np.array(S_infinity_time))


def exponential_regression(algorithm, tot_simulations):
    time_interval = (20, 40)

    path = myFun.get_path_of(algorithm)
    # plot style
    # plt.xkcd()
    plt.style.use("ggplot")
    plt.tight_layout()
    fig, ax = plt.subplots()
    ax.set_yscale('log')

    parameters = np.zeros(1)

    data = pd.read_csv(filepath_or_buffer=path + str(0) + ".csv", header=None)
    I = data[2].values
    R = data[3].values
    time = data[4].values
    indexes = np.argwhere(time_interval[0] < time)
    start_index = indexes.min()
    indexes = np.argwhere(time < time_interval[1])
    end_index = indexes.max()

    reg = LinearRegression()
    x = time[start_index:end_index + 1].reshape(-1, 1)
    reg.fit(x, np.log(I[start_index:end_index + 1]))
    y_pred = reg.predict(x)
    parameters[0] = reg.coef_[0]

    ax.plot(time[start_index:end_index + 1], I[start_index:end_index + 1], linestyle='-', linewidth=0.2,
            color='#FF4000', label='data')
    ax.plot(time[start_index:end_index + 1], np.exp(y_pred), linestyle='-', linewidth=0.2, color='#2E64FE',
            label='estimation')

    ax.set_xlabel('time')
    ax.legend()

    for i in range(1, tot_simulations):
        data = pd.read_csv(filepath_or_buffer=path + str(i) + ".csv", header=None)
        S = data[0].values
        I = data[2].values
        R = data[3].values
        time = data[4].values
        if R[-1] < (100):
            continue

        indexes = np.argwhere(time_interval[0] < time)
        start_index = indexes.min()
        indexes = np.argwhere(time < time_interval[1])
        end_index = indexes.max()

        reg = LinearRegression()

        x = time[start_index:end_index + 1].reshape(-1, 1)
        reg.fit(x, np.log(I[start_index:end_index + 1]))
        y_pred = reg.predict(x)
        parameters = np.append(parameters, reg.coef_[0])

        myFun.print_estimation(time[start_index:end_index + 1], I[start_index:end_index + 1], np.exp(y_pred), ax)

        fig.show()

    return parameters.mean(), parameters.var()


def logistic_regression(algorithm, tot_simulations):
    path = myFun.get_path_of(algorithm)

    # plot style
    # plt.xkcd()
    plt.style.use("ggplot")
    plt.tight_layout()
    fig, ax = plt.subplots()

    plt.title(label="estimation of r using Recovered")

    data = pd.read_csv(filepath_or_buffer=path + str(0) + ".csv", header=None)
    I = data[2].values
    R = data[3].values
    time = data[4].values
    cumulative_cases: ndarray = np.cumsum(I)
    parameters = np.zeros(tot_simulations)
    tmp_parameters, covariance = curve_fit(myFun.logistic_function, time, R, p0=(0.3, R[-1]),
                                           bounds=([0, R[-1] - (R[-1] * 0.0005)], np.inf), method='trf')
    parameters[0] = tmp_parameters[0]

    ax.plot(time, R, linestyle='-', linewidth=0.2, color='#FF4000', label='data')
    ax.plot(time, myFun.logistic_function(time, *tmp_parameters), linestyle='-', linewidth=0.2, color='#2E64FE',
            label='estimation')

    ax.set_xlabel('time')
    ax.legend()
    for i in range(1, tot_simulations):
        data = pd.read_csv(filepath_or_buffer=path + str(i) + ".csv", header=None)
        S = data[0].values
        I = data[2].values
        R = data[3].values
        time = data[4].values
        if R[-1] < S[0] / 2:
            continue
        cumulative_cases: ndarray = np.cumsum(I)
        tmp_parameters, covariance = curve_fit(myFun.logistic_function, time, R, p0=(0.3, R[-1]),
                                               bounds=([0, R[-1] - (R[-1] * 0.0005)], np.inf), method='trf')
        parameters[i] = tmp_parameters[0]
        estimation = myFun.logistic_function(time, *tmp_parameters)
        myFun.print_estimation(time, R, estimation, ax)

    fig.show()

    return parameters.mean()


def S_infinity_in_relation_to_1_over_R_zero(tot_simulations, algorithm, nh, betaG, betaH, gamma, nu,
                                            number_of_households):
    R_0 = r.R_0_Household(nh, betaG, betaH, gamma, nu)
    path = myFun.get_path_of(algorithm)
    S_infinity = []
    for i in range(0, tot_simulations):
        data = pd.read_csv(filepath_or_buffer=path + str(i) + ".csv", header=None)
        S = data[0].values
        if S[-1] < S[0] / 2:
            # normalize the total population to 1
            S_infinity.append(S[-1] / (nh * number_of_households))
    S_infinity = np.array(S_infinity)
    print("For the " + algorithm + " algorithm we have that (normalized) S_infinity is:\n")
    print("S_infinity= " + str(S_infinity.mean()) + " (variance = " + str(S_infinity.var()) + ")\n")
    print("while 1/R0 is: " + str(1 / R_0) + "\n")
