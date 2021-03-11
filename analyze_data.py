from numpy import ndarray
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

import myFunctions as myFun


def analyze_my_data(algorithm, tot_simulations):
    path = myFun.get_path_of(algorithm)
    infected_peak = []
    infected_peak_time = []
    total_infected = []
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
            major_outbreak = major_outbreak + 1

    myFun.print_analysis(algorithm, tot_simulations, np.array(infected_peak), np.array(infected_peak_time),
                         np.array(total_infected), major_outbreak)


def logistic_regression(algorithm, tot_simulations):
    path = myFun.get_path_of(algorithm)

    # plot style
    # plt.xkcd()
    plt.style.use("ggplot")
    plt.tight_layout()
    fig, ax = plt.subplots()

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
        myFun.print_simulation(time, R, ax, tmp_parameters)

    fig.show()

    return parameters.mean()
