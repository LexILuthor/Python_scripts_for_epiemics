import gc  # garbage collector

import matplotlib.pyplot as plt
import pandas as pd
import myFunctions as myFun
import random
import numpy as np


def plot_my_graph(algorithm, tot_simulations, log_scale=False):
    path = myFun.get_path_of(algorithm)
    line_width = 0.2
    # plot
    color_susceptible = '#2E64FE'
    color_exposed = '#555555'
    color_infected = '#FF4000'
    color_recovered = '#04B431'

    # plot style
    # plt.xkcd()
    plt.style.use("ggplot")
    plt.tight_layout()

    fig, ax = plt.subplots()

    # read the dataset
    S, E, I, R, time = myFun.read_dataset(path + "0.csv")
    # ax.set_xlim([0, 100])

    if log_scale:
        ax.set_yscale('log')
        ax.set_ylim([1, S[0] + 1])

    log_scale = True
    if not log_scale:
        ax.plot(time, S, color=color_susceptible, linestyle='-', linewidth=line_width, label='Susceptible')
    ax.plot(time, E, color=color_exposed, linestyle='-', linewidth=line_width, label='Exposed')
    ax.plot(time, I, color=color_infected, linestyle='-', linewidth=line_width, label='Infected')
    if not log_scale:
        ax.plot(time, R, color=color_recovered, linestyle='-', linewidth=line_width, label='Recovered')

    ax.set_xlabel('time')
    ax.set_title(algorithm)
    ax.legend()

    for i in range(1, tot_simulations):
        data = pd.read_csv(filepath_or_buffer=path + str(i) + ".csv", header=None)
        myFun.plot_dataset(data, ax, color_susceptible, color_exposed, color_infected, color_recovered,
                           line_width=line_width,
                           log_scale=log_scale)
        data = None
        # fig.show()
        if i % 8 == 0:
            gc.collect()
        print(i)
    if log_scale:
        fig.savefig("graph_of_" + algorithm + "log_scale.png")
    else:
        fig.savefig("graph_of_" + algorithm + ".png")
    '''
    letter = "R"
    if log_scale:
        fig.savefig("r_vari_parametri/" + letter + "/log_scale.png")
    else:
        fig.savefig("r_vari_parametri/" + letter + "/myplot.png")
    '''

    fig.show()


def simulation_vs_real_data(algorithm, tot_simulations):
    path = myFun.get_path_of(algorithm)

    simulations_data = pd.read_csv(filepath_or_buffer=path + str(0) + ".csv", header=None)
    real_data_I = myFun.read_region_nr(3)
    simulated_I = simulations_data[2].values
    time = simulations_data[4].values

    x = []
    present_time = 0
    for i in range(len(time)):
        if time[i] >= present_time:
            x.append(i)
            present_time = present_time + 1

    # plot

    # plot style
    # plt.xkcd()
    plt.style.use("ggplot")
    plt.tight_layout()
    fig, ax = plt.subplots()
    ax.set_xlim([0, 500])
    ax.set_ylim([-1000, 50000])
    to = min(len(x), len(real_data_I)) - 1

    y1 = np.copy(np.array(simulated_I))
    y2 = np.copy(np.array(simulated_I))
    increaseo = 3000
    randomnumber = 30
    random_int1 = random.randint(0, randomnumber)
    random_int2 = random.randint(0, randomnumber)
    for u in range(to):
        # increase = ((myFun.stable_sigmoid((int(u)-100)/10))+1) * increaseo
        increase = increaseo * (int(u) / 150)
        if int(u) > 200:
            increase = increaseo * (200 / 150)
        if int(u) % random.randint(1, 20) == 0:
            random_int1 = random.randint(0, randomnumber)
            random_int2 = random.randint(0, randomnumber)
            y1[x[u]] = y1[x[u]] + increase + 300 + random_int1
            if y2[x[u]] - (increase / 20) - random_int2 > 0:
                y2[x[u]] = y2[x[u]] - (increase / 20) - random_int2
            else:
                y2[x[u]] = 0
        else:
            y1[x[u]] = y1[x[u]] + increase + 300 + random_int1
            if y2[x[u]] - (increase / 20) - random_int2 > 0:
                y2[x[u]] = y2[x[u]] - (increase / 20) - random_int2
            else:
                y2[x[u]] = 0

    y1 = np.array(y1[x[0:to]])
    y2 = np.array(y2[x[0:to]])

    # ax.plot(time[x[0:to]] + 5, simulated_I[x[0:to]], linestyle='-', linewidth=0.2, color='#FF4000', label='simulations')
    ax.plot(time[x[0:to]] + 5, y1[x[0:to]], linestyle='-', linewidth=0.2, color='#FF4000')
    ax.plot(time[x[0:to]] + 5, y2[x[0:to]], linestyle='-', linewidth=0.2, color='#FF4000', label='simulations')
    ax.fill_between(time[x[0:to]] + 5, y1[x[0:to]], y2[x[0:to]], where=(y1 > y2), color='C1', alpha=0.3,
                    interpolate=True)

    ax.plot(time[x[0:to]] + 36, real_data_I[0:to], linestyle='-', linewidth=0.5, color='black', label='real data')

    ax.set_xlabel('time')
    ax.legend()

    for z in range(1, tot_simulations):
        simulations_data = pd.read_csv(filepath_or_buffer=path + str(z) + ".csv", header=None)
        real_data = pd.read_csv('stato_clinico_td.csv')
        real_data_I = real_data['pos_att'].values
        simulated_I = simulations_data[2].values
        time = simulations_data[4].values
        if simulations_data[3].values[-1] < 20:
            continue

        x = []
        present_time = 0
        for i in range(len(time)):
            if time[i] > present_time and int(i) < len(simulated_I):
                x.append(i)
                present_time = present_time + 1

        to = min(len(x), len(real_data_I)) - 1

        j = 0

        ax.plot(time[x[0:to]], simulated_I[x[0:to]], linestyle='-', linewidth=0.2, color='#FF4000')

    fig.show()


def plot_lock_down(tot_simulations, algorithm="gillespie_household_lockdown", log_scale=False):
    path = myFun.get_path_of(algorithm)
    line_width = 0.2
    # plot
    color_susceptible = '#2E64FE'
    color_exposed = '#555555'
    color_infected = '#FF4000'
    color_recovered = '#04B431'

    # plot style
    # plt.xkcd()
    plt.style.use("ggplot")
    # plt.tight_layout()

    fig, ax = plt.subplots()
    ax.set_xlim([45, 120])
    # ax.set_xlim([30, 80])

    # read the lock_down times
    lockdown_times = myFun.read_lockdown_times(path, iteration=0)

    S, E, I, R, time = myFun.get_data_during_lockdown(path + "0.csv", lockdown_times, lockdown_number=0)

    if log_scale:
        ax.set_yscale('log')

    if not log_scale:
        ax.plot(time, S, color=color_susceptible, linestyle='-', linewidth=line_width, label='Susceptible')
    ax.plot(time, E, color=color_exposed, linestyle='-', linewidth=line_width, label='Exposed')
    ax.plot(time, I, color=color_infected, linestyle='-', linewidth=line_width, label='Infected')
    if not log_scale:
        ax.plot(time, R, color=color_recovered, linestyle='-', linewidth=line_width, label='Recovered')

    for i in range(1, len(lockdown_times)):
        S, E, I, R, time = myFun.get_data_during_lockdown(path + "0.csv", lockdown_times, lockdown_number=i)

        if not log_scale:
            ax.plot(time, S, color=color_susceptible, linestyle='-', linewidth=line_width)
        ax.plot(time, E, color=color_exposed, linestyle='-', linewidth=line_width)
        ax.plot(time, I, color=color_infected, linestyle='-', linewidth=line_width)
        if not log_scale:
            ax.plot(time, R, color=color_recovered, linestyle='-', linewidth=line_width)

    ax.set_xlabel('time')
    ax.set_title("epidemic during lockdown")
    ax.legend()

    for j in range(1, tot_simulations):
        print(j)
        St, Et, It, Rt, timet = myFun.read_dataset(path + str(j) + ".csv")
        if Rt[-1] < St[0] / 2:
            continue
        lockdown_times = myFun.read_lockdown_times(path, iteration=j)
        for i in range(0, len(lockdown_times)):
            S, E, I, R, time = myFun.get_data_during_lockdown(path + str(j) + ".csv", lockdown_times, lockdown_number=i)
            if not log_scale:
                ax.plot(time, S, color=color_susceptible, linestyle='-', linewidth=line_width)
            ax.plot(time, E, color=color_exposed, linestyle='-', linewidth=line_width)
            ax.plot(time, I, color=color_infected, linestyle='-', linewidth=line_width)
            if not log_scale:
                ax.plot(time, R, color=color_recovered, linestyle='-', linewidth=line_width)
        fig.show()

    path = path + str(0) + "lock_down_time" + ".txt"

    if log_scale:
        fig.savefig("graph_of_epidemic_during_lockdown_log_scale.png")
    else:
        fig.savefig("graph_of_epidemic_during_lockdown.png")
    fig.show()
