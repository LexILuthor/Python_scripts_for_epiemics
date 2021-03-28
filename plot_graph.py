import gc  # garbage collector
import matplotlib.pyplot as plt
import pandas as pd
import myFunctions as myFun


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
    ax.set_xlim([0, 110])

    # read the dataset
    S, E, I, R, time = myFun.read_dataset(path + "0.csv")

    if log_scale:
        ax.set_yscale('log')
        ax.set_ylim([1, S[0] + 1])

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
        fig.show()
        if i % 8 == 0:
            gc.collect()
        print(i)
    if log_scale:
        fig.savefig("graph_of_" + algorithm + "log_scale.png")
    else:
        fig.savefig("graph_of_" + algorithm + ".png")
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
    #plt.tight_layout()

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
