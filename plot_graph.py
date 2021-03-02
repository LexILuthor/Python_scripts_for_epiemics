import gc  # garbage collector
import matplotlib.pyplot as plt
import pandas as pd
import myFunctions as myFun


def plot_my_graph(algorithm, tot_simulations):
    path = myFun.get_path_of(algorithm)

    # plot
    color_susceptible = '#2E64FE'
    color_exposed = '#555555'
    color_infected = '#FF4000'
    color_recovered = '#04B431'

    # plot style
    # plt.xkcd()
    # plt.style.use("ggplot")
    plt.tight_layout()

    fig, ax = plt.subplots()
    ax.set_xlim([0, 250])

    # read the dataset
    data = pd.read_csv(filepath_or_buffer=path + "0.csv", header=None)
    S = data[0].values
    E = data[1].values
    I = data[2].values
    R = data[3].values
    time = data[4].values

    ax.plot(time, S, color=color_susceptible, linestyle='-', linewidth=0.2, label='Susceptible')
    ax.plot(time, E, color=color_exposed, linestyle='-', linewidth=0.2, label='Exposed')
    ax.plot(time, I, color=color_infected, linestyle='-', linewidth=0.2, label='Infected')
    ax.plot(time, R, color=color_recovered, linestyle='-', linewidth=0.2, label='Recovered')

    ax.set_xlabel('time')
    ax.set_title(algorithm)
    ax.legend()

    for i in range(1, tot_simulations):
        data = pd.read_csv(filepath_or_buffer=path + str(i) + ".csv", header=None)
        myFun.plot_dataset(data, ax, color_susceptible, color_exposed, color_infected, color_recovered)
        data = None
        # fig.show()
        if i % 8 == 0:
            gc.collect()
        print(i)

    fig.savefig("graph_of_" + algorithm + ".png")
    fig.show()
