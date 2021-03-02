import gc  # garbage collector
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import myFunctions as myFun
import analyze_data as ad
import plot_graph as myplot
import r_star as r

# parameters

nh = 3
betaG = 1.136
# betaG during lockdown =0.216
betaH = 2.27
gamma = 0.45
nu = 0.21

tot_simulations = 0

# print("R* is: " + str(r.Rstar(nh, betaG, betaH, gamma, nu)) + "\n")

states_to_id, id_to_states, number_of_states = r.states(nh)
print(number_of_states)

# transition_matrix = r.get_transition_matrix(nh, betaH, nu, gamma)
# print(transition_matrix)

algorithm = "gillespie_household"

# ad.exponential_regression(algorithm, 0)

# print("R0 Household is: " + str(r.R_0_Household(nh, betaG, betaH, gamma, nu)) + "\n")

# ad.analyze_my_data(algorithm, tot_simulations)


# myplot.plot_my_graph(algorithm, tot_simulations)
