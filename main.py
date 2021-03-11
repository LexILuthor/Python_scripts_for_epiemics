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

tot_simulations = 80

# available
# functions:


algorithm = "gillespie_household"

# outputR = open("results_on_various_r.txt", "w")


# estimates r using a logistic regression
# N:B. it is done following the numbered of Recovered individuals
# outputR.write("r estimated by the logistic regression in the " + algorithm + " algortithm is: " + str(    ad.logistic_regression(algorithm, tot_simulations)[0]) + "\n")
# print("r estimated by the logistic regression in the " + algorithm + " algortithm is: " + str(    ad.logistic_regression(algorithm, tot_simulations)) + "\n")

# R_0 computed using r using formula given by Prof
# R0est = r.R0_from_r(algorithm, tot_simulations, nu, gamma)
# print = ("R_0 computed using r " + R0est + "\n")
# outputR.write("R_0 computed using r: " + str(R0est) + "\n")

# Rstar in a household model computed following section 2.3 Pellis_2 and appendix A of Pellis_1 to compute mu_k
# dubbio riguardo a mu_G, per ora è calcolato come BetaG/gamma
# (remember mu_G is the mean number of global contacts fo an individual)
# Rstar = r.Rstar(nh, betaG, betaH, gamma, nu)
# print("R* is: " + str(Rstar) + "\n")
# outputR.write("R* is: " + str(Rstar) + "\n")

# R_0 for the model with household computed following Pellis_1 corollary 1
# dubbio riguardo a q(k) vedi quaderno come è stato calcolato
# R_0_Household = r.R_0_Household(nh, betaG, betaH, gamma, nu)
# print("R0 Household is: " + str(R_0_Household) + "\n")
# outputR.write("R0 Household is: " + str(R_0_Household) + "\n")

# get on a file the  spike of infected, S_infinity and much more in only one minute!
# ad.analyze_my_data(algorithm, tot_simulations)

# plot the gaphs of the simulations
# myplot.plot_my_graph(algorithm, tot_simulations)


# compute r following pellis_markow

# preliminar actions to find the interval [a,b] of r (necessary in the function "compute_growth_rate_r" for the parameter a,b)
# transition_matrix, states_to_id, id_to_states = myFun.get_continuous_transition_matrix(nh, betaH, nu, gamma)
# test = 10000
# re = np.zeros(test)
# y = np.zeros(test)
# for i in range(test):
#    re[i] = (int(i) / 100)-0
# for i in range(test):
#    y[i] = myFun.laplace_transform_infectious_profile(re[i], nh, betaG, transition_matrix, id_to_states, states_to_id,result=-1)

# fig, ax = plt.subplots()
# ax.plot(re, y, linestyle='-')
# fig.show()

# call the function
growth_rate_r = r.compute_growth_rate_r(nh, betaG, betaH, nu, gamma, 0, 10, initial_infected=1)
print("Growth rate r computed following Pellis_markov: " + str(growth_rate_r))
# outputR.write("Growth rate computed following Pellis_markov: " + str(growth_rate_r))

# compute R_star following pellis_r
Rstar = r.compute_Rstar(nh, betaG, betaH, nu, gamma, initial_infected=1)
print("R_star computed following pellis_r: " + str(Rstar))
# outputR.write("R_star computed following pellis_r: " + str(Rstar))

# outputR.close()
