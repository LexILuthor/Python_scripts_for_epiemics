import r_star as r
import analyze_data as ad
import plot_graph as myplot
import numpy as np
import matplotlib.pyplot as plt

import myFunctions as myFun

# parameters
number_of_households = 200000
nh = 3
betaG = 0.5
betaH = 0.5
nu = 0.5
gamma = 0.25

tot_simulations = 10

# available
# functions:


algorithm = "gillespie_household"

# outputR = open("results_on_various_r.txt", "w")

# ----------------------------------------------------------------------------------------------------------------------
# estimates r using a logistic regression

# N:B. it is done following the numbered of Recovered individuals
# growth_rate = ad.logistic_regression(algorithm, tot_simulations)
# outputR.write("r estimated by the logistic regression in the " + algorithm + " algortithm is: " + str(    ad.logistic_regression(algorithm, tot_simulations)[0]) + "\n")
# print("r estimated by the logistic regression in the " + algorithm + " algortithm is: " + str(growth_rate) + "\n")

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# estimates r using linear regression

# N:B. it is done following the numbered of Infected individuals
# growth_rate, sd = ad.exponential_regression(algorithm, tot_simulations)
# print("r estimated by the linear regression in the " + algorithm + " algortithm is: " + str(    growth_rate) + " standard deviation " + str(sd) + "\n")
# outputR.write("r estimated by the linear regression in the " + algorithm + " algortithm is: " + str(growth_rate[0]) + "\n")

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------

# compute r following pellis_markow
growth_rate_r = r.compute_growth_rate_r(nh, betaG, betaH, nu, gamma, 0, 5, initial_infected=1)
print("Growth rate r computed following Pellis_markov: " + str(growth_rate_r))
# outputR.write("Growth rate computed following Pellis_markov: " + str(growth_rate_r))

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------

# plot the gaphs of the simulations

# myplot.plot_my_graph(algorithm, tot_simulations, log_scale=True)

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------

# growth rate r in SEIR model with 3 individuals per household
growth_rate_r = r.growth_rate_r_SEIR_3(nh, betaG, betaH, nu, gamma)
print("Growth rate r in SEIR with only 3 people: " + str(growth_rate_r))
# outputR.write("Growth rate r in SIR: " + str(growth_rate_r))

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------

# growth rate r in SIR model with 3 individuals per household
growth_rate_r = r.growth_rate_r_SIR(nh, betaG, betaH, gamma)
print("Growth rate r in SIR: " + str(growth_rate_r))
# outputR.write("Growth rate r in SIR: " + str(growth_rate_r))

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------

# plot the gaphs only during lock-down (algorithm is always gillespie_household_lockdown)

# myplot.plot_lock_down(tot_simulations,log_scale=False)

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------

# Rstar in a household model computed following section 2.3 Pellis_2 and appendix A of Pellis_1 to compute mu_k

# dubbio riguardo a mu_G, per ora è calcolato come BetaG/gamma
# (remember mu_G is the mean number of global contacts fo an individual)
# Rstar = r.Rstar_following_pellis_2(nh, betaG, betaH, nu, gamma)
# print("R* following pellis_2: " + str(Rstar) + "\n")
# outputR.write("R* is: " + str(Rstar) + "\n")

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------

# R_0 for the model with household computed following Pellis_1 corollary 1

# dubbio riguardo a q(k) vedi quaderno come è stato calcolato
# R_0_Household = r.R_0_Household(nh, betaG, betaH, gamma, nu)
# print("R0 Household is: " + str(R_0_Household) + "\n")
# outputR.write("R0 Household is: " + str(R_0_Household) + "\n")

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------

# get on a file the   S_infinity, total number of infected,  and much more in only one minute!

# ad.analyze_my_data(algorithm, tot_simulations)

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------

# print the value of 1/R0 and S_infinity
# note, R_0 is computed following the household formula using the function "R_0_Household(nh, betaG, betaH, gamma, nu)"

# !results wrong!

# ad.S_infinity_in_relation_to_1_over_R_zero(tot_simulations, algorithm, nh, betaG, betaH, gamma, nu, number_of_households)

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------

# compute r following pellis_markow

# preliminar actions to find the interval [a,b] of r (necessary in the function "compute_growth_rate_r" for the parameter a,b)
'''
transition_matrix, states_to_id, id_to_states = myFun.get_QH(nh, betaH, nu, gamma)
test = 10000
re = np.zeros(test)
y = np.zeros(test)
for i in range(test):
    re[i] = (int(i) / (test * 100)) + 0.37
for i in range(test):
    y[i] = myFun.laplace_transform_infectious_profile(re[i], nh, betaG, transition_matrix, states_to_id, id_to_states,
                                                      result=-1)

fig, ax = plt.subplots()
ax.plot(re, y, linestyle='-')
fig.show()
'''

# call the function

# compute r following pellis_markow
# growth_rate_r = r.compute_growth_rate_r(nh, betaG, betaH, nu, gamma, 0, 1, initial_infected=1)
# print("Growth rate r computed following Pellis_markov: " + str(growth_rate_r))
# outputR.write("Growth rate computed following Pellis_markov: " + str(growth_rate_r))

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------

# R_star = r.Rstar_from_r(nh, betaG, betaH, nu, gamma, a=0, b=1, initial_infected=1)
# print("R_star computed from r using my formula: " + str(R_star))

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------

# compute a new betaG given r and a new nh
# first compute r for the model with the usual parameters
# growth_rate_r = r.compute_growth_rate_r(nh, betaG, betaH, nu, gamma, a=0, b=10, initial_infected=1)
# new_nh = 21
# new_betaG = r.betaG_given_r(new_nh, growth_rate_r, betaG, betaH, nu, gamma, initial_infected=1)
# print("new betaG " + str(new_betaG))

# ----------------------------------------------------------------------------------------------------------------------

# compute R_star following pellis_markov

# Rstar = r.compute_Rstar_following_pellis_markov(nh, betaG, betaH, nu, gamma, initial_infected=1)
# print("R_star computed following pellis_markov: " + str(Rstar))
# outputR.write("R_star computed following pellis_r: " + str(Rstar))

# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# R_0 computed using r using formula given by Prof

# R0est = r.R0_from_r(algorithm, tot_simulations, nu, gamma)
# print = ("R_0 computed using r " + R0est + "\n")
# outputR.write("R_0 computed using r: " + str(R0est) + "\n")
# ----------------------------------------------------------------------------------------------------------------------


# outputR.close()
