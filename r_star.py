import myFunctions as myFun
import analyze_data as ad
from scipy.optimize import brentq
import numpy as np


def Rstar(nh, betaG, betaH, gamma, nu):
    # Go to function p in fiile myFun.py to change the way q(k) is computed (using Pellis method or using trasition matrix)
    summation = 0
    for k in range(0, nh - 1):
        summation = summation + myFun.mu(nh, 1, nh - 1, k, betaH, gamma, nu)
    return summation * (betaG / gamma)


def R_0_Household(nh, betaG, betaH, gamma, nu):
    epsilon = 0.0001
    return brentq(myFun.g_nh, 0 + epsilon, 20, args=(nh, betaG, betaH, gamma, nu))


def R0_from_r(algorithm, tot_simulations, nu, gamma):
    r = ad.logistic_regression(algorithm, tot_simulations)[0]
    R0 = 1 + (r * (r + nu + gamma) / (nu * gamma))
    return R0
