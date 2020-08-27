import scipy.special as sc
import numpy as np


class weibull():
    def __init__(self, rep_year, beta, rho, time):
        self.alpha = rep_year / (sc.gamma(rho + 1 / beta) / sc.gamma(rho))
        self.beta = beta
        self.rho = rho
        self.time = time


    def eol_total(self):
        self.eol_total = np.zeros((self.time, 1))
        self.eol_total[0] = 1
        b = np.zeros((self.time, 1))
        for i in range(1, self.time):
            b[i] = (i / self.alpha) ** self.beta
            self.eol_total[i] = 1 - sc.gammainc(self.rho, b[i])
        print('b=' + str(b), 'alpha=' + str(self.alpha))
        return self.eol_total

    def eol_per_y(self, eol_total):
        self.eol_total = eol_total
        self.eol_per_y = np.zeros((self.time, 1))
        for i in range(1, self.time):
            self.eol_per_y[i-1] = self.eol_total[i-1] - self.eol_total[i]
        return self.eol_per_y
