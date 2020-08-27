import numpy as np


class degradation():
    def __init__(self, temperature, voltage, SOC, Q, time):
        self.temperature = temperature
        self.voltage = voltage
        self.SOC = SOC
        self.Q = Q
        self.time = time


class deg_ncm(degradation):
    def __init__(self, temperature, voltage, SOC, Q, time):
        super(deg_ncm, self).__init__(temperature, voltage, SOC, Q, time)
        self.phiV = 3.7
        # self.alphacap = self.alphares = self.betacap = self.betares = self.caploss = \
        #     self.resloss = self.rep_year = 0

    def alpha(self):
        self.alphacap = (7.543 * self.voltage - 23.75) * (10 ** 6) * np.exp(-6976 / self.temperature)
        self.alphares = (5.270 * self.voltage - 16.32) * (10 ** 5) * np.exp(-5986 / self.temperature)
        return self.alphacap

    def beta(self):
        self.betacap = 7.348 * (10 ** -3) * ((self.phiV - 3.667) ** 2) + 7.600 * (10 ** -4) + \
                       4.081 * (10 ** -3) * (1 - self.SOC)
        self.betares = 2.153 * (10 ** -4) * ((self.phiV - 3.725) ** 2) - 1.521 * (10 ** -5) + \
                       2.798 * (10 ** -4) * (1 - self.SOC)
        return self.betacap

    def capaloss(self, alpha, beta):
        self.alphacap = alpha
        self.betacap = beta
        self.caploss = np.zeros((self.time, 1))
        self.resloss = np.zeros((self.time, 1))

        for t in range(self.time):
            self.caploss[t, :] = self.alphacap * (((t + 1) * 365) ** 0.75) + self.betacap * \
                                 np.sqrt([self.Q * (t + 1) * 52])
            self.resloss[t, :] = self.alphares * (((t + 1) * 365) ** 0.75) + self.betares * \
                                 self.Q * (t + 1) * 52
        return self.caploss

    def rep_year(self, caploss):
        self.caploss = caploss
        self.rep_year = np.min(np.where(self.caploss > 0.2), 1)[0]
        # print(rep_year)
        return self.rep_year

    # def read_deg_curve(self, caploss, rep_year):
