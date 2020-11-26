import numpy as np


# import numba as nb

class degradation(object):
    def __init__(self):
        self.R_ug = 8.314
        self.F = 96485.3329


class deg_ncm(degradation):  # [DOI: 10.1016/j.jpowsour.2014.02.012]
    def __init__(self, temperature, voltage, SOC, Q, time):
        super(deg_ncm, self).__init__()
        self.temperature = temperature
        self.voltage = voltage
        self.SOC = SOC
        self.Q = Q
        self.time = time
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

    # @nb.jit()
    def capaloss(self, alpha, beta, tpy):
        '''
        To indicate at which point the caploss reached 20%.
        fig, ax = plt.subplots()
        ax.plot(
            Dyn_MFA_System.IndexTable['Classification']['Time'].Items,
            caploss
        )
        plt.annotate(text='Capacity 80% remaining',xy=(2012.6,caploss[8]),
        xytext=(2010,0.12), color='r', arrowprops=dict(arrowstyle='-|>',
                                                        color='k'))
        ax.set_title('Capacity loss for LIBs')
        ax.set_xlabel('Year')
        ax.set_ylabel('Capacity Loss')
        ax.grid()
        '''
        self.tpy = tpy
        self.alphacap = alpha
        self.betacap = beta
        self.caploss = np.zeros((self.time, 1))
        self.resloss = np.zeros((self.time, 1))

        for t in range(self.time):
            self.caploss[t, :] = self.alphacap * (((t + 1) * 365) ** 0.75) + self.betacap * \
                                 np.sqrt([self.Q * (t + 1) * self.tpy])
            self.resloss[t, :] = self.alphares * (((t + 1) * 365) ** 0.75) + self.betares * \
                                 self.Q * (t + 1) * self.tpy
        return self.caploss

    def rep_year(self, soh):
        self.repurp_year = np.min(np.where(self.caploss > (1 - soh / 100)), 1)[0]
        return self.repurp_year


class degrad_ppm(degradation):  # [DOI: 10.23919/ACC.2017.7963578 ]
    def __init__(self):
        super(degrad_ppm, self).__init__()
        self.U_ref = 0.08
        self.T_ref = 298.15
        self.V_ref = 3.7  # Kandler Smith 2017

    @staticmethod
    def uv_soc(SOC):
        """
        :param SOC:[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].
        :return: An 1*2 list in which [0] is U_ under that SOC, [1] is V_OC under that SOC.
        """
        lookup_table = {
            '0': [1.2868, 3.0000],
            '0.1': [0.2420, 3.4679],
            '0.2': [0.1818, 3.5394],
            '0.3': [0.1488, 3.5950],
            '0.4': [0.1297, 3.6453],
            '0.5': [0.1230, 3.6876],
            '0.6': [0.1181, 3.7469],
            '0.7': [0.1061, 3.8400],
            '0.8': [0.0925, 3.9521],
            '0.9': [0.0876, 4.0668],
            '1.0': [0.0859, 4.1934]
        }
        if str(SOC) not in lookup_table.keys():
            raise ValueError('SOC can be only in range[0, 1.0, 11].')

        return lookup_table.get(str(SOC))

    def get_Q_Li(self, DOD_max, Tem, U_, V_OC, time, N, T_RPT, E_ad02=74860, tao_b3=5, b1_ref=0.003503, E_ab1=35392,
                 alpha_b1=1.0, gam_b1=2.472, beta_b1=2.157, b2_ref=1.541e-5, E_ab2=-36000,
                 b3_ref=0.02805, E_ab3=42800, alpha_b3=0.0066, theta=0.135, E_ad01=34300, d0_ref=75.10,
                 DOD=None):
        self.DODmax = DOD_max
        self.Tem = Tem
        self.U_ = U_
        self.V_OC = V_OC
        self.b1_ref = b1_ref
        self.E_ab1 = E_ab1
        self.alpha_b1 = alpha_b1
        self.gam_b1 = gam_b1
        self.beta_b1 = beta_b1
        self.b2_ref = b2_ref
        self.E_ab2 = E_ab2
        self.b3_ref = b3_ref
        self.E_ab3 = E_ab3
        self.alpha_b3 = alpha_b3
        self.DOD = DOD
        self.theta = theta
        self.r_d_0ref = d0_ref  # Ah
        self.T_RPT = T_RPT  # Kandler Smith, 2017
        self.E_ad01 = E_ad01
        self.E_ad02 = E_ad02

        self.r_d_0 = self.r_d_0ref * (np.exp(-self.E_ad01 / self.R_ug * (1 / self.T_RPT - 1 / self.T_ref) -
                                             ((self.E_ad02 / self.R_ug) ** 2) *
                                             ((1 / self.T_RPT - 1 / self.T_ref) ** 2)))

        self.r_b_1 = self.b1_ref * np.exp(((1 / self.Tem) - (1 / self.T_ref)) * (- self.E_ab1 / self.R_ug)) * \
                     np.exp((self.U_ / self.Tem - self.V_ref / self.T_ref) * self.alpha_b1 * self.F / self.R_ug) * \
                     np.exp(self.gam_b1 * (self.DODmax ** self.beta_b1))

        self.r_b_2 = self.b2_ref * np.exp(((1 / self.Tem) - (1 / self.T_ref)) * (- self.E_ab2 / self.R_ug)) * \
                     (self.DOD ** 2)

        self.r_b_3 = self.b3_ref * np.exp(((1 / self.Tem) - (1 / self.T_ref)) * (- self.E_ab3 / self.R_ug)) * \
                     np.exp((self.U_ / self.Tem - self.V_ref / self.T_ref) * self.alpha_b3 * self.F / self.R_ug) * \
                     (1 + self.DODmax * self.theta)

        self.r_b_0 = 1.07
        self.time = time  # days
        self.N = N
        self.tao_b3 = tao_b3

        self.Q_Li = self.r_d_0 * (self.r_b_0 - self.r_b_1 * (self.time ** 0.5) - self.r_b_2 * self.N -
                                  self.r_b_3 * (1 - np.exp(-self.time / self.tao_b3)))
        return self.Q_Li, self.r_d_0

    def get_Q_neg(self, Tem, N, DOD, c0_ref=75.64, E_ac0=2224, c2_ref=3.9193e-3, E_ac2=-48260, beta_c2=4.54):
        self.Tem = Tem
        self.N = N
        self.DOD = DOD
        self.c0_ref = c0_ref
        self.c2_ref = c2_ref
        self.E_ac0 = E_ac0
        self.E_ac2 = E_ac2
        self.beta_c2 = beta_c2

        self.c2 = self.c2_ref * np.exp(-self.E_ac2 / self.R_ug * (1 / self.Tem - 1 / self.T_ref) *
                                       (self.DOD ** self.beta_c2))
        self.c0 = self.c0_ref * np.exp(-self.E_ac0 / self.R_ug * (1 / self.Tem - 1 / self.T_ref))

        self.Q_neg = ((self.c0 ** 2) - 2 * self.c2 * self.c0 * self.N) ** 0.5

        return self.Q_neg

    def get_Q_pos(self, Ah_dis, T_RPT, r_d_0, r_d_3=0.46):
        self.Ah_dis = Ah_dis
        self.T_RPT = T_RPT
        self.r_d_0 = r_d_0
        self.r_d_3 = r_d_3

        self.Q_pos = self.r_d_0 + self.r_d_3 * (1 - np.exp(-self.Ah_dis / 228))

        return self.Q_pos
