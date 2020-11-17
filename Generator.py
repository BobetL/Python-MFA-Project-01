class LIB_ob(object):

    count = 0

    def __init__(self, SOH, yrin, yrout):
        LIB_ob.count += 1
        self.type = ['Pack', 'Module', 'Cell', 'Cathode', 'Cobalt']
        self.SOH = SOH
        self.yrin = yrin  # The year it entered the system.
        self.yrout = yrout  # the year it went out of the system.

        if self.SOH > 100 or self.SOH < 0:
            raise ValueError('SOH must in the range [0, 100], please check it again!')

    def typecheck(self, N):
        return self.type[N]

    def SOHcheck(self):
        return self.SOH

    def yearin(self):
        return self.yrin

    def yearout(self):
        return self.yrout

    def EoL(self):
        LIB_ob.count -= 1
        return LIB_ob.count

    @classmethod
    def print_count(cls):
        print('There are {} batteries in the system.'.format(cls.count))

if __name__ == '__main__':
    import numpy as np
    B = np.zeros((5, 5))
    for i in range(5):
        B[i] = LIB_ob(80, 2005, 2012).print_count()
