from comfa.deg_func import deg_ncm
from comfa.Generator import LIB_ob
import pandas as pd
import numpy as np
import multiprocessing as mp
import time

start = time.time()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
def mainbody(iter):
    # Define parameters.
    Values_EV_Sales = pd.read_excel(r'C:\Users\tieae\Downloads\研究相关\Matlab Research Project\MFA Model\FlowCatalog.xlsx',
                    sheet_name='Sheet2', usecols='B').values

    # r'C:\Users\tieae\Downloads\研究相关\Matlab Research Project\MFA Model\FlowCatalog.xlsx',
    #                                 sheet_name='Sheet2', usecols='B'
    # r'/Users/zhoujingwei/OneDrive/文档/Matlab Research Project/FlowCatalog.xlsx',
    #                                 sheet_name='Sheet2', usecols='B'

    B = pd.DataFrame(columns=range(len(Values_EV_Sales)))
    for i in range(iter + 1):
        for o in range(Values_EV_Sales[i, 0]):
            B.loc[o, i] = LIB_ob(80, i + 2005, i + 2013).yearin()

    B.to_excel('test.xls')

def multi():
    pool = mp.Pool()
    pool.map(mainbody, range(8))


if __name__ == '__main__':
    multi()

    end = time.time()

    print('\n' + 'Time consumption: ' + str(end - start) + ' s.')
