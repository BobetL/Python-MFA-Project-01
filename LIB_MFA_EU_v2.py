import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
# from comfa.Generator import LIB_ob
from comfa.deg_func import deg_ncm
from comfa.distributions import weibull
import ODYM_Classes as msc

start = time.time()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

ModelClassification = {
    'Time': msc.Classification(Name='Time',
                               Dimension='Time',
                               ID=1,
                               Items=list(np.arange(2005, 2031))),
    'Age-cohort': msc.Classification(Name='Age-cohort',
                                     Dimension='Age-cohort',
                                     ID=1,
                                     Items=list(np.arange(2005, 2031))),
    'Product': msc.Classification(Name='Product',
                                  Dimension='Product',
                                  ID=2,
                                  Items=['LIB_Pack', 'LIB_Module', 'LIB_Cell']),
    'Cathode': msc.Classification(Name='Cathode',
                                  Dimension='Cathode',
                                  ID=3,
                                  Items=['NCM_Cathode']),
    'Element': msc.Classification(Name='Elements',
                                  Dimension='Element',
                                  ID=4,
                                  Items=['Lithium', 'Cobalt', 'Nickel', 'Manganese'])
}

Model_Time_Start = int(min(ModelClassification['Time'].Items))
Model_Time_End = int(max(ModelClassification['Time'].Items))
Model_Duration = Model_Time_End - Model_Time_Start

IndexTable = pd.DataFrame({
    'Aspect': ['Time', 'Age-cohort', 'Product', 'Cathode', 'Element'],
    'Description': ['Model aspect "time"', 'Model aspect "Age-cohort"', 'Model aspect "Product"',
                    'Model aspect "Cathode"', 'Model aspect"Element"'],
    'Dimension': ['Time', 'Age-cohort', 'Product', 'Cathode', 'Element'],
    'Classification': [ModelClassification[Aspect] for Aspect in
                       ['Time', 'Age-cohort', 'Product', 'Cathode', 'Element']],
    'IndexLetter': ['t', 'c', 'p', 'm', 'e']
})  # The index letter 'm' is for 'Cathode', otherwise if it is not 'm', there will be an error.

IndexTable.set_index('Aspect', inplace=True)
IndexTable['IndexSize'] = pd.Series(
    [len(IndexTable.Classification[i].Items) for i in range(0, len(IndexTable.IndexLetter))], index=IndexTable.index)
IndexTable_ClassificationNames = [IndexTable.Classification[i].Name for i in range(0, len(IndexTable.IndexLetter))]
Nt = len(IndexTable.Classification[IndexTable.index.get_loc('Time')].Items)
NC = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('c')].Items)
NP = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('p')].Items)
NM = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('m')].Items)
NE = len(IndexTable.Classification[IndexTable.set_index('IndexLetter').index.get_loc('e')].Items)
LIB_EU_MFA = msc.MFAsystem(
    Name='UsedLIBsFlow',
    Geogr_Scope='Europe',
    Unit='',
    ProcessList=[],
    FlowDict={},
    StockDict={},
    ParameterDict={},
    Time_Start=Model_Time_Start,
    Time_End=Model_Time_End,
    IndexTable=IndexTable,
    Elements=IndexTable.loc['Element'].Classification.Items
)

# Processes.
LIB_EU_MFA.ProcessList = []
LIB_EU_MFA.ProcessList.append(msc.Process(Name='LIBs in the Market', ID=0))
LIB_EU_MFA.ProcessList.append(msc.Process(Name='Lost Batteries while Using', ID=1))
LIB_EU_MFA.ProcessList.append(msc.Process(Name='Spent Batteries Collected by Dismantle', ID=2))
LIB_EU_MFA.ProcessList.append(msc.Process(Name='Lost Batteries in Dismantle', ID=3))
LIB_EU_MFA.ProcessList.append(msc.Process(Name='Semi-Production Manufacturing', ID=4))
LIB_EU_MFA.ProcessList.append(msc.Process(Name='Cathode-D', ID=5))
LIB_EU_MFA.ProcessList.append(msc.Process(Name='Recycling-D', ID=6))
LIB_EU_MFA.ProcessList.append(msc.Process(Name='Cathode-B', ID=7))
LIB_EU_MFA.ProcessList.append(msc.Process(Name='Recycling-B', ID=8))
LIB_EU_MFA.ProcessList.append(msc.Process(Name='Supplier', ID=9))
LIB_EU_MFA.ProcessList.append(msc.Process(Name='Solar_PV', ID=10))
LIB_EU_MFA.ProcessList.append(msc.Process(Name='EV', ID=11))

# To read EV sales as initial input flow.
Values_EV_Sales = pd.read_excel(r'C:\Users\tieae\Downloads\研究相关\Matlab Research Project\MFA Model\FlowCatalog.xlsx',
                                sheet_name='Sheet2', usecols='B').values

# r'C:\Users\tieae\Downloads\研究相关\Matlab Research Project\MFA Model\FlowCatalog.xlsx',
#                                 sheet_name='Sheet2', usecols='B'
# r'/Users/zhoujingwei/OneDrive/文档/Matlab Research Project/FlowCatalog.xlsx',
#                                 sheet_name='Sheet2', usecols='B'


# Compositions
# To calculate the amount of LIBs in the EVs.
LIBs_Penetration_Rate1 = np.linspace(0.7, 0.8, 5, endpoint=False)
LIBs_Penetration_Rate2 = np.linspace(0.8, 1, 6)
LIBs_Penetration_Rate3 = np.linspace(1, 1, 15)
LIBs_Penetration_Rate = np.concatenate((LIBs_Penetration_Rate1, LIBs_Penetration_Rate2,
                                        LIBs_Penetration_Rate3))  # To concatenate 3 stairs of penetration rate.
LIBs_Penetration_Rate = LIBs_Penetration_Rate.reshape(
    LIBs_Penetration_Rate.shape[0], 1
)  # To transfer the obtained row into column.

Loss_Using_Rate1 = np.linspace(0.4, 0.1, 26)
# Loss_Using_Rate2 = np.linspace(0.1, 0.1, 5)
# Loss_Using_Rate1 = np.concatenate((Loss_Using_Rate1, Loss_Using_Rate2))
Loss_Using_Rate = Loss_Using_Rate1.reshape(Loss_Using_Rate1.shape[0], 1)  # Row to column.
LIBs_in_Market_Rate = LIBs_Penetration_Rate * (1 - Loss_Using_Rate)
LIBs_in_Market = Values_EV_Sales * LIBs_Penetration_Rate  # Batteries in the first use market before lost.

# To calculate the average lifetime of LIBs.
curve = deg_ncm(298, 3.667, 0.5, 2.15, 26)
caploss = curve.capaloss(curve.alpha(), curve.beta(), 52)  # 52 means charging once per week.
rep_year = curve.rep_year(80)  # The year when capacity drops to 80%.
rep_year_1 = curve.rep_year(60)
rep_year_2 = rep_year_1 - rep_year

# To calculate the lifetime distribution using Weibull distribution.
distr = weibull(rep_year, 3, 1, 26)
eolt = distr.eol_total()
eolp = distr.eol_per_y(eolt)

# Lifetime distribution will be set to uniform distribution.
EV_LT = 12  # EV lifetime 12 years.
LIB_Cell_LT_EV = rep_year
LIB_Module_LT_EV = rep_year + 1
LIB_Pack_LT_EV = LIB_Module_LT_EV + 1

LIB_Cell_LT_PV = rep_year_2
LIB_Module_LT_PV = rep_year_2 + 1
LIB_Pack_LT_PV = LIB_Module_LT_PV + 1

Loss_Dismantle_Rate = 0.1
toSecondUse = 1  # 100% of the flow can go to second-use stage, without any loss.

to_Cathode = 0.95  # https://patentimages.storage.googleapis.com/05/5d/7c/2f5fea9fdee86e/US20200078796A1.pdf
# 0327, Table 5 NCM.
# Table 8 shows Li, Ni, Co, Mn recycling reaction rates are all 100%.

Cathode_B2Rec = 1
Cathode_D2Rec = 1
Rec_B2Supp = 1
Rec_D2Supp = 1
Semi2PV = 0.6
Semi2EV = 0.4
Supp2PV = 0.6
Supp2EV = 0.4

Composition_Par_LIB = np.array([1, 8, 12])  # doi.org/10.1038/s41586-019-1682-5, BMW prismatic battery.
Composition_Par_Cell_Cathode = np.array([1])
Composition_Par_Cell_Ele = np.array(
    [0.012, 0.055, 0.055, 0.071]
)  # https://patentimages.storage.googleapis.com/05/5d/7c/2f5fea9fdee86e/US20200078796A1.pdf 0326, Table 1.

Mat_B_Dism = np.zeros(
    (Model_Duration + 1, Composition_Par_LIB[2])
)  # Batteries in the Dismantle stream reached EoL each year
Mat_B_EVYet = np.zeros(
    (Model_Duration + 1, Composition_Par_LIB[2])
)  # Batteries not reached EoL each year
Mat_2_D = np.zeros((Model_Duration + 1, Composition_Par_LIB[1] * Composition_Par_LIB[2]))  # Batteries go to dismantle
Mat_2_D_EoL = np.zeros(
    (Model_Duration + 1, Composition_Par_LIB[1] * Composition_Par_LIB[2]))  # Batteries go out from dismantle
F_cell = np.zeros((Model_Duration + 1, Composition_Par_LIB[2]))
F_module = np.zeros((Model_Duration + 1, Composition_Par_LIB[1]))
F_pack = np.zeros((Model_Duration + 1, Composition_Par_LIB[0]))

# Lifetime distribution of EVs
cardistr = weibull(EV_LT, 3, 1, 26)
careolt = cardistr.eol_total()
careolp = cardistr.eol_per_y(careolt)

# Process Parameters
LIB_EU_MFA.ParameterDict = {
    'EV_In': msc.Parameter(Name='EV_in_Market', ID=0, P_Res=0,
                           MetaData=None, Indices='t,c,p,m,e',
                           Values=Values_EV_Sales, Unit='1'),
    'F_In': msc.Parameter(Name='LIBs_in_Market', ID=1, P_Res=0,
                          MetaData=None, Indices='t,c,p,m,e',
                          Values=LIBs_in_Market, Unit='1'),
    'LossRate_1': msc.Parameter(Name='LossUsing', ID=2, P_Res=1,
                                MetaData=None, Indices='t',
                                Values=Loss_Using_Rate, Unit='1'),
    'Dismantle': msc.Parameter(Name='Car_Dismantle', ID=3, P_Res=2,
                               MetaData=None, Indices='t,c,p,m,e',
                               Values=None, Unit='1'),
    'Yet_EoL_EV': msc.Parameter(Name='Car_Not_EoL', ID=4, P_Res=None,
                                MetaData=None, Indices='t,c,p,m,e',
                                Values=None, Unit='1'),
    'LossRate_2': msc.Parameter(Name='LossDismantle', ID=5, P_Res=3,
                                MetaData=None, Indices='t',
                                Values=Loss_Dismantle_Rate, Unit='1'),
    'Pack_Semi_Rate_B': msc.Parameter(Name='Pack_2_Semi_B', ID=6, P_Res=4,
                                      MetaData=None, Indices='t,c,p,m,e',
                                      Values=None, Unit='1'),
    'Module_Semi_Rate_B': msc.Parameter(Name='Module_2_Semi_B', ID=7, P_Res=4,
                                        MetaData=None, Indices='t,c,p,m,e',
                                        Values=None, Unit='1'),
    'Cell_Semi_Rate_B': msc.Parameter(Name='Cell_2_Semi_B', ID=8, P_Res=4,
                                      MetaData=None, Indices='t,c,p,m,e',
                                      Values=None, Unit='1'),
    'Pack_Semi_Rate_D': msc.Parameter(Name='Pack_2_Semi_D', ID=9, P_Res=4,
                                      MetaData=None, Indices='t,c,p,m,e',
                                      Values=None, Unit='1'),
    'Module_Semi_Rate_D': msc.Parameter(Name='Module_2_Semi_D', ID=10, P_Res=4,
                                        MetaData=None, Indices='t,c,p,m,e',
                                        Values=None, Unit='1'),
    'Cell_Semi_Rate_D': msc.Parameter(Name='Cell_2_Semi_D', ID=11, P_Res=4,
                                      MetaData=None, Indices='t,c,p,m,e',
                                      Values=None, Unit='1')
}

EV_Mat = np.zeros((Nt, NC, NP, NM, NE))
EV_Mat_Yet = np.zeros((Nt, NC, NP, NM, NE))
# EV_Mat_Yet[0] = LIB_EU_MFA.ParameterDict['F_In'].Values[0]
# EV_Mat_Yet[1] = LIB_EU_MFA.ParameterDict['F_In'].Values[1]
# Interparam1 = np.zeros((Model_Duration + 1, 1))
# Interparam2 = np.zeros((Model_Duration + 1, 1))
# EV_Mat_Yet[1] = EV_Mat_Yet[0] + (LIB_EU_MFA.ParameterDict['F_In'].Values[0] *
#                                  (1 - LIB_EU_MFA.ParameterDict['LossRate_1'].Values[0]) * (1 - careolp[1])
#                                  + LIB_EU_MFA.ParameterDict['F_In'].Values[1] *
#                                  (1 - LIB_EU_MFA.ParameterDict['LossRate_1'].Values[1]) * (1 - careolp[0])) - \
#                 LIB_EU_MFA.ParameterDict['F_In'].Values[0] * \
#                 (1 - LIB_EU_MFA.ParameterDict['LossRate_1'].Values[0])
#
# EV_Mat_Yet[2] = EV_Mat_Yet[1] + (LIB_EU_MFA.ParameterDict['F_In'].Values[0] *
#                                  (1 - LIB_EU_MFA.ParameterDict['LossRate_1'].Values[0]) * (1 - careolp[2])
#                                  + LIB_EU_MFA.ParameterDict['F_In'].Values[1] *
#                                  (1 - LIB_EU_MFA.ParameterDict['LossRate_1'].Values[1]) * (1 - careolp[1])
#                                  + LIB_EU_MFA.ParameterDict['F_In'].Values[2] *
#                                  (1 - LIB_EU_MFA.ParameterDict['LossRate_1'].Values[1]) * (1 - careolp[0])) - \
#                 LIB_EU_MFA.ParameterDict['F_In'].Values[0] * \
#                 (1 - LIB_EU_MFA.ParameterDict['LossRate_1'].Values[0]) - \
#                 LIB_EU_MFA.ParameterDict['F_In'].Values[1] * \
#                 (1 - LIB_EU_MFA.ParameterDict['LossRate_1'].Values[1])
# EV_Mat_Yet[Model_Duration, Model_Duration] = (LIB_EU_MFA.ParameterDict['F_In'].Values[Model_Duration] *
#                                               (1 - LIB_EU_MFA.ParameterDict['LossRate_1'].Values[Model_Duration]) *
#                                               (1 - careolp[0]))
for cyear in range(1, Model_Duration + 1):
    # EV_Mat_Yet[caryear - 1, caryear - 1] = (LIB_EU_MFA.ParameterDict['F_In'].Values[caryear - 1] *
    #                                         (1 - LIB_EU_MFA.ParameterDict['LossRate_1'].Values[caryear - 1]) *
    #                                         (1 - careolp[0]))
    for pyear in range(cyear):
        EV_Mat[pyear, cyear, 0] = EV_Mat[pyear, cyear, 0] + (
                LIB_EU_MFA.ParameterDict['F_In'].Values[cyear] *
                (1 - LIB_EU_MFA.ParameterDict['LossRate_1'].Values[pyear])) * \
                                  careolp[cyear - pyear]
        EV_Mat_Yet[pyear, cyear, 0, :, :] = EV_Mat_Yet[pyear, cyear, 0, :, :] + (
                LIB_EU_MFA.ParameterDict['F_In'].Values[cyear] *
                (1 - LIB_EU_MFA.ParameterDict['LossRate_1'].Values[pyear])) * \
                                            (1 - careolp[cyear - pyear])
    # Interparam1[caryear] = Interparam1[caryear] + LIB_EU_MFA.ParameterDict['F_In'].Values[i] * \
    #                        (1 - LIB_EU_MFA.ParameterDict['LossRate_1'].Values[i] * (1 - careolp[caryear - i]))
    # Interparam2[caryear] = Interparam2[caryear] + LIB_EU_MFA.ParameterDict['F_In'].Values[i] * \
    #                        (1 - LIB_EU_MFA.ParameterDict['LossRate_1'].Values[i])
    # if caryear > 2:
    #     for o in range(1, caryear):
    #         EV_Mat_Yet[caryear] = EV_Mat_Yet[caryear - 1] + (LIB_EU_MFA.ParameterDict['F_In'].Values[i] *
    #                                                      (1 - LIB_EU_MFA.ParameterDict['LossRate_1'].Values[i]) *
    #                                                      (1 - careolp[caryear - i])) - \
    #                           (LIB_EU_MFA.ParameterDict['F_In'].Values[o] *
    #                            (1 - LIB_EU_MFA.ParameterDict['LossRate_1'].Values[o]))
    # EV_Mat_Yet[caryear] = EV_Mat_Yet[caryear - 1] +  Interparam1[caryear] - Interparam2[caryear] + \
    #                       (LIB_EU_MFA.ParameterDict['F_In'].Values[caryear] *
    #                        (1 - LIB_EU_MFA.ParameterDict['LossRate_1'].Values[caryear]))

LIB_EU_MFA.ParameterDict['Dismantle'].Values = EV_Mat * (1 - LIB_EU_MFA.ParameterDict['LossRate_2'].Values)
LIB_EU_MFA.ParameterDict['Yet_EoL_EV'].Values = EV_Mat_Yet

Cell_D = np.zeros((Model_Duration + 1, Composition_Par_LIB[1] * Composition_Par_LIB[2]))
Cell_B = np.zeros((Model_Duration + 1, Composition_Par_LIB[1] * Composition_Par_LIB[2]))
for year in range(1, Model_Duration + 1):
    for cell in range(Composition_Par_LIB[2]):
        F_cell[year, cell] = eolp[year]
        for i in range(year):
            Mat_B_Dism[year, cell] = Mat_B_Dism[year, cell] + np.sum(
                LIB_EU_MFA.ParameterDict['Dismantle'].Values[i, :]) * \
                                     F_cell[year - i, cell]
            Mat_B_EVYet[year, cell] = Mat_B_EVYet[year, cell] + np.sum(
                LIB_EU_MFA.ParameterDict['Yet_EoL_EV'].Values[i, :]) * \
                                      (1 - F_cell[year - i, cell])

    for cells in range(Composition_Par_LIB[1] * Composition_Par_LIB[2]):
        for o in range(year):
            Cell_D[year, cells] = Cell_D[year, cells] + LIB_EU_MFA.ParameterDict['Dismantle'].Values[o + 1, o] * eolp[
                year - o] * \
                                  Composition_Par_LIB[1] * Composition_Par_LIB[2]
            Cell_B[year, cells] = Cell_B[year, cells] + LIB_EU_MFA.ParameterDict['Yet_EoL_EV'].Values[o, o] * eolp[
                year - o] * \
                                  Composition_Par_LIB[1] * Composition_Par_LIB[2]

    # for module in range(Composition_Par_LIB[1]):
    #     F_module[year, module] = 1 - np.prod(1 - F_cell[year, :])
    # for pack in range(Composition_Par_LIB[0]):
    #     F_pack[year, pack] = 1 - np.prod(1 - F_module[year, :])

#  Flow definitions.
LIB_EU_MFA.FlowDict = {
    'F_0_1': msc.Flow(Name='Market_Loss', ID=0, P_Start=0, P_End=1,
                      Indices='t,c,p,m,e', Values=None, Unit='1'),
    'F_0_2': msc.Flow(Name='Market_to_EV_Dismantle', ID=1, P_Start=0, P_End=2,
                      Indices='t,c,p,m,e', Values=None, Unit='1'),
    'F_2_3': msc.Flow(Name='Dismantle_Loss', ID=2, P_Start=2, P_End=3,
                      Indices='t,c,p,m,e', Values=None, Unit='1'),
    'F_2_5': msc.Flow(Name='Cathode_D', ID=3, P_Start=2, P_End=5,
                      Indices='t,c,p,m,e', Values=None, Unit='1'),
    'F_5_6': msc.Flow(Name='Recycling_D', ID=4, P_Start=5, P_End=6,
                      Indices='t,c,p,m,e', Values=None, Unit='1'),
    'F_0_7': msc.Flow(Name='Cathode_B', ID=5, P_Start=0, P_End=7,
                      Indices='t,c,p,m,e', Values=None, Unit='1'),
    'F_7_8': msc.Flow(Name='Recycling_B', ID=6, P_Start=7, P_End=8,
                      Indices='t,c,p,m,e', Values=None, Unit='1'),
    'F_6_9': msc.Flow(Name='to_Supplier_D', ID=7, P_Start=6, P_End=9,
                      Indices='t,c,p,m,e', Values=None, Unit='1'),
    'F_8_9': msc.Flow(Name='to_Supplier_B', ID=8, P_Start=8, P_End=9,
                      Indices='t,c,p,m,e', Values=None, Unit='1'),
    'F_4_10': msc.Flow(Name='Semi_to_PV', ID=9, P_Start=4, P_End=10,
                       Indices='t,c,p,m,e', Values=None, Unit='1'),
    'F_4_11': msc.Flow(Name='Semi_to_EV', ID=10, P_Start=4, P_End=11,
                       Indices='t,c,p,m,e', Values=None, Unit='1'),
    'F_9_10': msc.Flow(Name='Supplier_to_PV', ID=11, P_Start=9, P_End=10,
                       Indices='t,c,p,m,e', Values=None, Unit='1'),
    'F_9_11': msc.Flow(Name='Supplier_to_EV', ID=12, P_Start=9, P_End=11,
                       Indices='t,c,p,m,e', Values=None, Unit='1'),
    'F_2_4_DP': msc.Flow(Name='To_Semi_DP', ID=13, P_Start=2, P_End=4,
                         Indices='t,c,p,m,e', Values=None, Unit='1'),
    'F_2_4_DM': msc.Flow(Name='To_Semi_DM', ID=14, P_Start=2, P_End=4,
                         Indices='t,c,p,m,e', Values=None, Unit='1'),
    'F_2_4_DC': msc.Flow(Name='To_Semi_DC', ID=15, P_Start=2, P_End=4,
                         Indices='t,c,p,m,e', Values=None, Unit='1'),
    'F_0_4_BP': msc.Flow(Name='To_Semi_BC', ID=16, P_Start=0, P_End=4,
                         Indices='t,c,p,m,e', Values=None, Unit='1'),
    'F_0_4_BM': msc.Flow(Name='To_Semi_BM', ID=17, P_Start=0, P_End=4,
                         Indices='t,c,p,m,e', Values=None, Unit='1'),
    'F_0_4_BC': msc.Flow(Name='To_Semi_BC', ID=18, P_Start=0, P_End=4,
                         Indices='t,c,p,m,e', Values=None, Unit='1')
}

LIB_EU_MFA.FlowDict['F_0_1'].Values = np.ceil(LIB_EU_MFA.ParameterDict['F_In'].Values * \
                                              LIB_EU_MFA.ParameterDict['LossRate_1'].Values)
LIB_EU_MFA.FlowDict['F_0_2'].Values = np.ceil(EV_Mat[:, 0])
LIB_EU_MFA.FlowDict['F_2_3'].Values = np.ceil(LIB_EU_MFA.ParameterDict['Dismantle'].Values * \
                                              LIB_EU_MFA.ParameterDict['LossRate_2'].Values)
LIB_EU_MFA.FlowDict['F_2_4_DP'].Values = None
LIB_EU_MFA.FlowDict['F_0_7'].Values = Cell_B * to_Cathode
LIB_EU_MFA.FlowDict['F_2_5'].Values = Cell_D * to_Cathode
LIB_EU_MFA.FlowDict['F_7_8'].Values = LIB_EU_MFA.FlowDict['F_0_7'].Values * Cathode_B2Rec
LIB_EU_MFA.FlowDict['F_8_9'].Values = LIB_EU_MFA.FlowDict['F_7_8'].Values * Rec_B2Supp
LIB_EU_MFA.FlowDict['F_5_6'].Values = LIB_EU_MFA.FlowDict['F_2_5'].Values * Cathode_D2Rec
LIB_EU_MFA.FlowDict['F_6_9'].Values = LIB_EU_MFA.FlowDict['F_5_6'].Values * Rec_D2Supp
LIB_EU_MFA.FlowDict['F_9_10'].Values = (LIB_EU_MFA.FlowDict['F_6_9'].Values + LIB_EU_MFA.FlowDict['F_8_9'].Values) * \
                                       Supp2PV
LIB_EU_MFA.FlowDict['F_9_11'].Values = (LIB_EU_MFA.FlowDict['F_6_9'].Values + LIB_EU_MFA.FlowDict['F_8_9'].Values) * \
                                       Supp2EV

plt.style.use('seaborn-deep')
fig, ax = plt.subplots()
x = LIB_EU_MFA.IndexTable['Classification']['Time'].Items
ax.plot(x,
        # LIB_EU_MFA.IndexTable['Classification']['Time'].Items,
        # LIB_EU_MFA.FlowDict['F_9_10'].Values[:, 0],
        LIB_EU_MFA.FlowDict['F_9_11'].Values[:, 0],
        color='#fabb3e',
        label='LIB cells for EV'
        )
ax.plot(x,
        LIB_EU_MFA.FlowDict['F_9_10'].Values[:, 0],
        color='#3334FF',
        label='LIB cells for PV')

ax.set_title('LIB cells destination')
ax.set_xlabel('Year')
ax.set_ylabel('LIB Cells/Units')
ax.grid()
ax.legend()

fig1, ax1 = plt.subplots()
width = 0.4
x = LIB_EU_MFA.IndexTable['Classification']['Time'].Items
y1 = LIB_EU_MFA.FlowDict['F_9_10'].Values[:, 0] / (LIB_EU_MFA.ParameterDict['EV_In'].Values[:, 0] *
                                                   Composition_Par_LIB[1] * Composition_Par_LIB[2])
y2 = LIB_EU_MFA.FlowDict['F_9_11'].Values[:, 0] / (LIB_EU_MFA.ParameterDict['EV_In'].Values[:, 0] *
                                                   Composition_Par_LIB[1] * Composition_Par_LIB[2])
x = np.arange(len(x))
ax1.bar(x, y2, width, alpha=0.9, label='Ratio of LIBs go to EV vs Initial EV Input')
ax1.bar(x + width, y1, width, alpha=0.9, label='Ratio of LIBs go to PV vs Initial EV Input')
# plt.show()
# ax1.bar(x,
# LIB_EU_MFA.FlowDict['F_9_11'].Values[:, 0] / (LIB_EU_MFA.ParameterDict['EV_In'].Values[:, 0] *
#                                               Composition_Par_LIB[1] * Composition_Par_LIB[2]),
# width,
# alpha=0.9,
# fc='#fabb3e',
# label='Ratio of LIBs go to EV vs Initial EV Input')

ax1.set_title('LIBs destination')
# ax1.set_xlim(0, 1)
ax1.set_xlabel('Year')
ax1.set_xticks(x + width / 2)
ax1.set_xticklabels(LIB_EU_MFA.IndexTable['Classification']['Time'].Items)
ax1.set_ylabel('Ratio')
ax1.grid()
ax1.legend()
plt.show()

fig2, ax2 = plt.subplots()

x = LIB_EU_MFA.IndexTable['Classification']['Time'].Items
ax2.plot(x,
         # LIB_EU_MFA.IndexTable['Classification']['Time'].Items,
         # LIB_EU_MFA.FlowDict['F_9_10'].Values[:, 0],
         LIB_EU_MFA.FlowDict['F_9_11'].Values[:, 0] * Composition_Par_Cell_Ele[0] * Composition_Par_LIB[1] *
         Composition_Par_LIB[2] * (10 ** -9),
         label='Li'
         )
ax2.plot(x,
         LIB_EU_MFA.FlowDict['F_9_11'].Values[:, 0] * Composition_Par_Cell_Ele[1] * Composition_Par_LIB[1] *
         Composition_Par_LIB[2] * (10 ** -9),
         label='Co'
         )
ax2.plot(x,
         LIB_EU_MFA.FlowDict['F_9_11'].Values[:, 0] * Composition_Par_Cell_Ele[2] * Composition_Par_LIB[1] *
         Composition_Par_LIB[2] * (10 ** -9),
         label='Ni'
         )
ax2.plot(x,
         LIB_EU_MFA.FlowDict['F_9_11'].Values[:, 0] * Composition_Par_Cell_Ele[3] * Composition_Par_LIB[1] *
         Composition_Par_LIB[2] * (10 ** -9),
         label='Mn'
         )

ax2.set_title('Elements go to EV')
ax2.set_xlabel('Year')
ax2.set_ylabel('Mt')
ax2.grid()
ax2.legend()

fig3, ax3 = plt.subplots()

x = LIB_EU_MFA.IndexTable['Classification']['Time'].Items
ax3.plot(x,
         # LIB_EU_MFA.IndexTable['Classification']['Time'].Items,
         # LIB_EU_MFA.FlowDict['F_9_10'].Values[:, 0],
         LIB_EU_MFA.FlowDict['F_9_10'].Values[:, 0] * Composition_Par_Cell_Ele[0] * Composition_Par_LIB[1] *
         Composition_Par_LIB[2] * (10 ** -9),
         label='Li'
         )
ax3.plot(x,
         LIB_EU_MFA.FlowDict['F_9_10'].Values[:, 0] * Composition_Par_Cell_Ele[1] * Composition_Par_LIB[1] *
         Composition_Par_LIB[2] * (10 ** -9),
         label='Co'
         )
ax3.plot(x,
         LIB_EU_MFA.FlowDict['F_9_10'].Values[:, 0] * Composition_Par_Cell_Ele[2] * Composition_Par_LIB[1] *
         Composition_Par_LIB[2] * (10 ** -9),
         label='Ni'
         )
ax3.plot(x,
         LIB_EU_MFA.FlowDict['F_9_10'].Values[:, 0] * Composition_Par_Cell_Ele[3] * Composition_Par_LIB[1] *
         Composition_Par_LIB[2] * (10 ** -9),
         label='Mn'
         )

ax3.set_title('Elements go to PV')
ax3.set_xlabel('Year')
ax3.set_ylabel('Mt')
ax3.grid()
ax3.legend()

end = time.time()
print('\n' + 'Time consumption: ' + str(end - start) + ' s.')
