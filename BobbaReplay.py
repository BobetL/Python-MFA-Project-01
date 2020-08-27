import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
# from scipy import stats

import ODYM_Classes as msc

start = time.time()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

ModelClassification = {
    'Time': msc.Classification(Name='Time',
                               Dimension='Time',
                               ID=1,
                               Items=list(np.arange(2005, 2036))),

    'Element': msc.Classification(Name='Elements',
                                  Dimension='Element',
                                  ID=2,
                                  Items=['LIB'])
}

Model_Time_Start = int(min(ModelClassification['Time'].Items))
Model_Time_End = int(max(ModelClassification['Time'].Items))
Model_Duration = Model_Time_End - Model_Time_Start

IndexTable = pd.DataFrame({'Aspect': ['Time', 'Element'],
                           'Description': ['Model aspect "time"', 'Model aspect"Element"'],
                           'Dimension': ['Time', 'Element'],
                           'Classification': [ModelClassification[Aspect] for Aspect in ['Time', 'Element']],
                           'IndexLetter': ['t', 'e']})

IndexTable.set_index('Aspect', inplace=True)
# print(Model_Time_End)
# print(IndexTable)

Dyn_MFA_System = msc.MFAsystem(Name='UsedLIBsFlow',
                               Geogr_Scope='Europe',
                               Unit='',
                               ProcessList=[],
                               FlowDict={},
                               StockDict={},
                               ParameterDict={},
                               Time_Start=Model_Time_Start,
                               Time_End=Model_Time_End,
                               IndexTable=IndexTable,
                               Elements=IndexTable.loc['Element'].Classification.Items)

# Processes.
Dyn_MFA_System.ProcessList = []
Dyn_MFA_System.ProcessList.append(msc.Process(Name='Market', ID=0))
Dyn_MFA_System.ProcessList.append(msc.Process(Name='SpentCollected_CarDealer', ID=1))
Dyn_MFA_System.ProcessList.append(msc.Process(Name='SpentCollected_Dismantle', ID=2))
Dyn_MFA_System.ProcessList.append(msc.Process(Name='Loss_Using', ID=3))
Dyn_MFA_System.ProcessList.append(msc.Process(Name='Loss_Dismantle', ID=4))
Dyn_MFA_System.ProcessList.append(msc.Process(Name='Loss_CarDealer', ID=5))
Dyn_MFA_System.ProcessList.append(msc.Process(Name='Remanufacturing', ID=6))
Dyn_MFA_System.ProcessList.append(msc.Process(Name='Repurposing', ID=7))
Dyn_MFA_System.ProcessList.append(msc.Process(Name='SecondUse', ID=8))
Dyn_MFA_System.ProcessList.append(msc.Process(Name='Recycling', ID=9))

# Define parameters.
Values_EV_Sales = pd.read_excel(r'C:\Users\tieae\Downloads\研究相关\Matlab Research Project\MFA Model\FlowCatalog.xlsx',
                                sheet_name='Sheet2', usecols='B').values

# r'C:\Users\tieae\Downloads\研究相关\Matlab Research Project\MFA Model\FlowCatalog.xlsx',
#                                 sheet_name='Sheet2', usecols='B'
# r'/Users/zhoujingwei/OneDrive/文档/Matlab Research Project/FlowCatalog.xlsx',
#                                 sheet_name='Sheet2', usecols='B'

# print(Values_EV_Sales)
LIBs_Penetration_Rate1 = np.linspace(0.7, 0.8, 5, endpoint=False)
LIBs_Penetration_Rate2 = np.linspace(0.8, 1, 6)
LIBs_Penetration_Rate3 = np.linspace(1, 1, 20)
LIBs_Penetration_Rate = np.concatenate((LIBs_Penetration_Rate1, LIBs_Penetration_Rate2, LIBs_Penetration_Rate3))
LIBs_Penetration_Rate = LIBs_Penetration_Rate.reshape(LIBs_Penetration_Rate.shape[0], 1)
# print(LIBs_Penetration_Rate)
Loss_Using_Rate1 = np.linspace(0.4, 0.1, Model_Duration + 1)
# Loss_Using_Rate2 = np.linspace(0.1, 0.1, 0)
Loss_Using_Rate = Loss_Using_Rate1.reshape(Loss_Using_Rate1.shape[0], 1)
# print(Loss_Using_Rate)
LIBs_in_Market_Rate = LIBs_Penetration_Rate * (1 - Loss_Using_Rate)
Loss_Dismantle_Rate = 0.1
Loss_CarDealer_Rate = 0
Dismantle2Rem_Rate = 0.2
Dismantle2Rep_Rate = 0.7
Dismantle2Rec_Rate = 1 - Loss_Dismantle_Rate - Dismantle2Rem_Rate - Dismantle2Rep_Rate
Rem2Rep_Rate = 0.2
CarDealer2Rep_Rate = 1
CarDealer2Rec_Rate = 1 - CarDealer2Rep_Rate - Loss_CarDealer_Rate
Rep2Rec = 0.1  # 0.1 of the amount from Dismantle2Rep.
Rep2SecondUse = 1 - Rep2Rec
SecondUse2Rec = 1  # Input value.
EV_LT = 10  # 10 years lifetime of an EV.

# Lifetime Distribution (Uniform distribution) simulated by Queueing Problem.

# InstallingTime = np.zeros(1)
# for i in range(Model_Duration + 1):
#     InstallingTime = np.random.uniform(Model_Time_Start, Model_Time_End)
#     InstallingTime.sort()
# lifetime = (6, 8, 10, 12)
# prob = (0.1, 0.4, 0.4, 0.1)
# WorkingTime = stats.rv_discrete(values=(lifetime, prob))
# print(InstallingTime)
# print(WorkingTime.rvs(size=1000))
# StartingTime = [0 for i in range(Model_Duration + 1)]
# FinishTime = [0 for i in range(Model_Duration + 1)]
# print(FinishTime[0], StartingTime[0], WorkingTime.rvs(size=100)[0])
# The first generation LIBs.
# FinishTime[0] = StartingTime[0] + WorkingTime.rvs(size=Values_EV_Sales[0])[0]

# After the first generation.

# for i in range(1, len(Values_EV_Sales)):
#     if FinishTime[i - 1] > StartingTime[i]:
#         StartingTime[i] = FinishTime[i - 1]
#     FinishTime[i] = StartingTime[i] + WorkingTime.rvs(size=Values_EV_Sales[i])[i]
# print(FinishTime, StartingTime)

# for i in range(Model_Duration + 1):
#     if FinishTime[i] - StartingTime[i] == 6:
#         LIBs2CarDealer[i] = 0.1
#     elif FinishTime[i] - StartingTime[i] == 8:
#         LIBs2CarDealer[i] = 0.4
#     elif FinishTime[i] - StartingTime[i] == 10:
#         LIBs2Dismantle[i] = 0.4
#     elif FinishTime[i] - StartingTime[i] == 12:
#         LIBs2CarDealer[i] = 0.1


# print(LIBs2CarDealer, LIBs2Dismantle, LIBs_in_Market_Rate)
# print(Values_EV_Sales, LIBs_Penetration_Rate)

ParameterDict = {
    'F_In': msc.Parameter(Name='Inflow_LIBs', ID=1, P_Res=0,
                          MetaData=None, Indices='te',
                          Values=np.einsum('...t,...e->...t', Values_EV_Sales, LIBs_Penetration_Rate),
                          Unit='Unit'),

    'LossRate_1': msc.Parameter(Name='LossUsing', ID=4, P_Res=3,
                                MetaData=None, Indices='e',
                                Values=Loss_Using_Rate, Unit='1'),
    'LossRate_2': msc.Parameter(Name='LossDismantle', ID=5, P_Res=4,
                                MetaData=None, Indices='e',
                                Values=Loss_Dismantle_Rate, Unit='1'),
    'LossRate_3': msc.Parameter(Name='LossCarDealer', ID=6, P_Res=5,
                                MetaData=None, Indices='e',
                                Values=Loss_CarDealer_Rate, Unit='1'),
    'RemRate': msc.Parameter(Name='RemufacturingRate', ID=7, P_Res=6,
                             MetaData=None, Indices='e',
                             Values=Dismantle2Rem_Rate, Unit='1'),
    'RepRate_1': msc.Parameter(Name='CarDealer2Rep', ID=8, P_Res=7,
                               MetaData=None, Indices='e',
                               Values=CarDealer2Rep_Rate, Unit='1'),
    'RepRate_2': msc.Parameter(Name='Rem2Rep', ID=9, P_Res=7,
                               MetaData=None, Indices='e',
                               Values=Rem2Rep_Rate, Unit='1'),
    'RepRate_3': msc.Parameter(Name='Dismantle2Rep', ID=10, P_Res=7,
                               MetaData=None, Indices='e',
                               Values=Dismantle2Rep_Rate, Unit='1'),
    'SecondUse': msc.Parameter(Name='SecondUseRate', ID=11, P_Res=8,
                               MetaData=None, Indices='e',
                               Values=Rep2SecondUse, Unit='1'),
    'RecRate_1': msc.Parameter(Name='Rep2Rec', ID=12, P_Res=9,
                               MetaData=None, Indices='e',
                               Values=Rep2Rec, Unit='1'),
    'RecRate_2': msc.Parameter(Name='CarDealer2Rec', ID=13, P_Res=9,
                               MetaData=None, Indices='e',
                               Values=CarDealer2Rec_Rate, Unit='1'),
    'RecRate_3': msc.Parameter(Name='Dismantle2Rec', ID=14, P_Res=9,
                               MetaData=None, Indices='e',
                               Values=Dismantle2Rec_Rate, Unit='1'),
    'RecRate_4': msc.Parameter(Name='SecondUse2Rec', ID=15, P_Res=9,
                               MetaData=None, Indices='e',
                               Values=SecondUse2Rec, Unit='1')  # User's input.
}

Dyn_MFA_System.ParameterDict = ParameterDict
# print(ParameterDict['RecRate_3'].Values)
# print(Dyn_MFA_System.Consistency_Check())
Dyn_MFA_System.FlowDict = {
    'F_0_1': msc.Flow(Name='LIBs from the Market to car dealer', P_Start=0, P_End=1,
                      Indices='t,e', Values=None),
    'F_0_2': msc.Flow(Name='LIBs from the Market to dismantle', P_Start=0, P_End=2,
                      Indices='t,e', Values=None),
    'F_0_3': msc.Flow(Name='LIBs loss while first using', P_Start=0, P_End=3,
                      Indices='t,e', Values=None),
    'F_1_5': msc.Flow(Name='Car dealer loss', P_Start=1, P_End=5,
                      Indices='t,e', Values=None),
    'F_1_7': msc.Flow(Name='Car dealer to repurposing', P_Start=1, P_End=7,
                      Indices='t,e', Values=None),
    'F_1_9': msc.Flow(Name='Car dealer to Recycling', P_Start=1, P_End=9,
                      Indices='t,e', Values=None),
    'F_2_4': msc.Flow(Name='Dismantle loss', P_Start=2, P_End=4,
                      Indices='t,e', Values=None),
    'F_2_6': msc.Flow(Name='LIBs to Remanufacturing', P_Start=2, P_End=6,
                      Indices='t,e', Values=None),
    'F_2_7': msc.Flow(Name='Dismantle to Repurposing', P_Start=2, P_End=7,
                      Indices='t,e', Values=None),
    'F_2_9': msc.Flow(Name='Dismantle to Recycling', P_Start=2, P_End=9,
                      Indices='t,e', Values=None),
    'F_6_7': msc.Flow(Name='Remanu to Repurp', P_Start=6, P_End=7,
                      Indices='t,e', Values=None),
    'F_7_8': msc.Flow(Name='Repurp to second use', P_Start=7, P_End=8,
                      Indices='t,e', Values=None),
    'F_7_9': msc.Flow(Name='Repurp to Recyc', P_Start=7, P_End=9,
                      Indices='t,e', Values=None),
    'F_8_9': msc.Flow(Name='Second use to Recyc', P_Start=8, P_End=9,
                      Indices='t,e', Values=None)
}

Dyn_MFA_System.StockDict = {
    'S_0': msc.Stock(Name='Market', P_Res=0, Type=0,
                     Indices='t,e', Values=None),
    'dS_0': msc.Stock(Name='Market change', P_Res=0, Type=1,
                      Indices='t,e', Values=None),
    'S_8': msc.Stock(Name='Second use', P_Res=8, Type=0,
                     Indices='t,e', Values=None),
    'dS_8': msc.Stock(Name='Second use change', P_Res=8, Type=1,
                      Indices='t,e', Values=None)
}

Dyn_MFA_System.Initialize_FlowValues()
# print(Dyn_MFA_System.ParameterDict['F_In'].Values)
# print(Dyn_MFA_System.Consistency_Check())

# Intermediate computations.
# LIBs_Market = np.zeros((Model_Duration + 1, Model_Duration + 1))
# LIBs_Market[0, 0] = Dyn_MFA_System.ParameterDict['F_In'].Values[0, 0]
# # print(LIBs_Market)
# for r in range(Model_Duration + 1):
#     for c in range(r, Model_Duration + 1):
#         LIBs_Market[r, c] = Dyn_MFA_System.ParameterDict['F_In'].Values[c - r, 0]
# data = pd.DataFrame(LIBs_Market)
# writer = pd.ExcelWriter('Flow_Data.xlsx')
# data.to_excel(writer, 'Sheet_1', float_format='%.5f')
# # writer.save()
# Dyn_MFA_System.ParameterDict.update({
#     'Mat_B_Market': msc.Parameter(Name='LIBs_Market_yrly', ID=16, P_Res=0,
#                                   MetaData=None, Indices='e',
#                                   Values=LIBs_Market, Unit='1')
# })

# LIBs2CarDealer = np.array([0.1, 0.4, 0.4, 0.1])
# LIBs2Dismantle = np.array([0.1, 0.4, 0.4, 0.1])
# LIBs2CarDealer = LIBs2CarDealer.reshape(4, 1)
# LIBs2Dismantle = LIBs2Dismantle.reshape(4, 1)

# for c in range(Model_Duration + 1):
#     for r in range(0, c + 1):
#         # while LIBs_Market[r, c] > 0:
#         if c - r == 6:
#             LIBs2CarDealer[r, c] = 0.1
#         elif c - r == 8:
#             LIBs2CarDealer[r, c] = 0.4
#         elif c - r == 10:
#             LIBs2Dismantle[r, c] = 0.4
#         elif c - r == 12:
#             LIBs2CarDealer[r, c] = 0.1
#
# print(LIBs2CarDealer, LIBs2Dismantle)
# Dyn_MFA_System.ParameterDict.update({
#     'LIBs2CarDealer': msc.Parameter(Name='LIBs2CarDealer', ID=2, P_Res=1,
#                                     MetaData=None, Indices='e',
#                                     Values=LIBs2CarDealer, Unit='1'),
#     'LIBs2Dismantle': msc.Parameter(Name='LIBs2Dismantle', ID=3, P_Res=2,
#                                     MetaData=None, Indices='e',
#                                     Values=LIBs2Dismantle, Unit='1'),
# })
# print(np.dot(Dyn_MFA_System.ParameterDict['Mat_B_Market'].Values[3, :],
#              Dyn_MFA_System.ParameterDict['LIBs2CarDealer'].Values[3, :]))
#
# Batts = np.zeros((Model_Duration + 1, 1))
# print(Dyn_MFA_System.ParameterDict['Mat_B_Market'].Values.shape,
#       Dyn_MFA_System.ParameterDict['LIBs2CarDealer'].Values.shape,
#       Dyn_MFA_System.ParameterDict['LossRate_1'].Values.shape)
# for r in range(Model_Duration + 1):
#     Batts[r, :] = np.dot(Dyn_MFA_System.ParameterDict['LossRate_1'].Values[r, :],
#                          np.dot(Dyn_MFA_System.ParameterDict['Mat_B_Market'].Values[r, :],
#                                 Dyn_MFA_System.ParameterDict['LIBs2CarDealer'].Values[r, :]))
#
# print(Batts)

Rate_Batts = np.zeros((Model_Duration + 1, 2))
LIBs2Dismantle = np.zeros((Model_Duration + 1, 1))
for t in range(Model_Duration + 1):
    if t >= 5:
        Rate_Batts[t, 0] = 0.1
    if t >= 7:
        Rate_Batts[t, 1] = 0.4
    if t >= 9:
        LIBs2Dismantle[t] = 0.4

Mat_B = np.zeros((Model_Duration + 1, 1))
Mat_B_D = np.zeros((Model_Duration + 1, 1))
for t in range(Model_Duration + 1):
    if t < 5:
        Mat_B[t] = 0
    if 7 > t >= 5:
        Mat_B[t] = Dyn_MFA_System.ParameterDict['F_In'].Values[t - 5] * Rate_Batts[t, 0]
    if t >= 7:
        Mat_B[t] = Dyn_MFA_System.ParameterDict['F_In'].Values[t - 5] * Rate_Batts[t, 0] + \
                   Dyn_MFA_System.ParameterDict['F_In'].Values[t - 7] * Rate_Batts[t, 1]
    if t >= 9:
        Mat_B_D[t] = Dyn_MFA_System.ParameterDict['F_In'].Values[t - 9] * LIBs2Dismantle[t]

Dyn_MFA_System.ParameterDict.update({
    'Mat_B_Market': msc.Parameter(Name='LIBs_Market_yrly', ID=16, P_Res=1,
                                  MetaData=None, Indices='e',
                                  Values=Mat_B, Unit='1'),
    'LIBs2Dismantle': msc.Parameter(Name='Dismantled EVs', ID=3, P_Res=2,
                                    MetaData=None, Indices='e',
                                    Values=LIBs2Dismantle, Unit='1')
})
# print(Mat_B)

# print(Rate_Batts)

# Programming the solutions.

Dyn_MFA_System.FlowDict['F_0_1'].Values = Dyn_MFA_System.ParameterDict['Mat_B_Market'].Values * \
                                          (1 - Dyn_MFA_System.ParameterDict['LossRate_1'].Values)
Dyn_MFA_System.FlowDict['F_0_2'].Values = Dyn_MFA_System.ParameterDict['Mat_B_Market'].Values * \
                                          (1 - Dyn_MFA_System.ParameterDict['LossRate_1'].Values) * \
                                          Dyn_MFA_System.ParameterDict['LIBs2Dismantle'].Values
Dyn_MFA_System.FlowDict['F_0_3'].Values = Dyn_MFA_System.ParameterDict['F_In'].Values * \
                                          Dyn_MFA_System.ParameterDict['LossRate_1'].Values
Dyn_MFA_System.FlowDict['F_1_5'].Values = Dyn_MFA_System.ParameterDict['LossRate_3'].Values * \
                                          Dyn_MFA_System.FlowDict['F_0_1'].Values
Dyn_MFA_System.FlowDict['F_1_7'].Values = Dyn_MFA_System.ParameterDict['RepRate_1'].Values * \
                                          Dyn_MFA_System.FlowDict['F_0_1'].Values
Dyn_MFA_System.FlowDict['F_1_9'].Values = Dyn_MFA_System.ParameterDict['RecRate_2'].Values * \
                                          Dyn_MFA_System.FlowDict['F_0_1'].Values
Dyn_MFA_System.FlowDict['F_2_4'].Values = Dyn_MFA_System.ParameterDict['LossRate_2'].Values * \
                                          Dyn_MFA_System.FlowDict['F_0_2'].Values
Dyn_MFA_System.FlowDict['F_2_6'].Values = Dyn_MFA_System.ParameterDict['RemRate'].Values * \
                                          Dyn_MFA_System.FlowDict['F_0_2'].Values
Dyn_MFA_System.FlowDict['F_2_7'].Values = Dyn_MFA_System.ParameterDict['RepRate_3'].Values * \
                                          Dyn_MFA_System.FlowDict['F_0_2'].Values
Dyn_MFA_System.FlowDict['F_2_9'].Values = Dyn_MFA_System.ParameterDict['RecRate_3'].Values * \
                                          Dyn_MFA_System.FlowDict['F_0_2'].Values
Dyn_MFA_System.FlowDict['F_6_7'].Values = Dyn_MFA_System.ParameterDict['RepRate_2'].Values * \
                                          Dyn_MFA_System.FlowDict['F_2_6'].Values
Dyn_MFA_System.FlowDict['F_7_9'].Values = Dyn_MFA_System.ParameterDict['RecRate_1'].Values * \
                                          Dyn_MFA_System.FlowDict['F_2_7'].Values
Dyn_MFA_System.FlowDict['F_7_8'].Values = Dyn_MFA_System.ParameterDict['SecondUse'].Values * \
                                          (Dyn_MFA_System.FlowDict['F_1_7'].Values +
                                           Dyn_MFA_System.FlowDict['F_2_7'].Values +
                                           Dyn_MFA_System.FlowDict['F_6_7'].Values)
Dyn_MFA_System.FlowDict['F_8_9'].Values = Dyn_MFA_System.ParameterDict['RecRate_4'].Values * \
                                          Dyn_MFA_System.FlowDict['F_7_8'].Values

Dyn_MFA_System.StockDict['dS_0'].Values = Dyn_MFA_System.ParameterDict['F_In'].Values - \
                                          Dyn_MFA_System.FlowDict['F_0_1'].Values - \
                                          Dyn_MFA_System.FlowDict['F_0_2'].Values - \
                                          Dyn_MFA_System.FlowDict['F_0_3'].Values
Dyn_MFA_System.StockDict['dS_8'].Values = Dyn_MFA_System.FlowDict['F_7_8'].Values - \
                                          Dyn_MFA_System.FlowDict['F_8_9'].Values
Dyn_MFA_System.StockDict['S_0'].Values = Dyn_MFA_System.StockDict['dS_0'].Values.cumsum(axis=0)
Dyn_MFA_System.StockDict['S_8'].Values = Dyn_MFA_System.StockDict['dS_8'].Values.cumsum(axis=0)

# print(Dyn_MFA_System.FlowDict['F_7_8'].Values)
# Mass Balance
Bal = Dyn_MFA_System.MassBalance()
# print(Bal.shape)
# print(np.abs(Bal).sum(axis=0).sum(axis=1))

fig, ax = plt.subplots()
ax.bar(Dyn_MFA_System.IndexTable['Classification']['Time'].Items,
       Dyn_MFA_System.FlowDict['F_7_8'].Values[:, 0])
end = time.time()
print('\n' + 'Time consumption: ' + str(end - start) + ' s.')
plt.show()
