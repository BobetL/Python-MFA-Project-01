import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go

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

IndexTable = pd.DataFrame({
    'Aspect': ['Time', 'Element'],
    'Description': ['Model aspect "time"', 'Model aspect"Element"'],
    'Dimension': ['Time', 'Element'],
    'Classification': [ModelClassification[Aspect] for Aspect in ['Time', 'Element']],
    'IndexLetter': ['t', 'e']
})

IndexTable.set_index('Aspect', inplace=True)
# print(Model_Time_End)
# print(IndexTable)

Dyn_MFA_System = msc.MFAsystem(
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
Dyn_MFA_System.ProcessList = []
Dyn_MFA_System.ProcessList.append(msc.Process(Name='LIBs in the Market', ID=0, ))
Dyn_MFA_System.ProcessList.append(msc.Process(Name='Spent Batteries Collected by Car Dealer', ID=1))
Dyn_MFA_System.ProcessList.append(msc.Process(Name='Spent Batteries Collected by Dismantle', ID=2))
Dyn_MFA_System.ProcessList.append(msc.Process(Name='Lost Batteries while Using', ID=3))
Dyn_MFA_System.ProcessList.append(msc.Process(Name='Lost Batteries in Dismantle', ID=4))
Dyn_MFA_System.ProcessList.append(msc.Process(Name='Lost Batteries by Car Dealer', ID=5))
Dyn_MFA_System.ProcessList.append(msc.Process(Name='Re-manufacturing', ID=6))
Dyn_MFA_System.ProcessList.append(msc.Process(Name='Repurposing', ID=7))
Dyn_MFA_System.ProcessList.append(msc.Process(Name='Second Use', ID=8))
Dyn_MFA_System.ProcessList.append(msc.Process(Name='Recycling', ID=9))

# Define parameters.
Values_EV_Sales = pd.read_excel(r'C:\Users\Users\Downloads\FlowCatalog.xlsx',
                                sheet_name='Sheet2', usecols='B').values

# print(Values_EV_Sales)
LIBs_Penetration_Rate1 = np.linspace(0.7, 0.8, 5, endpoint=False)
LIBs_Penetration_Rate2 = np.linspace(0.8, 1, 6)
LIBs_Penetration_Rate3 = np.linspace(1, 1, 20)
LIBs_Penetration_Rate = np.concatenate((LIBs_Penetration_Rate1, LIBs_Penetration_Rate2, LIBs_Penetration_Rate3))
LIBs_Penetration_Rate = LIBs_Penetration_Rate.reshape(LIBs_Penetration_Rate.shape[0], 1)
# print(LIBs_Penetration_Rate)
Loss_Using_Rate1 = np.linspace(0.4, 0.1, 26)
Loss_Using_Rate2 = np.linspace(0.1, 0.1, 5)
Loss_Using_Rate1 = np.concatenate((Loss_Using_Rate1, Loss_Using_Rate2))
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
Rep2SecondUse = 1
SecondUse2Rec = 1  # Input value.


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
                      Indices='t,e', Values=None),
    'F_6_0': msc.Flow(Name='Remanu to Market', P_Start=6, P_End=0,
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
    'Mat_B_Market': msc.Parameter(Name='LIBs_Market_yrly', ID=2, P_Res=1,
                                  MetaData=None, Indices='e',
                                  Values=Mat_B, Unit='1'),
    'LIBs2Dismantle': msc.Parameter(Name='Dismantled EVs', ID=3, P_Res=2,
                                    MetaData=None, Indices='e',
                                    Values=LIBs2Dismantle, Unit='1')
})
# print(Mat_B)

# print(Rate_Batts)

# Programming the solutions.

Dyn_MFA_System.FlowDict['F_0_1'].Values = (Dyn_MFA_System.ParameterDict['Mat_B_Market'].Values +
                                           Dyn_MFA_System.FlowDict['F_6_0'].Values) * \
                                          (1 - Dyn_MFA_System.ParameterDict['LossRate_1'].Values)
Dyn_MFA_System.FlowDict['F_0_2'].Values = (Dyn_MFA_System.ParameterDict['Mat_B_Market'].Values +
                                           Dyn_MFA_System.FlowDict['F_6_0'].Values) * \
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
Dyn_MFA_System.FlowDict['F_7_8'].Values = (Dyn_MFA_System.ParameterDict['SecondUse'].Values *
                                           (Dyn_MFA_System.FlowDict['F_1_7'].Values +
                                            Dyn_MFA_System.FlowDict['F_2_7'].Values +
                                            Dyn_MFA_System.FlowDict['F_6_7'].Values)) - \
                                          Dyn_MFA_System.FlowDict['F_7_9'].Values
Dyn_MFA_System.FlowDict['F_8_9'].Values = Dyn_MFA_System.ParameterDict['RecRate_4'].Values * \
                                          Dyn_MFA_System.FlowDict['F_7_8'].Values
Dyn_MFA_System.FlowDict['F_6_0'].Values = Dyn_MFA_System.FlowDict['F_2_6'].Values - \
                                          Dyn_MFA_System.FlowDict['F_6_7'].Values

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

# Draw Sankey Diagram
label = [Dyn_MFA_System.ProcessList[0].Name, Dyn_MFA_System.ProcessList[1].Name, Dyn_MFA_System.ProcessList[2].Name,
         Dyn_MFA_System.ProcessList[3].Name, Dyn_MFA_System.ProcessList[4].Name, Dyn_MFA_System.ProcessList[5].Name,
         Dyn_MFA_System.ProcessList[6].Name, Dyn_MFA_System.ProcessList[7].Name, Dyn_MFA_System.ProcessList[8].Name,
         Dyn_MFA_System.ProcessList[9].Name]

source = [Dyn_MFA_System.ProcessList[0].ID, Dyn_MFA_System.ProcessList[0].ID, Dyn_MFA_System.ProcessList[0].ID,
          Dyn_MFA_System.ProcessList[0].ID, Dyn_MFA_System.ProcessList[1].ID, Dyn_MFA_System.ProcessList[1].ID,
          Dyn_MFA_System.ProcessList[1].ID, Dyn_MFA_System.ProcessList[2].ID, Dyn_MFA_System.ProcessList[2].ID,
          Dyn_MFA_System.ProcessList[2].ID, Dyn_MFA_System.ProcessList[2].ID, Dyn_MFA_System.ProcessList[6].ID,
          Dyn_MFA_System.ProcessList[7].ID, Dyn_MFA_System.ProcessList[7].ID, Dyn_MFA_System.ProcessList[8].ID,
          Dyn_MFA_System.ProcessList[6].ID]

target = [Dyn_MFA_System.ProcessList[0].ID, Dyn_MFA_System.ProcessList[1].ID, Dyn_MFA_System.ProcessList[2].ID,
          Dyn_MFA_System.ProcessList[3].ID, Dyn_MFA_System.ProcessList[5].ID, Dyn_MFA_System.ProcessList[7].ID,
          Dyn_MFA_System.ProcessList[9].ID, Dyn_MFA_System.ProcessList[4].ID, Dyn_MFA_System.ProcessList[6].ID,
          Dyn_MFA_System.ProcessList[7].ID, Dyn_MFA_System.ProcessList[9].ID, Dyn_MFA_System.ProcessList[7].ID,
          Dyn_MFA_System.ProcessList[9].ID, Dyn_MFA_System.ProcessList[8].ID, Dyn_MFA_System.ProcessList[9].ID,
          Dyn_MFA_System.ProcessList[0].ID]

timesee = 2009
timecheck = timesee - 2005
value = [Dyn_MFA_System.StockDict['dS_0'].Values[timecheck], Dyn_MFA_System.FlowDict['F_0_1'].Values[timecheck],
         Dyn_MFA_System.FlowDict['F_0_2'].Values[timecheck], Dyn_MFA_System.FlowDict['F_0_3'].Values[timecheck],
         Dyn_MFA_System.FlowDict['F_1_5'].Values[timecheck], Dyn_MFA_System.FlowDict['F_1_7'].Values[timecheck],
         Dyn_MFA_System.FlowDict['F_1_9'].Values[timecheck], Dyn_MFA_System.FlowDict['F_2_4'].Values[timecheck],
         Dyn_MFA_System.FlowDict['F_2_6'].Values[timecheck], Dyn_MFA_System.FlowDict['F_2_7'].Values[timecheck],
         Dyn_MFA_System.FlowDict['F_2_9'].Values[timecheck], Dyn_MFA_System.FlowDict['F_6_7'].Values[timecheck],
         Dyn_MFA_System.FlowDict['F_7_9'].Values[timecheck], Dyn_MFA_System.FlowDict['F_7_8'].Values[timecheck],
         Dyn_MFA_System.FlowDict['F_8_9'].Values[timecheck], Dyn_MFA_System.FlowDict['F_6_0'].Values[timecheck]]

color_link = ['#d7ffc2', '#b0ffde', '#b0ffde', '#d1d1d1',
              '#d1d1d1', '#ffcc99', '#fc9090',
              '#d1d1d1', '#ffcc99', '#ffcc99', '#fc9090',
              '#ffcc99', '#fc9090', '#ffcc99', '#fc9090', '#b0ffde']
color_node = ['#3af25c', '#fabb3e', '#fabb3e', '#d1d1d1', '#d1d1d1', '#d1d1d1',
              '#3af25c', '#fabb3e', '#fabb3e', '#fc0303']
link = dict(source=source, target=target, value=value, color=color_link)
node = dict(label=label, pad=20, thickness=18, color=color_node)
data = go.Sankey(link=link, node=node)
fig1 = go.Figure(data)
fig1.update_layout(hovermode='x', title='Used LIBs Flow', font=dict(size=20, color='white'), paper_bgcolor='#3b3737')
fig1.show()

# Matplotlib
fig, ax = plt.subplots()
ax.bar(Dyn_MFA_System.IndexTable['Classification']['Time'].Items,
       Dyn_MFA_System.FlowDict['F_7_8'].Values[:, 0])
end = time.time()
print('\n' + 'Time consumption: ' + str(end - start) + ' s.')
plt.show()
