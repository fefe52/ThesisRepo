"********* KTH THESIS PROJECT FEDERICA ARESU **********"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import time
from matplotlib.pyplot import figure
from pathlib import Path

"----- Formula from Forza -----"
def Forza(gt,FS,Sensibility,Gain):
    Val = (FS *1000)/(Sensibility * Gain * 5);
    Val = Val * 9.807;   #To have the force in Newton instead of Kg
    F = pd.DataFrame(gt.iloc[:] * Val)
    return F

"----- Formula Torque -----"
def Torque(F,MA):
    TRQ = pd.DataFrame(F.iloc[:]*MA)
    return TRQ
    


allTRQ_DF = [];
allTRQ_PF = [];
TRQ_PF = 0;
TRQ_DF= 0;

os.chdir(r'C:\Users\fedea\Desktop\HDEMG_CSV\F_test05');

for root, dirs, files in os.walk("C:/Users/fedea/Desktop/HDEMG_CSV/F_test05"):
    for name in files:
        if name.endswith((".csv")):
            gt = pd.read_csv(name, sep=';' , engine ='python');
            gt += np.arange(len(gt.columns))
            gt = gt.drop(gt.columns[0], axis=1)
            
            
            Gain = 100;
            FS = 100;
            Sens = 2;
            F = Forza(gt,FS,Sens,Gain)
            
            ## CHECK FOR MOMENT ARM LENGTH ##
            if (name.find("AN15") != -1):
                MA = 0.2;
                
            if (name.find("AN30") != -1):
                MA = 0.1;
            
            if (name.find("AP0") != -1):
                MA = 0.5;
                
            if (name.find("AP10") != -1):
                MA = 0.4;
            
            
            
            TRQ = Torque(F,MA)
            strPF = "_PF_";
            strDF = "_DF_";
            if (name.find(strPF) != -1):
                TRQ_PF = TRQ;
                allTRQ_PF.append(TRQ_PF)
            if (name.find(strDF) != -1):
                TRQ_DF = TRQ;
                allTRQ_DF.append(TRQ_DF)

            
            




            # "------plot of the Force-------"

            # plt.plot(gt.iloc[:100])
            # plt.title("original data from AUX Input")
            # plt.show()
            
            # pd.set_option('precision',10)
            # figure(1)
            # plt.plot(F.iloc[:100])
            # plt.title("Force")

            # figure(2)
            # plt.plot(TRQ.iloc[:100])
            # plt.title("Torque")
            # plt.show()

      