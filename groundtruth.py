"********* KTH THESIS PROJECT FEDERICA ARESU **********"

import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import time
from matplotlib.pyplot import figure
from pathlib import Path



"----- Formula from Forza -----"
def Forza(gt,FS,Sensibility,Gain):
    Val = FS/(Sensibility * Gain * 5);
    Val = Val * 9.807;   #To have the force in Newton instead of Kg
    F = pd.DataFrame(gt.iloc[:] * Val)
    return F

"----- Formula Torque -----"
def Torque(F,MA):
    n = len(F);
    TRQ = []
    TRQ_VAL = np.zeros([len(F),1]);
    TRQ_VAL[:] = F.iloc[:]*MA
    len_window = 1024;
    for i in range(0,n,512):
        if n in range(i,(i+len_window)):
            break
        z = TRQ_VAL[i:(i+len_window)]; 
        TRQ.append(np.sum(abs(z))/len(z))
    TRQ = np.array(TRQ);
    return TRQ
    

allTRQ_DF = [];
allTRQ_PF = [];
TRQ_PF = 0;
TRQ_DF= 0;

CWD = os.getcwd()
subfolders = r"data/F_test05"
finalpath_groundtruth = os.path.join(CWD, subfolders)
os.chdir(finalpath_groundtruth);

for root, dirs, files in os.walk(finalpath_groundtruth):
    for name in files:   
        if name.endswith((".csv")):
            gt = pd.read_csv(name, sep=';' , engine ='python');
            gt += np.arange(len(gt.columns))
            gt = gt.drop(gt.columns[0], axis=1)
            
            
            Gain = 100;
            FS = 100;
            Sens = 0.002;
            F = Forza(gt,FS,Sens,Gain)
            MA = 10.5;                   #measured value from lab
            
            TRQ = Torque(F,MA)
            out_pre_seq = TRQ.reshape((len(TRQ), 1))
            print("size of out_seq", len(out_pre_seq))
            # strPF = "_PF_";
            # strDF = "_DF_";
            # if (name.find(strPF) != -1):
            #     TRQ_PF = TRQ;
            #     allTRQ_PF.append(TRQ_PF)
            # if (name.find(strDF) != -1):
            #     TRQ_DF = TRQ;
            #     allTRQ_DF.append(TRQ_DF)

            
            




            # "------plot of the Force-------"

            #plt.plot(gt.iloc[:100])
            #plt.title("original data from AUX Input")
            #plt.show()
            
            #pd.set_option('precision',10)
            figure(1)
            plt.plot(F.iloc[:100])
            #plt.title("Force")
            plt.savefig(CWD + '/figures/Force.png')
            plt.show()

            # figure(2)
            # plt.plot(TRQ.iloc[:100])
            # plt.title("Torque")
            # plt.show()

      
