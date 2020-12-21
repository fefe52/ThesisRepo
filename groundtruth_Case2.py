"********* KTH THESIS PROJECT FEDERICA ARESU **********"
""" STUDY CASE 2 """
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
def Forza(gt,FS,Sensibility,Gain,offset):
    Val = FS/(Sensibility * Gain * 5);
    #Val = 1;
    Val = Val * 9.807;   #To have the force in Newton instead of Kg    ## take out the offset
    F = pd.DataFrame((gt.iloc[:] - offset) * Val)
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
    return TRQ
    

allTRQ_DF = [];
TRQ_all = [];

CWD = os.getcwd()
subfolders = r"data/F_tests"

finalpath_groundtruth = os.path.join(CWD, subfolders)
#print(os.walk(finalpath_groundtruth))

os.chdir(finalpath_groundtruth);
for root, dirs, files in os.walk(finalpath_groundtruth):
    if (root == finalpath_groundtruth):
        continue
    
    
    
    
    if (root.endswith('02')):
        offset = 2.328;
    if (root.endswith('05')):
        offset = 2.257;
        
    if (root.endswith('06')):
        offset = 2.383;  
    
    if (root.endswith('07')):
        continue
    if (root.endswith('10')):
        continue
        
        
    os.chdir(root);
    for name in files:
        if name.endswith((".csv")):
            
            if ("AN30" in name):
                continue
            if("AP10" in name):
                continue
            if("AP0" in name):
                continue
            print("forcefile", name)
            gt = pd.read_csv(name, sep=';' , engine ='python');
            #gt += np.arange(len(gt.columns))
            gt = gt.drop(gt.columns[0], axis=1)
            
            

            Gain = 100;
            FS = 100;
            Sens = 0.002;
            F = Forza(gt,FS,Sens,Gain,offset)
            MA = 0.105;                   #measured value from lab in m
            
            TRQ = Torque(F,MA)
            if (len(TRQ) > 190):
                del TRQ[190:len(TRQ)]
            TRQ_all = TRQ_all + TRQ;
            
               
            
            
TRQ_all = np.array(TRQ_all);
out_pre_seq = TRQ_all.reshape((len(TRQ_all), 1))
print("size of out_seq", len(out_pre_seq))
            
"------plot of the Force-------"
figure() 
plt.plot(TRQ_all)
plt.title("torque all")
plt.ylabel("N*m")
# plt.savefig(CWD + "/figures/Force.png")
plt.show()
            
            # #pd.set_option('precision',10)
            # #"------plot of the Torque------"
            # figure()
            # plt.plot(TRQ)
            # plt.title("Torque in Nm")
            # plt.savefig(CWD + '/figures/Torque.png')
            # plt.show()

            # figure(2)
            # plt.plot(TRQ.iloc[:100])
            # plt.title("Torque")
            # plt.show()

      

