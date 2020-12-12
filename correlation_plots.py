"MASTER THESIS FEDERICA ARESU"
"CORRELATION PLOTS"
from differentiation import *
from features import *
from groundtruth import out_pre_seq
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
from matplotlib.pyplot import figure
import time
import seaborn as sns

"------------IMPORT AND MAIN PART-----------"
CWD = os.getcwd()
subfolders = r"data/Tests/test06"
finalpath = os.path.join(CWD, subfolders)
os.chdir(finalpath);

Fsample = 2048;


#       the dataframe has the time in x-axis and channels in y-axis         #
for root, dirs, files in os.walk(finalpath):
    for name in files:
        if name.endswith((".csv")):
            df = pd.read_csv(name, sep=';' , engine ='python');
            df += np.arange(len(df.columns))
            df = df.drop(df.columns[0], axis=1)  #delete first channel with ramp of values
            n_channels = len(df.columns);
            t = np.arange(len(df.index))/Fsample
            first_row = df.iloc[0,:];  #32 elements
            first_column = df.iloc[:,0];  #250337 elements
            
            "---- divide file per muscle ----"
            df_gl = pd.DataFrame(df.iloc[:,0:32]);
            gl_channels = len(df_gl.columns);
            
            df_p = pd.DataFrame(df.iloc[:,32:64]);
            p_channels = len(df_p.columns);
            
            df_gm = pd.DataFrame(df.iloc[:,64:128]);
            gm_channels = len(df_gm.columns);
            
            df_ta = pd.DataFrame(df.iloc[:,128:192]);
            ta_channels = len(df_ta.columns);
            
            df_sol = pd.DataFrame(df.iloc[:,192:256]);
            sol_channels = len(df_sol.columns);
            
            
            # # "------- Plot of a single channel versus time -----"
            
            # figure(num=4, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
        
            # plt.plot(t,df_gl[:,33],linewidth=0.5)  #plotting only gastrocnemio lateralis
            # plt.title('single channel of data')
            # plt.xlabel('time')
            # plt.ylabel('value')
            # plt.show()

           

            diff_gl_sEMG = Differential_sEMG(df_gl);
            diff_p_sEMG = Differential_sEMG(df_p);
            diff_gm_sEMG = Differential_sEMG(df_gm);
            diff_ta_sEMG = Differential_sEMG(df_ta);
            diff_sol_sEMG = Differential_sol_sEMG(df_sol);

            
                  
            # gl_diffchannels = diff_gl.shape[0];
            # p_diffchannels = diff_p.shape[0];
            # gm_diffchannels = diff_gm.shape[0];
            # ta_diffchannels = diff_ta.shape[0];
            # sol_diffchannels = diff_sol.shape[0];

           
            
         
            
            # "-------Features-------"

            # "Gastrocnemius Lateralis and Peroneus Longus with 32 channels,"
            # "and Gastrocnemius Medialis, Tibialis Anterior and Soleus with 64 channels"
            
            " Gastrocnemium lateralis and Peroneus Longus "
            
            "for regular sEMG"
            MAVgl_channels_sEMG = []
            MAVp_channels_sEMG = []
            MAVgl_channels_sEMG.append(fMAV(diff_gl_sEMG[:]))
            MAVp_channels_sEMG.append(fMAV(diff_p_sEMG[:]))
            
            " Gastrocnemium medialis, Tibialis Anterior and Soleus "
           

            "for regular sEMG"     
            MAVgm_channels_sEMG = []
            MAVta_channels_sEMG =[]
            MAVsol_channels_sEMG = []
            MAVgm_channels_sEMG.append(fMAV(diff_gm_sEMG[:]))
            MAVta_channels_sEMG.append(fMAV(diff_ta_sEMG[:]))
            MAVsol_channels_sEMG.append(fMAV(diff_sol_sEMG[:]))
            
            "first trial"
            MAV_channels_sEMG = np.array(MAV_channels_sEMG);
            
     