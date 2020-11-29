"********* KTH THESIS PROJECT FEDERICA ARESU **********"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import time
from matplotlib.pyplot import figure
from pathlib import Path

from differentiation import *
from features import *
#from groundtruth import *


"------------IMPORT AND MAIN PART-----------"
CWD = os.getcwd()
subfolders = r"data/test05"
finalpath = os.path.join(CWD, subfolders)
os.chdir(finalpath);

Fsample = 2048;

MAV_muscle = np.zeros(5);
WL_muscle = np.zeros(5);
WAMP_muscle = np.zeros(5);

allFTR_PF = [];
allFTR_DF = [];


#       the dataframe has the time in x-axis and channels in y-axis         #
for root, dirs, files in os.walk(finalpath):
    for name in files:
        if name.endswith((".csv")):
            df = pd.read_csv(name, sep=';' , engine ='python');
            #df += np.arange(len(df.columns))
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
            
            diff_gl = Differential(df_gl,gl_channels);
            diff_p = Differential(df_p,p_channels);
            diff_gm = Differential(df_gm,gm_channels);
            diff_ta = Differential(df_ta,ta_channels);
            diff_sol = Differential_sol(df_sol,sol_channels);
            diff_gl.to_csv(CWD + '/figures/diff_gl_data.csv')
            diff_p.to_csv(CWD + '/figures/diff_p_data.csv')
            diff_gm.to_csv(CWD + '/figures/diff_gm_data.csv')
            diff_ta.to_csv(CWD + '/figures/diff_ta_data.csv')
            diff_sol.to_csv(CWD + '/figures/diff_sol_data.csv')
            
            
            gl_diffchannels = diff_gl.shape[0];
            p_diffchannels = diff_p.shape[0];
            gm_diffchannels = diff_gm.shape[0];
            ta_diffchannels = diff_ta.shape[0];
            sol_diffchannels = diff_sol.shape[0];
            
            
        
           
            
            
            
            "-------Features-------"

            "Gastrocnemius Lateralis and Peroneus Longus with 32 channels,"
            "and Gastrocnemius Medialis, Tibialis Anterior and Soleus with 64 channels"
            
            # " Gastrocnemium lateralis and Peroneus Longus "
            # MAVgl_channels = []
            # MAVp_channels = []
            # for c in range(gl_diffchannels):
            #     MAVgl_channels.append(fMAV(diff_gl[c,:]))
            #     MAVp_channels.append(fMAV(diff_p[c,:]))
            
        
            
            
            # " Gastrocnemium medialis, Tibialis Anterior and Soleus "
            # MAVgm_channels = []
            # MAVta_channels =[]
            # MAVsol_channels = []
            # for b in range(gm_diffchannels):
            #     MAVgm_channels.append(fMAV(diff_gm[b,:]))
            #     MAVta_channels.append(fMAV(diff_ta[b,:]))
            #     MAVsol_channels.append(fMAV(diff_sol[b,:]))
            
          
            # "All muscles features "
            # MAV_channels = []
            # MAV_channels = MAVgl_channels + MAVp_channels + MAVgm_channels + MAVta_channels + MAVsol_channels;
            # MAV_channels = np.array(MAV_channels);
            # MAV_channels = np.transpose(MAV_channels);
            # print("MAV_channels",MAV_channels.shape)
            
            
            

            "------ plot MAV profile ------" #only gastrocnemio lateralis
            # plots= diff_gl.shape[0]
            # figure(num=5 , figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
            # for i in range(len(MAVgl_channels)):
            #     ax = plt.subplot(plots,1,i + 1)
            #     plt.plot(MAVgl_channels[i],color='blue',linewidth=0.5)
            #     ax.spines['top'].set_visible(False)
            #     ax.spines['bottom'].set_visible(False)
            #     ax.spines['left'].set_visible(False)
            #     ax.spines['right'].set_visible(False)
            #     plt.xticks([])
            #     plt.yticks([])  
            # plt.show()
            
            
            # "------ plot MAV profile ------" #only Peroneus
            # plots= diff_p.shape[0]
            # figure(num=5 , figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
            # for i in range(len(MAVp_channels)):
            #      ax = plt.subplot(plots,1,i + 1)
            #      plt.plot(MAVp_channels[i],color='blue',linewidth=0.5)
            #      ax.spines['top'].set_visible(False)
            #      ax.spines['bottom'].set_visible(False)
            #      ax.spines['left'].set_visible(False)
            #      ax.spines['right'].set_visible(False)
            #      plt.xticks([])
            #      plt.yticks([])  
            # plt.show()



            # "------- Plot single channel WL profile -----"
            # figure(num=6, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
            # plt.plot(WLgl_channels[22],linewidth=0.5)  #plotting only gastrocnemio lateralis
            # plt.title('single channel WL profile')
            # plt.xlabel('time')
            # plt.ylabel('WL')
            # plt.show()

    
            
            
            # for c in range(p_diffchannels):
            #         MAVp_channels[0,c] = fMAV(diff_p[c,:]);  
            #         WLp_channels[0,c] = fWL(diff_p[c,:]);
            #         WAMPp_channels[0,c] = fWAMP(diff_p[c,:],0.1);
            #         MAV_calc += MAVp_channels[0,c];
            #         WL_calc += WLp_channels[0,c];
            #         WAMP_calc += WAMPp_channels[0,c];
            # MAV_muscle[1] = MAV_calc / p_diffchannels;
            # WL_muscle[1] = WL_calc / p_diffchannels;
            # WAMP_muscle[1] = WAMP_calc/ p_diffchannels;
            # FTR[0,1] = MAV_muscle[1]
            # FTR[1,1] = WL_muscle[1]
            # FTR[2,1] = WAMP_muscle[1]

            
            
        
            
            
            # for c in range(gm_diffchannels):
            #         MAVgm_channels[0,c] = fMAV(diff_gm[c,:]);  
            #         WLgm_channels[0,c] = fWL(diff_gm[c,:]);
            #         WAMPgm_channels[0,c] = fWAMP(diff_gm[c,:],0.1);
            #         MAV_calc += MAVgm_channels[0,c];
            #         WL_calc += WLgm_channels[0,c];
            #         WAMP_calc += WAMPgm_channels[0,c];
            # MAV_muscle[2] = MAV_calc / gm_diffchannels;
            # WL_muscle[2] = WL_calc / gm_diffchannels;
            # WAMP_muscle[2] = WAMP_calc/ gm_diffchannels;
            # FTR[0,2] = MAV_muscle[2]
            # FTR[1,2] = WL_muscle[2]
            # FTR[2,2] = WAMP_muscle[2]

            
            
         
            
            # for c in range(ta_diffchannels):
            #         MAVta_channels[0,c] = fMAV(diff_ta[c,:]);  
            #         WLta_channels[0,c] = fWL(diff_ta[c,:]);
            #         WAMPta_channels[0,c] = fWAMP(diff_ta[c,:],0.1);
            #         MAV_calc += MAVta_channels[0,c];
            #         WL_calc += WLta_channels[0,c];
            #         WAMP_calc += WAMPta_channels[0,c];
            # MAV_muscle[3] = MAV_calc / ta_diffchannels;
            # WL_muscle[3] = WL_calc / ta_diffchannels;
            # WAMP_muscle[3] = WAMP_calc/ ta_diffchannels;
            # FTR[0,3] = MAV_muscle[3]
            # FTR[1,3] = WL_muscle[3]
            # FTR[2,3] = WAMP_muscle[3]

            
            


            
            
            # for c in range(sol_diffchannels):
            #         MAVsol_channels[0,c] = fMAV(diff_sol[c,:]);  
            #         WLsol_channels[0,c] = fWL(diff_sol[c,:]);
            #         WAMPsol_channels[0,c] = fWAMP(diff_sol[c,:],0.1);
            #         MAV_calc += MAVsol_channels[0,c];
            #         WL_calc += WLsol_channels[0,c];
            #         WAMP_calc += WAMPsol_channels[0,c];
            # MAV_muscle[4] = MAV_calc / sol_diffchannels;
            # WL_muscle[4] = WL_calc / sol_diffchannels;
            # WAMP_muscle[4] = WAMP_calc/ sol_diffchannels;
            # FTR[0,4] = MAV_muscle[4]
            # FTR[1,4] = WL_muscle[4]
            # FTR[2,4] = WAMP_muscle[4]
            
            
            
            # strPF = "_PF_";
            # strDF = "_DF_";

            # if (name.find(strPF) != -1):
            #     FTR[3,:] = 1;
            #     FTR_PF = np.copy(FTR);
            #     allFTR_PF.append(FTR_PF)
            # if (name.find(strDF) != -1):
            #     FTR[3,:] = 0;
            #     FTR_DF = np.copy(FTR);
            #     allFTR_DF.append(FTR_DF)
                
            
            
            

            

            # "------ plot original data ---------"
            # plots=32
            # figure(num=5 , figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
            # for i in range(plots):
            #         ax = plt.subplot(plots,1,i + 1)
            #         plt.plot(df_gl.iloc[:,i],color='blue',linewidth=0.5)
            #         ax.spines['top'].set_visible(False)
            #         ax.spines['bottom'].set_visible(False)
            #         ax.spines['left'].set_visible(False)
            #         ax.spines['right'].set_visible(False)
            #         plt.xticks([])
            #         plt.yticks([])    
            # plt.show()


            # "------- Plot a single channel versus time ------"
            # figure(num=2, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
            # plt.plot(t,df_gl.iloc[:,1],linewidth=0.5)
            # plt.title('single channel of original data')
            # plt.xlabel('time')
            # plt.ylabel('value')
            # plt.show()

            # "------ plot channels after differentiation ------" #only peroneus
            # plots= diff_p.shape[0]
            # figure(num=5 , figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
            # for i in range(plots-1):
            #      ax = plt.subplot(plots,1,i + 1)
            #      plt.plot(diff_p[i,:],color='blue',linewidth=0.5)
            #      ax.spines['top'].set_visible(False)
            #      ax.spines['bottom'].set_visible(False)
            #      ax.spines['left'].set_visible(False)
            #      ax.spines['right'].set_visible(False)
            #      plt.xticks([])
            #      plt.yticks([])  
            # plt.show()


          

            # "------- Plot of a single channel versus time after differentiation -----"
            
            # figure(num=4, figsize=(8, 6), dpi=200, facecolor='w', edgecolor='k')
        
            # plt.plot(t,diff_gl[0,:],linewidth=0.5)  #plotting only gastrocnemio lateralis
            # plt.title('single channel of data CHECK after differentiation')
            # plt.xlabel('time')
            # plt.ylabel('value')
            # plt.show()






















