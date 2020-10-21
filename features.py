"********* KTH THESIS PROJECT FEDERICA ARESU **********"

import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import pandas as pd
import csv
import time
from matplotlib.pyplot import figure
from differentiation import *


"features calculation with slide windows of varying size"
"with or without overlapping"


"---------------MAV------------------"  
# Mean absolute value function
def fMAV(y):
    n = len(y);
    #mav = np.zeros(239);
    mav = []
    len_window = 1024;
    count = 0;
    for i in range(n):
        if n in range((i*1024),((i+1)*1024)):
            break
        z = y[(i*1024):((i+1)*1024)];
        count = count + 1;
        mav.append(np.sum(abs(z))/len(z))
    mav = np.array(mav);
    print(mav)
    print("length",len(mav))
    return mav
 
# "--------------WL-------------------"

# # Waveform length function
# def fWL(y):
#     temp_wl = []
#     n = len(y);
#     wl = []
#     for i in range(1,n):
#         temp_wl.append(abs(y[i] - y[i-1]))
#     for i in range(n):
#         z = temp_wl[(i*2048):((i+1)*2048)]
#         wl.append(sum(z))
#         if i == (118):
#             break
#     return wl
    
    
# #remember the sum
# "------------WAMP------------------"
# # Willison Amplitude function
# def fWAMP(y,thres):
#     wamp = 0;
#     n = len(y);
#     for i in range(n-1):
#         if abs(y[i] - y[i+1])>thres:
#             wamp = wamp + 1;
#     return wamp

