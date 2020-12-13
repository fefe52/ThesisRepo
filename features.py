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
    mav = []
    len_window = 1024;
    for i in range(0,n,512):
        if n in range(i,(i+len_window)):
            break
        z = y[i:(i+len_window)];
        mav.append(np.sum(abs(z))/len(z))
    mav = np.array(mav);
    return mav
 
"--------------WL-------------------"
# Waveform length function
def fWL(y):
    n = len(y);
    wl = []
    len_window = 1024;
    for i in range(0,n,512):
        if n in range(i,(i+len_window)):
            break
        z = y[i:(i+len_window)];
        wl.append(np.sum(abs(z[i+1] - z[i])))   
    return wl
    
    
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

