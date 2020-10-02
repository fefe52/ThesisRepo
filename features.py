"********* KTH THESIS PROJECT FEDERICA ARESU **********"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import time
from matplotlib.pyplot import figure
from differentiation import *


"features calculation with slide windows of varying size"
"with or without overlapping"


# "---------------MAV------------------"  
# # Mean absolute value function
# def fMAV(y):
#     mav = []
#     n = len(y);
#     for i in range(n):
#         z = y[(i*2048):((i+1)*2048)]; 
#         mav.append(np.sum(abs(z))/len(z));
#         if i == (119):
#              break
#     return mav
    
 
"--------------WL-------------------"

# Waveform length function
def fWL(y):
    temp_wl = []
    n = len(y);
    wl = []
    for i in range(1,n):
        temp_wl.append(abs(y[i] - y[i-1]))
    for i in range(n):
        z = temp_wl[(i*2048):((i+1)*2048)]
        wl.append(sum(z))
        if i == (118):
            break
    return wl
    
    
#remember the sum
"------------WAMP------------------"
# Willison Amplitude function
def fWAMP(y,thres):
    wamp = 0;
    n = len(y);
    for i in range(n-1):
        if abs(y[i] - y[i+1])>thres:
            wamp = wamp + 1;
    return wamp

