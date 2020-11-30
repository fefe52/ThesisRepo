"********* KTH THESIS PROJECT FEDERICA ARESU **********"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import time

def Differential_sEMG(data):
    diffData = np.zeros((len(data)));
    diffData[:] = (data.iloc[:,1] - 1) - (data.iloc[:,0]);
    return diffData
    
def Differential(data,grid_size):
    lFiber = 8;
    nFiber = grid_size/lFiber;
    diffData = np.zeros((int((nFiber*(lFiber-1))),len(data)))
    idx1 = 0;
    count = 0;
    
    if grid_size == 64:
            for row in range(int(lFiber*nFiber-1),0,-1):
                if idx1 == (grid_size - nFiber):
                    break
                if count == 7:
                    count = 0;
                    continue
                count = count + 1;
                idx2 = int(row);
                idx3 = int(row)-1;
                #print("idx1,idx2,idx3",idx1,idx2,idx3)
                diffData[idx1,:] = (data.iloc[:,idx2] - idx2) - (data.iloc[:,idx3] - idx3);
                #diffData[idx1,:] = (data.iloc[:,idx2]) - (data.iloc[:,idx3]);
                idx1 = idx1 + 1;           
            return diffData
    
    
    
    #differential along the same fiber
    count = 0;
    if grid_size == 32:
            for row in range(int(lFiber*nFiber)):
                if idx1 == (grid_size - nFiber):
                    break      
                if count == 7:
                    count = 0;
                    continue
                count = count + 1;
                idx2 = int(row) + 1;
                idx3 = int(row);
                #print("idx1, idx2 and idx3 ",idx1,idx2, idx3)
                
                diffData[idx1,:] = ((data.iloc[:,idx2] - idx2) - (data.iloc[:,idx3] - idx3));
                #diffData[idx1,:] = ((data.iloc[:,idx2]) - (data.iloc[:,idx3]));
                idx1 = idx1 + 1; 
            return diffData

def Differential_sol(data,grid_size):
       lFiber = 8;
       nFiber = grid_size/lFiber;
       diffData = np.zeros((int((nFiber*(lFiber-1))),len(data)))
       count = 0;
       idx1 = 0;
    
       if grid_size == 64:
               for row in range(int(lFiber*nFiber-1),0,-1):
                   if idx1 == (grid_size - nFiber):
                       break
                   if count == 7:
                       count = 0;
                       continue
                   count = count + 1;
                   idx2 = int(row)-1;
                   idx3 = int(row);
                   #print("idx1,idx2,idx3", idx1,idx2,idx3)
                   diffData[idx1,:] = (data.iloc[:,idx2] - idx2) - (data.iloc[:,idx3] - idx3);
                   #diffData[idx1,:] = (data.iloc[:,idx2]) - (data.iloc[:,idx3]);
                   idx1 = idx1 + 1;           
               return diffData
    
    
        
    
    
