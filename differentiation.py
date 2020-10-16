"********* KTH THESIS PROJECT FEDERICA ARESU **********"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import time


def Differential(data,grid_size):
    lFiber = 8;
    nFiber = grid_size/lFiber;
    diffData = np.zeros((int((nFiber*(lFiber-1))),len(data)))
    noisedata = np.zeros((int(nFiber),len(data)))
    
    idx1 = 0;
    
    if grid_size == 64:
        for col in range(int(nFiber)):
            for row in range(int(lFiber-1),0,-1):
                idx2 = int((col-1)*lFiber + row);
                idx3 = int((col-1)*lFiber + (row-1));
                diffData[idx1,:] = (data.iloc[:,idx2] - idx2) - (data.iloc[:,idx3] - idx3);
                idx1 = idx1 + 1;           
        return diffData
    
    # if grid_size == 32:
    #     for col in range(int(nFiber),0,-1):
    #         for row in range(int(lFiber-1),0,-1):
    #             idx2 = int((col-1)*lFiber + row);
    #             idx3 = int((col-1)*lFiber + (row-1));
    #             diffData[idx1,:] = (data.iloc[:,idx2] - idx2) - (data.iloc[:,idx3] - idx3);
    #             idx1 = idx1 + 1;           
    #     return diffData
    
    
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
                diffData[idx1,:] = ((data.iloc[:,idx2] - idx2) - (data.iloc[:,idx3] - idx3));
                idx1 = idx1 + 1; 
            return diffData
    
    
    # #differential along the same fiber
    # count = 0;
    # val_n = 0;
    # if grid_size == 32:
    #     for i in range(int(nFiber)):
    #         noisedata[i,:] = (data.iloc[:,(i*lFiber + 7)] - (i*lFiber + 7)) - (data.iloc[:,(i*lFiber)]);
    #     for row in range(int(lFiber*nFiber)):
    #         if idx1 == (grid_size - nFiber):
    #             break
    #         if count == 7:
    #             count = 0;
    #             val_n = val_n + 1;
    #             continue
    #         count = count + 1;
    #         idx2 = int(row) + 1;
    #         diffData[idx1,:] = ((data.iloc[:,idx2] - idx2) - noisedata[val_n,:]);
    #         idx1 = idx1 + 1; 
    #     return diffData
    
def Differential_sol(data,grid_size):
       lFiber = 8;
       nFiber = grid_size/lFiber;
       diffData = np.zeros((int((nFiber*(lFiber-1))),len(data)))
    
       idx1 = 0;
    
       if grid_size == 64:
           for col in range(int(nFiber),0,-1):
               for row in range(int(lFiber-1)):
                   idx2 = int((col-1)*lFiber + (row -1));
                   idx3 = int((col-1)*lFiber + (row));
                   diffData[idx1,:] = (data.iloc[:,idx2] - idx2) - (data.iloc[:,idx3] - idx3);
                   idx1 = idx1 + 1;           
           return diffData
    
    
        
    
    