"---Project thesis Federica Aresu----"
### IMPLEMENTING REGRESSION MODELS ####
import os
import csv
import pandas as pd
import numpy as np
from numpy import hstack
import keras
import sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import FitFailedWarning

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import warnings

from thesisproject_Fede import *
from features import fMAV
from differentiation import *
from groundtruth import *


def main():
    # split a multivariate sequence into samples
    def split_sequences(sequences, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out-1
		    # check if we are beyond the dataset
            if out_end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)
    # get the features and labels as array


    #### FOLLOW SAME ORDER OF GIVING FEATURES AND LABELS TO ASSOCIATE IN THE RIGHT WAY

    

    # convert to [rows, columns] structure
    in_seq = MAVgl_channels; 
    #### data scaling from 0 to 1, since in_seq and out_seq have very different scales
    #X_scaler = preprocessing.MinMaxScaler()
    #y_scaler = preprocessing.MinMaxScaler()
    #in_seq = (X_scaler.fit_transform(in_pre_seq.reshape(-1,1)))
    #out_seq = (y_scaler.fit_transform(out_pre_seq.reshape(-1,1)))

    # horizontally stack columns
    dataset = hstack((in_seq, out_pre_seq))
    # choose a number of time steps
    n_steps_in, n_steps_out = 5, 1


    # convert into input/output
    X, Y = split_sequences(dataset, n_steps_in, n_steps_out)
    print("X.shape is, and Y.shape is," , X.shape, Y.shape)
    # summarize the data
    #for i in range(len(X)):
    #	print(X[i], Y[i])

    # Flatten input and output
    n_input = X.shape[1] * X.shape[2]    

    X = X.reshape((X.shape[0], n_input))

    # Split of the data
    train_features, test_features, train_target, test_target = train_test_split(X, Y, test_size = 0.25, random_state = 42, shuffle = False)
    
    # Normalize data
    #train_features = train_features.describe()
    #train_features = train_features.transpose()
    #print("train_features characteristics",train_features)

    # Create the model
    model = Sequential()
    model.add(Dense(100, activation='relu', input_dim=n_input))
    model.add(Dropout(.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(n_steps_out))
    
    # select the optimizer with learning rate 
    #optimizer=keras.optimizers.Adam(lr=.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # Configure the model and start training
    #we use MSE because it is a regression problem
    #the optimizer shows how we update the weights
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error', 'mean_squared_error'])
    model.summary()

    # Early stopping
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)    

    # validation_split=0.2 TO USE
    model_history = model.fit(train_features, train_target, epochs=1000, verbose=0, validation_split = 0.2)   # I can select the learning rate through the optimizer


    ### to plot model's training cost/loss and model's validation split cost/loss
    hist = pd.DataFrame(model_history.history)
    hist['epoch'] = model_history.epoch
    print("hist_tail",hist.tail())

    ### Predictions
    train_targets_pred = model.predict(train_features)
    test_targets_pred = model.predict(test_features)

    ### R2 score of training and testing data 
    # R2 is a statistical measure of how close the data is to the regression model (its output)
    print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(train_target,train_targets_pred)))
    print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(test_target,test_targets_pred)))
    ## if we are having r2_score bigger of train set then in test set, we are probably overfitting

    "--Plots--"
    def plot_history(history):
        hist = pd.DataFrame(model_history.history)
        hist['epoch'] = model_history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error')
        plt.plot(hist['epoch'], hist['mean_absolute_error'],label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_absolute_error'],label = 'Val Error')
        plt.legend()
        plt.savefig(CWD + '/figures/Mean abs Error.png')

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error ')
        plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
        plt.legend()
        plt.savefig(CWD + '/figures/Mean Square Error.png')
        plt.show()

        # plt.figure()
        # plt.xlabel('Epoch')
        # plt.ylabel('Prediction values')
        # plt.plot(train_target)
        # plt.plot(train_targets_pred)
        # plt.savefig(CWD + '/figures/Predictions vs groundtruth.png')
        # plt.show()


    plot_history(model_history)
    
if __name__ == "__main__":
    main()
    




