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
from keras.wrappers.scikit_learn import KerasRegressor

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
    
    "scatter plot for correlation"
    plt.plot(in_seq,out_seq, 'o',color='cadetblue', label='correlation for same recording')
    plt.title("features vs groundtruth")
    plt.xlabel("MAV values")
    plt.ylabel("Torque values")
    plt.legend()
    plt.savefig(CWD + '/figures/correlation plot.png')
    plt.show()
    # horizontally stack columns
    dataset = hstack((in_seq, out_seq))
    print("correlation values", dataset.corr())
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
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(n_steps_out))
    
    # select the optimizer with learning rate 
    #optimizer=keras.optimizers.Adam(lr=.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # Configure the model and start training
    #we use MSE because it is a regression problem
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error', 'mean_squared_error'])
    model.summary()

    # Early stopping
    #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)    

    # validation_split=0.2 TO USE
    model_history = model.fit(train_features, train_target, epochs=400, batch_size=1, verbose=1, validation_data=(test_features, test_target))
    # I can select the learning rate through the optimizer

    # Evaluate the model
    # _, train_acc = model.evaluate(train_features, train_target, verbose=0)
    #_, test_acc = model.evaluate(test_features, test_target, verbose=0)
    #print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    

    hist = pd.DataFrame(model_history.history)
    hist['epoch'] = model_history.epoch
    print("hist_tail",hist.tail())

    # Predictions
    predictions = model.predict(train_features)
    print("predictions",predictions.shape)


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

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Prediction values')
        plt.legend(["Train data", "Test data"])
        plt.plot(train_target)
        plt.plot(predictions)
        plt.savefig(CWD + '/figures/Predictions.png')
        plt.show()


    plot_history(model_history)
    
if __name__ == "__main__":
    main()
    




