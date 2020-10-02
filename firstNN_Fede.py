"---first experiment of creating a neural network----"
import os
import csv
import pandas as pd
import numpy as np
import keras
import sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

"-------------------------------------------------------------------"
"Below we define the function                                       "
"to create the baseline model to be evaluated.                      " 
"It is a simple model that has a single fully connected hidden layer"
"with the same number of neurons as input attributes (13)           "
"The network uses good practices such as the rectifier activation function" 
"for the hidden layer. No activation function is used for the output layer"
"because it is a regression problem and we are interested in      "
"predicting numerical values directly without transform.           "

"The efficient ADAM optimization algorithm is used"
"and a mean squared error loss function is optimized."
"-------------------------------------------------------------------"

os.chdir(r'C:\Users\fedea\Desktop');

# load dataset
dataframe = pd.read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Y = dataset[:,13]




# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model, I first used Adam but got a 255.98 MSE
	model.compile(loss='mean_squared_error', optimizer='sgd')
	return model


# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
# the n_splts should be the same as the n_samples
kfold = KFold(n_splits=3)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
