"---Project thesis Federica Aresu----"
### IMPLEMENTING REGRESSION MODELS ####
import os
import csv
import pandas as pd
import numpy as np
import keras
import sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import FitFailedWarning

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import warnings

from features import fMAV,fWL,fWAMP
from differentiation import *
from thesisproject_Fede import *
from groundtruth import *

os.chdir(r'C:\Users\fedea\Desktop');

# get the features and labels as array
Y = np.zeros([3,1])
Y[0,0] = allTRQ_PF[0].max();
Y[1,0] = allTRQ_DF[0].max();
Y[2,0] = allTRQ_DF[1].max();
#labels[1,0] = TRQ.min();
#labels[2,0] = TRQ.mean();


#### FOLLOW SAME ORDER OF GIVING FEATURES AND LABELS TO ASSOCIATE IN THE RIGHT WAY

X = np.zeros([3,20]);
X[0,:] = np.copy(allFTR_PF[0].ravel());
X[1,:] = np.copy(allFTR_DF[0].ravel());
X[2,:] = np.copy(allFTR_DF[1].ravel());

# Split of the data
train_features, test_features, train_target, test_target = train_test_split(X, Y, test_size = 0.25, random_state = 42, shuffle = False)

# Create the model
input_shape = (20,)
model = Sequential()
model.add(Dense(16, input_shape=input_shape, activation='relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1))

# Configure the model and start training
#we use MSE because it is a regression problem
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
model.summary()

# Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)    

# validation_split=0.2 TO USE
model_history = model.fit(train_features, train_target, epochs=400, batch_size=1, verbose=1, callbacks=[es], validation_data=(test_features, test_target))
# I can select the learning rate through the optimizer

# Evaluate the model
_, train_acc = model.evaluate(train_features, train_target, verbose=0)
_, test_acc = model.evaluate(test_features, test_target, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# Predictions
predictions = model.predict(train_features[:10])


"--Plots--"
# plt.figure(figsize=(30,10))
# plt.plot(range(1,401),model_history.history["val_loss"])
# plt.plot(range(1,401),model_history.history["loss"])
# plt.legend(["Val Mean Sq Error (Val Loss)","Train Mean Sq Error (Train Loss)"])
# plt.xlabel("EPOCHS")
# plt.ylabel("Mean Sq Error")
# plt.xticks(range(1,401))
# plt.show()





