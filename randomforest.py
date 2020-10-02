"---Project thesis Federica Aresu----"
#### RANDOM FOREST REGRESSION ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from features import fMAV,fWL,fWAMP
from differentiation import *
from thesisproject_Fede import *
from groundtruth import *


os.chdir(r'C:\Users\fedea\Desktop');

# get the features and labels as array
labels = np.zeros([3,1])
labels[0,0] = allTRQ_PF[0].max();
labels[1,0] = allTRQ_DF[0].max();
labels[2,0] = allTRQ_DF[1].max();
#labels[1,0] = TRQ.min();
#labels[2,0] = TRQ.mean();


#### FOLLOW SAME ORDER OF GIVING FEATURES AND LABELS TO ASSOCIATE IN THE RIGHT WAY

features = np.zeros([3,20]);
features[0,:] = np.copy(allFTR_PF[0].ravel());
features[1,:] = np.copy(allFTR_DF[0].ravel());
features[2,:] = np.copy(allFTR_DF[1].ravel());

        
#Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

#check number of training features and testing features
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# baseline predictions
baseline_preds = test_features[0,:]
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ' ,round(np.mean(baseline_errors),2))


"---- train model ----"
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

