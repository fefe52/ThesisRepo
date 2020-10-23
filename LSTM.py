"********* KTH THESIS PROJECT FEDERICA ARESU **********"

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from numpy import hstack
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from thesisproject_Fede import *
from features import fMAV
from differentiation import *
from groundtruth import *





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
    
 # convert to [rows, columns] structure
in_seq = MAVgl_channels;   

# horizontally stack columns
dataset = hstack((in_seq, out_seq))
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
    
train_features, test_features, train_target, test_target = train_test_split(X, Y, test_size = 0.25, random_state = 42, shuffle = False)
# reshape input to be 3D [samples, timesteps, features]
train_features = train_features.reshape((train_features.shape[0], 1, train_features.shape[1]))
test_features = test_features.reshape((test_features.shape[0], 1, test_features.shape[1]))
print(train_features.shape, train_target.shape, test_target.shape, test_features.shape)

# design network
model = Sequential()
model.add(LSTM(100, input_shape=(train_features.shape[1], train_features.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
model.summary()
# fit network
history = model.fit(train_features, train_target, epochs=50, batch_size=72, validation_data=(test_features, test_target), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# make a prediction
yhat = model.predict(test_features)
test_features = test_features.reshape((test_features.shape[0], test_features.shape[2]))
# invert scaling for forecast
#inv_yhat = concatenate((yhat, test_features[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,0]
# # invert scaling for actual
# test_target = test_y.reshape((len(test_target), 1))
# inv_y = concatenate((test_target, test_features[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,0]
# # calculate RMSE
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)
