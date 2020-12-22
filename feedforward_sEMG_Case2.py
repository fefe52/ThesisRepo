"---Project thesis Federica Aresu----"
" for tradition sEMG - CASE 2"
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
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from matplotlib.font_manager import FontProperties
from thesisproject_Fede_Case2 import *
from features import *
from differentiation import *
from groundtruth_Case2 import *


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
    in_seq = all_rec_sEMG; 

    # horizontally stack columns
    dataset = hstack((in_seq, out_pre_seq))
    # choose a number of time steps
    n_steps_in, n_steps_out = 20, 1


    # convert into input/output
    X, Y = split_sequences(dataset, n_steps_in, n_steps_out)
    print("X.shape is, and Y.shape is," , X.shape, Y.shape)
    # summarize the data
    #for i in range(len(X)):
    #	print(X[i], Y[i])

    # Flatten input and output
    n_input = X.shape[1] * X.shape[2]    

    X = X.reshape((X.shape[0], n_input))
    print("X shape",X.shape) ### The shape is now number of elements x (n*split input * channels)


    #using a split of 60-(40-60)
    features_size = int(len(X)*0.67)
    #test_size = int(len(X)*0.100)
    target_size = int(len(Y)*0.67)
    train_features, test_features = X[0:features_size], X[features_size:len(X)]
    val_size = int(len(test_features)*0.40)
    val_features, test_features = test_features[0:val_size],test_features[val_size:len(test_features)]
    train_target, test_target = Y[0:target_size], Y[target_size:len(Y)] 
    target_scaler = MinMaxScaler()
    target_scaler.fit(train_target)
    train_target = target_scaler.transform(train_target);
    test_target = target_scaler.transform(test_target);
    val_target, test_target = test_target[0:val_size], test_target[val_size:len(test_target)] 
    
    
    figure()
    plt.plot(Y)
    plt.savefig(CWD + '/figures/all target.png')
    plt.show()
    
    figure()
    plt.plot(test_target)
    plt.savefig(CWD + '/figures/test_target.png')
    plt.show()


    print("train_target",len(train_target))
    print("validation_target",len(val_target))
    print("test_target",len(test_target))

    print("----------")
    print("train_features", len(train_features))
    # Normalize data
    #train_features = train_features.describe()
    #train_features = train_features.transpose()
    #print("train_features characteristics",train_features)

    # Create the model
    model = Sequential()
    model.add(Dense(40,kernel_regularizer=regularizers.l2(0.001),  activation='relu',input_dim=n_input))
    #model.add(Dropout(.2))
    model.add(Dense(20,kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    #model.add(Dense(10, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(5, activation = 'relu'))
    model.add(Dense(5,activation = 'relu'))
    model.add(Dense(n_steps_out))
    kernel_regularizer=regularizers.l2(0.001)
    # select the optimizer with learning rate 
    optim_adam=keras.optimizers.Adam(lr=0.001)

    # Configure the model and start training
    #we use MSE because it is a regression problem
    #the optimizer shows how we update the weights
    model.compile(loss='mean_squared_error', optimizer=optim_adam, metrics=['mean_absolute_error', 'mean_squared_error'])
    model.summary()

    # Early stopping
    
    class MyThresholdCallback(keras.callbacks.Callback):
        def __init__(self, threshold):
            super(MyThresholdCallback, self).__init__()
            self.threshold = threshold

        def on_epoch_end(self, epoch, logs=None):
            val_loss = logs["val_loss"]
            if val_loss < self.threshold:
                self.model.stop_training = True

    my_callback = MyThresholdCallback(threshold=0.20)
    

    # validation_split=0.2 TO USE
    model_history = model.fit(train_features, train_target, validation_data=(val_features,val_target), epochs=8000, batch_size = len(train_target), verbose=1, callbacks= [my_callback])
    ### to plot model's training cost/loss and model's validation split cost/loss
    hist = pd.DataFrame(model_history.history)
    hist['epoch'] = model_history.epoch
    print("hist_tail",hist.tail())

    ### Predictions
    train_targets_pred = model.predict(train_features);
    train_targets_pred = target_scaler.inverse_transform(train_targets_pred);
    test_targets_pred = model.predict(test_features);
    test_targets_pred = target_scaler.inverse_transform(test_targets_pred);

    train_target = target_scaler.inverse_transform(train_target);
    test_target = target_scaler.inverse_transform(test_target);
    print("len of train pred", train_targets_pred.shape)
    print("len of test pred", test_targets_pred.shape)
    ### R2 score of training and testing data 
    # R2 is a statistical measure of how close the data is to the regression model (its output)
    print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(train_target,train_targets_pred)))
    print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(test_target,test_targets_pred)))
    ## if we are having r2_score bigger of train set then in test set, we are probably overfitting
    RMSE_train = np.square(np.subtract(train_target, train_targets_pred)).mean()
    RMSE_test = np.square(np.subtract(test_target,test_targets_pred)).mean()
    mean_test_target = test_target.mean()
    SI_test = RMSE_test / mean_test_target
    print("The RMSE on Train set is: ", RMSE_train)
    print("The RMSE on Test set is: ", RMSE_test)
    print("The SI on Test set is: ", SI_test)

    




    "--Plots--"
    def plot_history(history):
        hist = pd.DataFrame(model_history.history)
        hist['epoch'] = model_history.epoch
        
        fontP = FontProperties()
        fontP.set_size('xx-small')
        
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error')
        plt.title('MAE using MLP on sEMG data - study case 2')
        plt.plot(hist['epoch'], hist['mean_absolute_error'],label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_absolute_error'],label = 'Val Error')
        plt.legend()
        plt.savefig(CWD + '/figures/Case2/sEMG/MLP/sEMG_MLP_MAE_studycase2.png')

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error ')
        plt.title('MSE using MLP on sEMG data - study case 2')
        plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
        plt.legend()
        plt.savefig(CWD + '/figures/Case2/sEMG/MLP/sEMG_MLP_MSE_studycase2.png')
        plt.show()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Torque[Nm]')
        plt.title('MLP predictions on sEMG training - study case 2')
        plt.plot(train_target)
        plt.plot(train_targets_pred)
        plt.savefig(CWD + '/figures/Case2/sEMG/MLP/sEMG_MLP_pred_training_studycase2.png')
        plt.show()

        #plot
        #plt.figure()
        #plt.scatter(train_target,edgecolors='g')
        #plt.plot(train_targets_pred,'r')
        #plt.legend([ 'Predictated Y' ,'Actual Y'])
        #plt.savefig(CWD + '/figures/trainpredictions.png')
        #plt.show()
        
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Torque[Nm]')
        plt.plot(test_target,'g')
        plt.plot(test_targets_pred,'r')
        plt.title('MLP predictions on sEMG test - study case 2')
        plt.legend(['actual target','predictated values'], prop = fontP)
        plt.savefig(CWD + '/figures/Case2/sEMG/MLP/sEMG_MLP_pred_test_studycase2.png')
        plt.show()
    plot_history(model_history)
    
if __name__ == "__main__":
    main()
    





