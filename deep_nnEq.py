# Make and train a Deep NN Eq in Python

import sys
import os
import scipy.io as spio
import numpy as np
import pandas as pd
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.losses import MeanSquaredError
#SNR = str(sys.argv[1])
#gpuNum = str(sys.argv[2])
#print(SNR)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

num_classes = 2
batch_size = 1024
epochs = 1024

for SNR in range(20,31):

    SNRs = str(SNR).zfill(2)
    print(SNRs)
    # Load the data
    matname = "deepSNR" + SNRs + ".mat"
    print(matname)
    mat = spio.loadmat(matname, squeeze_me=True)
    x_train = mat['x']
    x_valid = mat['x_val']
    x_test = mat['x_test']
    y_train = mat['y']
    y_valid = mat['y_val']
    #y_test = mat['y_test']


    # Convert the data to floats between 0 and 1.
    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_valid /= 255
    x_test /= 255
    print(x_train.shape, 'train samples')
    print(x_valid.shape, 'valid samples')
    #print(x_test.shape, 'test samples')
    print(y_train.shape, 'train labels')
    print(y_valid.shape, 'valid labels')
    #print(y_test.shape, 'test labels')
    print('Label Examples:\n', y_train[0:9]);


    # Formatting
    fmtLen = int(math.ceil(math.log(max(batch_size, y_valid.shape[0]),10)))

    # Define the network
    model = Sequential()
    model.add(Dense(50, activation='tanh', input_dim=18))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(num_classes, activation='tanh'))

    model.summary()

    model.compile(loss=keras.metrics.mean_squared_error,
                  optimizer=SGD(),
                  metrics=[keras.metrics.RootMeanSquaredError(name='rmse')])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_valid, y_valid))

    score = model.evaluate(x_train, y_train, verbose=1)
    print('Final Training MSE:', score[0])
    print('Final Training RMSE:', score[1])

    score = model.evaluate(x_valid, y_valid, verbose=1)
    print('Final Validation MSE:', score[0])
    print('Final Validation RMSE:', score[1])

    #score = model.evaluate(x_test, y_test, verbose=1)
    #print('Test MSE:', score[0])
    #print('Test RMSE:', score[1])

    predictions = model.predict(x_test)
    csvname = "predictionsSNR" + SNRs + ".csv"
    prediction = pd.DataFrame(predictions, columns=['I','Q']).to_csv(csvname)
    #weight = model.get_weights()
    #np.savetxt('weight.csv' , weight , fmt='%s', delimiter=',')
    #print(predictions.shape)
    #print(predictions[0:9])
    #print(y_test[0:9])
    savename = "deep_model_SNR" + SNRs + ".h5"
    model.save(savename)
