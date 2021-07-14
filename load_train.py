# Make and train a Deep NN Eq in Python

import sys
import os
import scipy.io as spio
import math
import time
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import MeanSquaredError

start_time = time.time()
SNR = str(sys.argv[1])
SNRs = str(SNR).zfill(2)
print(SNRs)

num_classes = 2
batch_size = 64
epochs = 30

savename = "deep_model_SNR" + SNRs
model = keras.models.load_model(savename)
model.summary()

#Load the data
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

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_valid, y_valid))

score = model.evaluate(x_train, y_train, verbose=2)
print('Final Training MSE:', score[0])
print('Final Training RMSE:', score[1])

score = model.evaluate(x_valid, y_valid, verbose=2)
print('Final Validation MSE:', score[0])
print('Final Validation RMSE:', score[1])

predictions = model.predict(x_test)
matname = "predictionsSNR" + SNRs + ".mat"
spio.savemat(matname, {'pred': predictions})
model.save(savename)

print("--- %.2f seconds ---" % (time.time() - start_time))
