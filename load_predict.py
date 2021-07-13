# Make and train a Deep NN Eq in Python

import sys
import os
import scipy.io as spio
import math
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import MeanSquaredError
SNR = str(sys.argv[1])
print(SNR)

model = keras.models.load_model('deep_model_SNR40.h5')
model.summary()


# Load the data
matname = "predictionsSNR" + SNR + ".mat"
print(matname)
mat = spio.loadmat(matname, squeeze_me=True)
