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
print(SNR)

SNR = str(sys.argv[1])
SNRs = str(SNR).zfill(2)
print(SNRs)
savename = "deep_model_SNR" + SNRs + '.h5'
model = keras.models.load_model(savename)
model.summary()

print("--- %.2f seconds ---" % (time.time() - start_time))
