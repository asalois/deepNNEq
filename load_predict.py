# Make and train a Deep NN Eq in Python

import sys
import os
import scipy.io as spio
import numpy as np
import pandas as pd
import math
import keras
SNR = str(sys.argv[1])
gpuNum = str(sys.argv[2])
print(SNR)

os.environ["CUDA_VISIBLE_DEVICES"]=gpuNum

# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
 
model = load_model('deep_model_SNR40.h5')
model.summary()


# Load the data
matname = "predictSNR" + SNR + ".mat"
print(matname)
mat = spio.loadmat(matname, squeeze_me=True)

x_test = mat['data']


# Convert the data to floats between 0 and 1.
x_test = x_test.astype('float32')
x_test /= 255
print(x_test.shape, 'test samples')

predictions = model.predict(x_test)
print(predictions.shape)
print(predictions[0:9])
#spio.savemat('predict_SNR40_out.mat',{'pred',predictions})
prediction = pd.DataFrame(predictions, columns=['I','Q']).to_csv('prediction.csv')
#print(y_test[0:9])
