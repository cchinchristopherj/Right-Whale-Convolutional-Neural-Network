'''
Usage:
python data.py
'''
# Generate data for training and test set from raw audio recordings 
# Generates X_train, Y_train, X_testV, X_testH, Y_test

import whale_cnn
import numpy as np
import glob
from skimage.transform import resize
import aifc
import pylab as pl
import os
from matplotlib import mlab
from matplotlib.pyplot import imshow

# Spectrogram parameters 
params = {'NFFT':256,'Fs':2000,'noverlap':192}
# Load in the audio files from the training dataset
path = 'Documents/Bioacoustics_MachineLearning/train2'
filenames = glob.glob(path+'/*.aif')
# As there are ~47000 audio files, 42000 are chosen to train the CNN, leaving
# ~5000 (~10%) for the test set 
num_files = 42000
# For each audio file, extract the spectrograms with vertically-enhanced contrast 
# separately from the spectrograms with horizontally-enhanced contrast. This in 
# effect doubles the amount of data for training, and presents the CNN with different 
# perspectives of the same spectrogram image of the original audio file 
training_featuresV = np.array([extract_featuresV(x,params=params) for x in filenames[0:num_files]])
training_featuresH = np.array([extract_featuresH(x,params=params) for x in filenames[0:num_files]])
# Concatenate the two feature matrices together to form a double-length feature matrix
X_train = np.append(training_featuresV,training_featuresH,axis=0)
# Axis 0 indicates the number of examples, Axis 1 and 2 are the features (64x64 image
# spectrograms). Add Axis 3 to indicate 1 channel (depth of 1 for spectrogram image) for 
# compatibility with Keras CNN model 
X_train = X_train[:,:,:,np.newaxis]
# Extract labels for the training dataset. Since the vertically-enhanced and 
# horizontally-enhanced images are concatenated to form a training dataset twice as long,
# append a duplicate copy of the training labels to form a training label vector 
# twice as long 
Y_train = np.array([extract_labels(x) for x in filenames[0:num_files]])
Y_train = np.append(Y_train,Y_train)
# Repeat the same procedure for the 5000 files in the test dataset 
# This time, do not concatenate the two feature matrices together. The two test
# feature matrices will be evaluated by the model independently. 
X_testV = np.array([extract_featuresV(x,params=params) for x in filenames[num_files:num_files+5000]])
X_testH = np.array([extract_featuresH(x,params=params) for x in filenames[num_files:num_files+5000]])
X_testV = X_testV[:,:,:,np.newaxis]
X_testH = X_testH[:,:,:,np.newaxis]
# Do not append a duplicate copy of the test labels to form a test label vector twice
# as long, since the two feature matrices were not concatenated together previously. 
# The number of elements in the test label vector is the number of original audio
# files in the test dataset 
Y_test = np.array([extract_labels(x) for x in filenames[num_files:num_files+5000]])
