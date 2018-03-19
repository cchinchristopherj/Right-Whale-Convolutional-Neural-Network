'''
Usage:
python training.py
'''
# Train Convolutional Neural Network for right whale upcall recognition 

import whale_cnn
import numpy as np
import matplotlib.pyplot as plt
import glob
from skimage.transform import resize
import aifc
import scipy.signal as sp
import scipy
import pylab as pl
import os
from sklearn.metrics import roc_auc_score
from matplotlib import mlab
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import Callback
from keras import layers
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model, Sequential
from keras.optimizers import SGD
from keras.utils import plot_model
from matplotlib.pyplot import imshow
import keras.backend as K
K.set_image_data_format('channels_last')
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (13,5)

def create_model():
    ''' create_model Method
            CNN Model for Right Whale Upcall Recognition 
            
            Returns: 
                Q: Compiled Keras CNN Model
                
    '''
    model = Sequential()
    # Dropout on the visible layer (1 in 5 probability of dropout) 
    model.add(Dropout(0.2,input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3]),name='drop1'))
    # Conv2D -> BatchNorm -> Relu Activation -> MaxPooling2D
    model.add(Conv2D(15,(7,7),strides=(1,1),name='conv1'))
    model.add(BatchNormalization(axis=3,name='bn1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2),name='max_pool1'))
    # Conv2D -> BatchNorm -> Relu Activation -> MaxPooling2D
    model.add(Conv2D(30,(7,7),strides=(1,1),name='conv2'))
    model.add(BatchNormalization(axis=3,name='bn2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2),name='max_pool2'))
    # Flatten to yield input for the fully connected layer 
    model.add(Flatten())
    model.add(Dense(200,activation='relu',name='fc1'))
    # Dropout on the fully connected layer (1 in 2 probability of dropout) 
    model.add(Dropout(0.5,name='drop2'))
    # Single unit output layer with sigmoid nonlinearity 
    model.add(Dense(1,activation='sigmoid',name='fc2'))
    # Use Stochastic Gradient Discent for optimization 
    sgd = SGD(lr=0.1,decay=0.005,nesterov=False)
    model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])
    return model

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

# Use sklearn wrapper on the Keras model to be able to use sklearn methods 
# such as GridSearchCV
model = KerasClassifier(build_fn=create_model,epochs=45,batch_size=100)

# Fit the model using training data and use the roc_callback class to print
# roc_auc score every epoch 
model.fit(X_train,Y_train,callbacks=[roc_callback(training_data=(X_train,Y_train))])
# Generate predicted labels for the vertically-enhanced test feature matrix and
# predicted labels for the horizontally-enhanced test feature matrix. 
# The final predicted label is the union of these two predicted labels. 
# For example, if the vertically-enhanced image predicts label 0, but the 
# horizontally-enhanced version of the same image predicts label 1, the label is
# determined to be label 1. If both sets predict label 0, the label is 0. If both sets
# predict 1, the label is 1. The union operation is implemented by adding both
# predicted label vectors and setting the maximum value to 1
Y_predV = model.predict(X_testV)
Y_predH = model.predict(X_testH)
Y_pred = Y_predV + Y_predH 
Y_pred[Y_pred>1] = 1
score = roc_auc_score(Y_test,Y_pred)
print('Test ROC_AUC Score = ' + str(score))

# Trained model architecture and weights can be saved into an h5 file using:
# model.model.save('whale_cnn.h5')

# h5 file can be loaded into a Keras model using: 
# loaded_model = load_model('whale_cnn.h5')
