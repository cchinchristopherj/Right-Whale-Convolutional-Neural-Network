'''
Usage:
python tuning.py
'''
# Tuning Number of Neurons in Fully-Connected Layer 

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

def create_model(neurons=100):
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
# Use a subset of the ~47000 files in the training dataset for hyperparameter 
# optimization
num_files = 5000
# For each audio file, extract the spectrograms with vertically-enhanced contrast 
# separately from the spectrograms with horizontally-enhanced contrast. This in 
# effect doubles the amount of data for tuning, and presents the CNN with different 
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

# Use sklearn wrapper on the Keras model to be able to use sklearn methods 
# such as GridSearchCV
model = KerasClassifier(build_fn=create_model,epochs=10,batch_size=100)
# Create dictionary of hyperparameters for GridSearchCV to evaluate 
# In this case, find the "best" number of neurons for the model's fully
# connected layer 
neurons=[100,125,150,175,200,225,250]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model,param_grid=param_grid)
# Fit using subset of training data and labels 
grid_result = grid.fit(X_train,Y_train)
# Print best combination of parameters found, using accuracy as the metric
print('Best: %f using %s' % (grid_result.best_score_,grid_result.best_params_))
# Print mean and standard deviation of accuracy scores for each combination
# of parameters evaluated by GridSearchCV
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean,std,param in zip(means,stds,params):
    print('%f (%f) with: %r' % (mean,std,param))

