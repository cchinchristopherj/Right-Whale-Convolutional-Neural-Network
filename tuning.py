'''
Usage:
python tuning.py
'''
# Tuning Number of Neurons in Fully-Connected Layer 

import whale_cnn
import numpy as np
import glob
from skimage.transform import resize
import aifc
import pylab as pl
import os
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
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

# The following general approach to hyperparameter optimization using GridSearchCV 
# and a sklearn wrapper for Keras is based on:
# https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
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

# Use the data() method from whale_cnn.py to generate the training and test datasets 
# and labels
X_train, Y_train, X_testV, X_testH, Y_test = data()

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

