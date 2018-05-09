'''
Usage:
python training.py
'''
# Train Convolutional Neural Network for right whale upcall recognition 

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

# Use the data() method from whale_cnn.py to generate the training and test datasets 
# and labels
X_train, Y_train, X_testV, X_testH, Y_test = data()

# Use sklearn wrapper on the Keras model to be able to use sklearn methods 
# such as GridSearchCV
model = KerasClassifier(build_fn=create_model,epochs=45,batch_size=100)

# Fit the model using training data and use the roc_callback class to print
# roc_auc score every epoch 
model.fit(X_train,Y_train,callbacks=[roc_callback(validation_data=(X_testV,Y_test))])
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
