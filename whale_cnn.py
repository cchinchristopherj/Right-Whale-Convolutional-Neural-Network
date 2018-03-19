''' whale_cnn.py
        Contains classes and functions used by training.py and tuning.py
'''

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
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (13,5)

# roc_callback class from: https://github.com/keras-team/keras/issues/3230#issuecomment-319208366
class roc_callback(Callback):
    ''' roc_callback Class
            Uses keras.callbacks.Callback abstract base class to build new callbacks 
            for visualization of internal states/statistics of the CNN model 
            during training. In this case, print the roc_auc score (from sklearn)
            for every epoch during training 
    '''
    def __init__(self,training_data):
        ''' __init__ Method 
            
            Args:
                training_data: 2 element tuple, the first element of which is the 
                training dataset and the second element of which is the training
                labels  
        '''  
        self.x = training_data[0]
        self.y = training_data[1]
        
    # The following three methods are not necessary for calculating the roc_auc 
    # score. Threfore, simply return 
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return
    
    def on_epoch_end(self, epoch, logs={}):
        ''' on_epoch_end Method 
            
            Args:
                epoch: Current epoch number  
                logs: Dictionary containing keys for quantities relevant 
                to current epoch 
        '''  
        # Use CNN model to predict labels for the training dataset 
        y_pred = self.model.predict(self.x)
        # Compute the roc_auc score using the predicted labels and 
        # ground truth training labels 
        roc = roc_auc_score(self.y, y_pred)
        print('\rroc-auc: %s' % (str(round(roc,4))),end=100*' '+'\n')
        return

    # The following two methods are not necessary for calculating the roc_auc 
    # score. Threfore, simply return 
    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    
# ReadAIFF Function from: https://github.com/nmkridler/moby/blob/master/fileio.py
def ReadAIFF(file):
    ''' ReadAIFF Method
            Read AIFF and convert to numpy array
            
            Args: 
                file: string file to read 
            Returns:
                numpy array containing whale audio clip      
                
    '''
    s = aifc.open(file,'r')
    nFrames = s.getnframes()
    strSig = s.readframes(nFrames)
    return np.frombuffer(strSig,np.short).byteswap()

# Based off H1Sample Function from: https://github.com/nmkridler/moby/blob/master/fileio.py
def SpecGram(file,params=None):
    ''' SpecGram Method 
            Convert audio file to spectrogram for CNN and pre-process input for
            input shape uniformity 
            
            Args:
                file: string file to read 
                params: dictionary containing spectrogram parameters  
            Returns: 
                Pre-Processed Spectrogram matrix and frequency/time bins as 1-D arrays
                
    '''
    s = ReadAIFF(file)
    # Convert to spectrogram 
    P,freqs,bins = mlab.specgram(s,**params)
    m,n = P.shape
    # Ensure all image inputs to the CNN are the same size. If the number of time bins 
    # is less than 59, pad with zeros 
    if n < 59:
        Q = np.zeros((m,59))
        Q[:,:n] = P
    else:
        Q = P
    return Q,freqs,bins

# slidingWindowV Function from: https://github.com/nmkridler/moby2/blob/master/metrics.py 
def slidingWindowV(P,inner=3,outer=64,maxM=50,minM=7,maxT=59,norm=True):
    ''' slidingWindowV Method
            Enhance the contrast vertically (along frequency dimension)
            
            Args:
                P: 2-D numpy array image
                inner: int length of inner exclusion region (to calculate local mean)
                outer: int length of the window (to calculate overall mean)
                maxM: int size of the output image in the y-dimension
                norm: boolean indicating whether or not to cut off extreme values 
            Returns: 
                Q: 2-D numpy array image with vertically-enhanced contrast  
                
    '''
    Q = P.copy()
    m, n = Q.shape
    if norm:
        # Cut off extreme values 
        mval, sval = np.mean(Q[minM:maxM,:maxT]), np.std(Q[minM:maxM,:maxT])
        fact_ = 1.5
        Q[Q > mval + fact_*sval] = mval + fact_*sval
        Q[Q < mval - fact_*sval] = mval - fact_*sval
        Q[:minM,:] = mval
    # Set up the local mean window 
    wInner = np.ones(inner)
    # Set up the overall mean window 
    wOuter = np.ones(outer)
    # Remove overall mean and local mean using np.convolve 
    for i in range(maxT):
        Q[:,i] = Q[:,i] - (np.convolve(Q[:,i],wOuter,'same') - np.convolve(Q[:,i],wInner,'same'))/(outer - inner)
    Q[Q < 0] = 0.
    return Q[:maxM,:]

# slidingWindowH Function from: https://github.com/nmkridler/moby2/blob/master/metrics.py
def slidingWindowH(P,inner=3,outer=32,maxM=50,minM=7,maxT=59,norm=True):
    ''' slidingWindowH Method
            Enhance the contrast horizontally (along temporal dimension)
            
            Args:
                P: 2-D numpy array image
                inner: int length of inner exclusion region (to calculate local mean)
                outer: int length of the window (to calculate overall mean)
                maxM: int size of the output image in the y-dimension
                norm: boolean indicating whether or not to cut off extreme values 
            Returns: 
                Q: 2-D numpy array image with horizontally-enhanced contrast  
                
    '''
    Q = P.copy()
    m, n = Q.shape
    if outer > maxT:
        outer = maxT
    if norm:
        # Cut off extreme values 
        mval, sval = np.mean(Q[minM:maxM,:maxT]), np.std(Q[minM:maxM,:maxT])
        fact_ = 1.5
        Q[Q > mval + fact_*sval] = mval + fact_*sval
        Q[Q < mval - fact_*sval] = mval - fact_*sval
        Q[:minM,:] = mval
    # Set up the local mean window 
    wInner = np.ones(inner)
    # Set up the overall mean window 
    wOuter = np.ones(outer)
    if inner > maxT:
        return Q[:maxM,:]
    # Remove overall mean and local mean using np.convolve 
    for i in range(maxM):
        Q[i,:maxT] = Q[i,:maxT] - (np.convolve(Q[i,:maxT],wOuter,'same') - np.convolve(Q[i,:maxT],wInner,'same'))/(outer - inner)
    Q[Q < 0] = 0.
    return Q[:maxM,:]

# PlotSpecgram Function from: https://github.com/nmkridler/moby2/blob/master/plotting.py
def PlotSpecgram(P,freqs,bins):
    ''' PlotSpecgram Method 
            Plot the spectrogram
            
            Args:
                P: 2-D numpy array image
                freqs: 1-D array of frequency bins
                bins: 1-D array of time bins   
    '''
    # Use np.flipud so that the spectrogram plots correctly 
    Z = np.flipud(P)
    xextent = 0,np.amax(bins)
    xmin,xmax = xextent
    extent = xmin,xmax,freqs[0],freqs[-1]
    im = pl.imshow(Z,extent=extent)
    pl.axis('auto')
    pl.xlim([0.0,bins[-1]])
    pl.ylim([0,400])

def extract_labels(file):
    ''' extract_labels Method 
            Since the dataset file names contain the labels (0 or 1) right before
            the extension, appropriately parse the string to obtain the label 
            
            Args:
                file: string file to read 
            Returns: 
                int label of the file (0 or 1) 
                
    '''
    name,extension = os.path.splitext(file)
    label = name[-1]
    return int(label)

def extract_featuresV(file,params=None):
    ''' extract_featuresV Method 
            Extract spectrogram with vertically-enhanced contrast from audio file
            for input to CNN
            
            Args:
                file: string file to read 
                params: dictionary containing spectrogram parameters 
            Returns: 
                Q: 2-D numpy array image with vertically-enhanced contrast  
                
    '''
    P,freqs,bins = SpecGram(file,params)
    Q = slidingWindowV(P,inner=3,maxM=50,maxT=bins.size)
    # Resize spectrogram image into a square matrix 
    Q = resize(Q,(64,64),mode='edge')
    return Q

def extract_featuresH(file,params=None):
    ''' extract_featuresH Method 
            Extract spectrogram with horizontally-enhanced contrast from audio file
            for input to CNN
            
            Args:
                file: string file to read 
                params: dictionary containing spectrogram parameters 
            Returns: 
                Q: 2-D numpy array image with horizontally-enhanced contrast  
                
    '''
    P,freqs,bins = SpecGram(file,params)
    W = slidingWindowH(P,inner=3,outer=32,maxM=50,maxT=bins.size)
    # Resize spectrogram image into a square matrix 
    W= resize(W,(64,64),mode='edge')
    return W
