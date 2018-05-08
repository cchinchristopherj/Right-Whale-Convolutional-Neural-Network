''' whale_cnn_unsup.py
        Contains classes and functions used by data.py, training_v1.py,
	and training_v2.py
'''

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

def MiniBatchKMeansAutoConv(X, patch_size, max_patches, n_clusters, conv_orders, batch_size=20):
    # Input Shape: (Number of samples, Number of filters, Height, Width)
    sz = X.shape
    # Transpose to Shape: (Number of samples, Height, Width, Number of Filters) and extract
    # patches from each sample up to the maximum number of patches using sklearn's
    # PatchExtractor
    X = image.PatchExtractor(patch_size=patch_size,max_patches=max_patches).transform(X.transpose((0,2,3,1))) 
    # For later processing, ensure that X has 4 dimensions (add an additional last axis of
    # size 1 if there are fewer dimensions)
    if(len(X.shape)<=3):
        X = X[...,numpy.newaxis]
    # Local centering by subtracting the mean
    X = X-numpy.reshape(numpy.mean(X, axis=(1,2)),(-1,1,1,X.shape[-1])) 
    # Local scaling by dividing by the standard deviation 
    X = X/(numpy.reshape(numpy.std(X, axis=(1,2)),(-1,1,1,X.shape[-1])) + 1e-10) 
    # Transpose to Shape: (Number of samples, Number of Filters, Height, Width)
    X = X.transpose((0,3,1,2)) 
    # Number of batches determined by number of samples in X and batch size
    n_batches = int(numpy.ceil(len(X)/float(batch_size)))
    # Array to store patches modified by recursive autoconvolution
    autoconv_patches = []
    for batch in range(n_batches):
        # Obtain the samples corresponding to the current batch 
        X_order = X[numpy.arange(batch*batch_size, min(len(X)-1,(batch+1)*batch_size))]
        # conv_orders is an array containing the desired orders of recursive autoconvolution
        # (with 0 corresponding to no recursive autoconvolution and 3 corresponding to 
        # recursive autoconvolution of order 3) 
        for conv_order in conv_orders:
            if conv_order > 0:
                # Perform recursive autoconvolution using "autoconv2d"
                X_order = autoconv2d(X_order)
                # In order to perform recursive autoconvolution, the height and width 
                # dimensions of X were doubled. Therefore, after recursive autoconvolution is
                # performed, reduce the height and width dimensions of X by 2. 
                X_sampled = resize_batch(X_order, [int(numpy.round(s/2.)) for s in X_order.shape[2:]])
                if conv_order > 1:
                    X_order = X_sampled
                # Resize X_sampled to the expected shape for MiniBatchKMeans
                if X_sampled.shape[2] != X.shape[2]:
                    X_sampled = resize_batch(X_sampled, X.shape[2:])
            else:
                X_sampled = X_order
            # Append the output of each order of recursive autoconvolution to "autoconv_patches"
            # in order to provide MiniBatchKMeans with a richer set of patches 
            autoconv_patches.append(X_sampled)
        print('%d/%d ' % (batch,n_batches))
    X = numpy.concatenate(autoconv_patches) 
    # Reshape X into a 2-D array for input into MiniBatchKMeans
    X = numpy.asarray(X.reshape(X.shape[0],-1),dtype=numpy.float32)
    # 
    X = mat2gray(X)
    pca = PCA(whiten=True)
    X = pca.fit_transform(X)
    X = normalize(X)
    km = MiniBatchKMeans(n_clusters = n_clusters,batch_size=batch_size,init_size=3*n_clusters).fit(X).cluster_centers_
    return km.reshape(-1,sz[1],patch_size[0],patch_size[1])

def mat2gray(X):
    m = numpy.min(X,axis=1,keepdims=True)
    X_range = numpy.max(X,axis=1,keepdims=True)-m
    idx = numpy.squeeze(X_range==0)
    X[idx,:] = 0
    X[numpy.logical_not(idx),:] = (X[numpy.logical_not(idx),:]-m[numpy.logical_not(idx)])/X_range[numpy.logical_not(idx)]
    return X

def learn_connections_unsup(feature_maps,num_groups,group_size,flag):
    num_samples = feature_maps.shape[0]
    num_filters = feature_maps.shape[1]
    height = feature_maps.shape[2]
    width = feature_maps.shape[3]
    if flag == 0: 
        feature_maps = feature_maps.reshape((num_samples*num_filters,height*width))
    else: 
        feature_maps = feature_maps.reshape((num_samples,num_filters,height*width))
        feature_maps = feature_maps.transpose((0,2,1))
        feature_maps = feature_maps.reshape((num_samples*height*width,num_filters))
    min_max_scaler = MinMaxScaler()
    feature_maps = min_max_scaler.fit_transform(feature_maps)
    pca = PCA(.99,whiten=True)
    feature_maps = pca.fit_transform(feature_maps)
    feature_maps = numpy.square(feature_maps)
    S_all = mat2gray(pairwise_distances(feature_maps,feature_maps,metric='correlation'))
    n_it = 100
    connections_all = dict()
    S_total = numpy.zeros(n_it)
    for it in range(n_it):
        S_all_it = S_all;
        rand_feats = numpy.random.randint(0,S_all_it.shape[0],size=num_groups)
        connections_tmp = numpy.zeros((num_groups,group_size))
        for ii in range(num_groups):
            while(True):
                S_tmp = S_all_it
                S_tmp[rand_feats[ii],rand_feats[ii]] = float('inf')
                ids = numpy.argsort(S_tmp[rand_feats[ii],:])
                S_min = S_tmp[rand_feats[ii],:][ids]
                closest = S_min[0:group_size-1]
                if flag == 0:
                    group = numpy.sort(numpy.append([rand_feats[ii]%num_filters],[ids[0:group_size-1]%num_filters]))
                else:
                    group = numpy.sort(numpy.append([rand_feats[ii]%(height*width)],[ids[0:group_size-1]%(height*width)]))
                if(len(numpy.unique(group)) == len(group)):
                    connections_tmp[ii,:] = group
                    break
                rand_feats[ii] = numpy.random.randint(0,S_all_it.shape[0],size=1)
            S_all_it = S_tmp
            S_total[it] = S_total[it]+numpy.sum(closest)
        connections_tmp = numpy.asarray(connections_tmp,dtype=numpy.int32)
        connections_all[it] = connections_tmp
    ids = numpy.argsort(S_total)
    n_unique_feats = numpy.zeros(round(n_it/10))
    for it in range(round(n_it/10)):
        n_unique_feats[it] = len(numpy.unique(connections_all[ids[it]].flatten()))
    num_maxind = len(numpy.where(n_unique_feats == numpy.amax(n_unique_feats))[0])
    if num_maxind == 1:
        it = ids[numpy.squeeze(numpy.where(n_unique_feats == numpy.amax(n_unique_feats)))]
    else: 
        it = ids[numpy.squeeze(numpy.where(n_unique_feats == numpy.amax(n_unique_feats)))[0]]
    connections = numpy.unique(connections_all[it],axis=0)
    if flag == 0:
        while(connections.shape[0] != num_groups):
            indices = numpy.arange(num_filters)
            numpy.random.shuffle(indices)
            connections = numpy.concatenate((connections,indices[0:group_size].reshape((1,group_size))),axis=0)
    return connections

def model_fp(model,model_train,layer_name):
    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)  
    output = intermediate_layer_model.predict(model_train)
    return output 

def autoconv2d(X):
    sz = X.shape 
    X -= numpy.reshape(numpy.mean(X, axis=(2,3)),(-1,sz[-3],1,1)) 
    X /= numpy.reshape(numpy.std(X, axis=(2,3)),(-1,sz[1],1,1)) + 1e-10 
    X = numpy.pad(X, ((0,0),(0,0),(0,sz[-2]-1),(0,sz[-1]-1)), 'constant') 
    return numpy.real(numpy.fft.ifft2(numpy.fft.fft2(X)**2))

def resize_batch(X, new_size):
    out = []
    for x in X:
        out.append(resize(x.transpose((1,2,0)), new_size, order=1, preserve_range=True,mode='constant').transpose((2,0,1)))
    return numpy.stack(out)

def separate_trainlabels(Y_train):
    indices = numpy.arange(len(Y_train))
    numpy.random.shuffle(indices)
    labels = Y_train[indices]
    Y_pos = []
    Y_neg = []
    for ii in range(len(labels)):
        if labels[ii] == 1:
            if len(Y_pos) < 450:
                Y_pos.append(ii)
        else:
            if len(Y_neg) < 450:
                Y_neg.append(ii)
        if len(Y_pos) == 450 and len(Y_neg) == 450:
            break
    return Y_pos,Y_neg,indices

class roc_callback(Callback):
    def __init__(self,training_data):
        super(roc_callback,self).__init__()
        self.x = training_data[0]
        self.y = training_data[1]
    def on_train_begin(self, logs={}):
        if not('roc_auc_val' in self.params['metrics']):
            self.params['metrics'].append('roc_auc_val')
    def on_train_end(self, logs={}):
        return
    def on_epoch_begin(self, epoch, logs={}):
        return
    def on_epoch_end(self, epoch, logs={}):
        logs['roc_auc_val'] = float('-inf')
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        logs['roc_auc_val'] = roc
        print('\rroc-auc: %s' % (str(round(roc,4))),end=100*' '+'\n')
        return
    def on_batch_begin(self, batch, logs={}):
        return
    def on_batch_end(self, batch, logs={}):
        return
