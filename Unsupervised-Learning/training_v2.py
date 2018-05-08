import numpy 
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction import image
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.metrics.pairwise import pairwise_distances
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import layers
from keras.layers import Input, Lambda, Dense, Reshape, Dropout, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Concatenate
from keras.models import load_model, Model
from keras.optimizers import Adam
import tensorflow as tf
from matplotlib.pyplot import imshow
import keras
import keras.backend as K
K.set_image_data_format('channels_last')

def create_model(num_filters_0,num_filters_1,reduce_size,kernel_size,input_shape):
    ''' create_model Method
            CNN Model for Right Whale Upcall Recognition with filters learned through K-Means
            
            Args:
                num_filters_0: int number of filters for convolutional layer in Layer 0
                num_filters_1: int number of filters for convolutional layer in Layer 1
                reduce_size: int number of feature maps for the 1x1 convolutional layer 
                             between Layer 0 and Layer 1
                kernel_size: int size of the filters in the Conv2D layers (same kernel size
                             for all convolutional layers)
                input_shape: (Number of Samples, Height, Width, Number of Filters)
            Returns: 
                Compiled Keras CNN Model
                
    '''
    # There are two layers: Layer 0 is the convolution of the raw input layer with the
    # first set of learned filters (each filter is of depth 1 because the raw input layer 
    # is of depth 1). After the output of the convolutional layer is maxpooled, the number of
    # feature maps is reduced via 1x1 convolutions. The number of feature maps becomes 
    # "reduce_size." Layer 1 corresponds to the second set of learned filters that is 
    # convolved with the output of the 1x1 convolutional layer. The output of the 
    # convolutional layer is then maxpooled, flattened, concatenated with the flattened 
    # output of layer 0, then fed into two fully connected layers. 
    # Input Shape: (Number of Samples, Height, Width, Number of Filters)
    X_input = Input(shape=input_shape,name='input')
    # Dropout on the visible layer (1 in 10 probability of dropout) 
    X = Dropout(0.1,name='dropout0')(X_input)
    # BatchNorm on axis=3 (on the axis corresponding to the number of filters)
    X = BatchNormalization(axis=3,name='bn0')(X_input)
    # Layer 0 Convolutional layer. Since the filters are determined through unsupervised
    # learning, they are not updated through backpropagation (i.e. trainable=False)
    X = Conv2D(filters=num_filters_0,kernel_size=kernel_size,use_bias=False,activation='relu',name='conv0',trainable=False)(X)
    # Maxpooling for translation invariance and to halve height and width dimensions 
    X_maps1 = MaxPooling2D(name='maxpool0')(X)
    # 1x1 convolutional layer to reduce number of feature maps to "reduce_size"
    X = Conv2D(filters=reduce_size,kernel_size=1,name='reduce1')(X_maps1)
    # Layer 1 Convolutional Layer.
    X = Conv2D(filters=num_filters_1,kernel_size=kernel_size,use_bias=False,activation='relu',name='conv1',trainable=False)(X)
    # Maxpooling 
    X_maps2 = MaxPooling2D(name='maxpool1')(X)
    # Flatten the output of the maxpooling layer in Layer 0
    X_maps1_f = Flatten(name='flatten1')(X_maps1)
    # Flatten the output of the maxpooling layer in Layer 1
    X_maps2_f = Flatten(name='flatten2')(X_maps2)
    # Concatenate the flattened outputs into one feature vector
    X_maps = keras.layers.concatenate([X_maps1_f,X_maps2_f],name='final3')
    # Feed the concatenated output to two fully connected layers
    X = Dense(200,activation='relu',name='fc1')(X_maps2_f)
    # Dropout on the fully connected layer (1 in 2 probability of dropout) 
    X = Dropout(0.5,name='dropout3')(X)
    X_output = Dense(1,activation='sigmoid',name='fc_final')(X)
    # Use Adam optimizer
    opt = Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,decay=0.01)
    model = Model(inputs=X_input,outputs=X_output)
    # Compile the model for binary classification
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
    return model

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

# Parameters for Training Model:
# Size of Filters (kernel_size x kernel_size) in Keras model for all convolutional layers
kernel_size = 7
# Number of filters for Layer 0
num_filters_0 = 256
# The output of maxpooling in Layer 0 will have 256 feature maps. Since K-Means performs
# most effectively when the dimensionality of the samples fed into it is kept relatively low
# (100 or 200 features per sample), reduce the number of feature maps in the output via 
# 1x1 convolutions. Reduce the number of feature maps from num_filters_0 to reduce_size
reduce_size = 8
# Number of filters for Layer 1
num_filters_1 = 512
# Input Shape: (Number of Samples, Height, Width, Number of Filters)
input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])
# Instantiate the Keras model 
model = create_model(num_filters_0,num_filters_1,reduce_size,kernel_size,input_shape)
# "k_train" is the set of samples fed to K-Means to learn filters unsupervised
k_train = X_train.transpose((0,3,1,2))
# The "separate_trainlabels" function shuffles the training set (and training labels), and 
# yields the indices to replicate the shuffling operation. The function also generates
# "Y_pos" and "Y_neg," arrays containing indices of the samples in the training set belonging
# to the positive and negative class, respectively. ("Y_pos" and "Y_neg" are each set to 
# contain 450 samples). Equal number of samples from each of these arrays are therefore given 
# to K-Means, so that K-Means is able to learn its dictionary of filters based off features 
# from both the positive and negative class. 
Y_pos,Y_neg,indices = separate_trainlabels(Y_train)
# Shuffle k_train, model_train, and model_labels according to the indices specified by 
# "separate_trainlabels"
k_train = k_train[indices]
model_train = X_train[indices]
model_labels = Y_train[indices]
# Feed 50 examples from the positive class and 50 examples from the negative class to 
# K-Means to learn a dictionary of filters. Identify those examples in k_train using the
# indices in the Y_pos and Y_neg arrays 
k_train = numpy.concatenate((k_train[Y_pos[0:50]],k_train[Y_neg[0:50]]),axis=0)
# Use the "MiniBatchKMeansAutoConv" function to learn the dictionary of filters. Extract
# 1/3 of the total number of patches randomly from each example for training, learn
# num_filters_0 filters (centroids), and do not use recursive autoconvolution
centroids_0 = MiniBatchKMeansAutoConv(k_train,(kernel_size,kernel_size),0.33,num_filters_0,[0])
# The learned filters (centroids) will be set as the weights of the "conv0" layer.
layer_toset = model.get_layer('conv0')
# Reshape the learned filters so that they are the same shape as expected by the Keras layer
filters = centroids_0.transpose((2,3,1,0))
filters = filters[numpy.newaxis,...]
layer_toset.set_weights(filters)
# Construct an intermediate model (using model_fp) that generates the output of layer 
# "reduce1." This is the output that the next set of learned filters will be convolved with.
# Therefore, generate the output so that random patches from each example in the output can
# be fed to K-Means to learn a new dictionary of filters.
# For sake of efficiency, continue using only the 450 samples in Y_pos and 450 samples in Y_neg 
# for the processing below (i.e. forward pass only these 900 samples from the training set
# through the Keras model and obtain the output of layer "reduce1" for only these samples).
fp_train = numpy.concatenate((model_train[Y_pos],model_train[Y_neg]),axis=0)
output_0 = model_fp(model,fp_train,'reduce1')
# Transpose the output into shape: (Number of samples, number of filters, height, width)
# for "MiniBatchKMeansAutoConv"
output_0 = output_0.transpose((0,3,1,2))
# Feed all of output_0 to K-Means to learn a new dictionary of filters.
k_train = output_0
# Use the "MiniBatchKMeansAutoConv" function to learn the dictionary of filters. Extract
# 1/2 of the total number of patches randomly from each sample for training, learn
# num_filters_1 filters (centroids), and do not use recursive autoconvolution.
centroids_1 = MiniBatchKMeansAutoConv(k_train,(kernel_size,kernel_size),0.5,num_filters_1,[0])
# The learned filters (centroids) will be set as the weights of the "conv1" layer.
layer_toset = model.get_layer('conv1')
# Reshape the learned filters so that they are the same shape as expected by the Keras layer
filters = centroids_1.transpose((2,3,1,0))
filters = filters[numpy.newaxis,...]
layer_toset.set_weights(filters)
