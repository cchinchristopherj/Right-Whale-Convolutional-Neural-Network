Whale Convolutional Neural Network (Unsupervised)
=========================

Convolutional Neural Network to Recognize Right Whale Upcalls via Filters Learned Through K-Means

Tackles the same [challenge](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux) and uses the same [dataset](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/data) as the supervised CNN model. The same data() method in whale_cnn.py is used to perform pre-processing on the raw data, i.e. horizontal and vertical contrast enhancement and resizing to (64x64x1) images, thereby producing the same X_train, Y_train, X_testV, X_testH, and Y_test. 

Two different model architectures were created, in which filters for convolutional layers are learned unsupervised. Both models were built using the same guiding principle that K-Means learns centroids more effectively if the dimensionality of the samples' features is kept relatively small. In the 0th convolutional layer (raw input), this is not an issue because the samples are 7x7 patches (49-dimensional features) from 64x64x1 raw spectrogram images. Therefore, both models feed full 7x7x1 patches to K-Means to learn a dictionary of 256 filters. However, note that even if maxpooling is applied to the output of the 0th convolutional layer, yielding an output shape of (batch_size,29,29,256), and 7x7 patches are once again extracted, the samples fed to K-Means are 7x7x256 = 12544-dimensional, which will prohibit K-Means from learning effectively. Two approaches are taken in the two architectures:
- Model 1: Energy-Correlated Receptive Fields
- Model 2: 1x1 Convolution Dimensionality Reduction

Model 1 (Energy-Correlated Receptive Fields Grouping)
=========================

- After the output of the 0th convolutional layer is maxpooled, yielding an output of shape (batch_size,29,29,256), calculate the squared-energy correlation between all 256 feature maps
- Create "num_groups" groups of "group_size" feature maps that are most strongly correlated with each other. In this case, hyperparameter tuning found "num_groups" = 32 and "group_size" = 8 to be most effective. 
- By breaking the entire set of feature maps into smaller groups that have high squared-energy correlation, K-Means will be able to learn more discriminative filters. (Since 7x7 patches are extracted from the smaller groups of "group_size" = 8, the samples fed to K-Means are 7x7x8 = 392-dimensional, two orders of magnitude smaller than the original approach). 

Model 2 (1x1 Convolution Dimensionality Reduction)
=========================

- After the output of the 0th convolutional layer is maxpooled, yielding an output of shape (batch_size,29,29,256), use 1x1 convolutions to reduce the depth (i.e. number of feature maps). The output of the 1x1 convolutional layer is of shape (batch_size,29,29,group_size), where group_size is a hyperparameter to be tuned. 
- "group_size" = 8 was found to be most effective, and also allowed for direct comparison with the Model 1 method. (Since there are "group_size" filters in the 1x1 convolutional layer, this will yield the exact same dimensionality of samples fed to K-Means as in Model 1). 
- The possible advantage of this approach is the drastic reduction in the complexity of Model 1's architecture. In addition, 1x1 convolutions function as micro networks, or multi-layer perceptrons that are slid across the image like convolutional layers, providing an additional nonlinearity to learn better representations. 

The final tuned model architecture for Model 2 is as depicted below: 

![cnn_architecture](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Unsupervised-Learning/Images/Model-1/cnn_architecture_unsup.png)

Correct Usage
=========================

My trained CNN model architecture and weights for Model 2 are saved in the "model_v2.h5" file. This file can be loaded using the command:

    loaded_model = load_model('model_v2.h5')  
    
Note: "load_model" is a function from keras.models. 
With this model loaded, you can follow the procedure as described in training_v2.py to predict the label of a new audio file that may or may not contain a right whale upcall. 

Due to the complicated architecture of Model 1, it is not possible to directly load the model using Keras' "load_model" command. Instead, use the "model_v1_load.py" function to first re-create the model architecture, then load in the weights to the appropriate layers: 

    python model_v1_load.py 
    
With this model loaded, you can once again follow the procedure as described in training_v1.py to predict the label of a new audio file.

If you would like to replicate the process I used to train the CNN models, perform the following:
First, download the training set "train_2.zip" from [here](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/data) to the desired directory on your computer.
Then run either:

    python training_v1.py 
    
or:

    python training_v2.py 
    
This constructs the CNN model architectures of Model 1 or Model 2, respectively, trains the filters unsupervised via K-Means, and trains the weights on the dataset. This trained model can be saved to your computer using the command:

    model.save('model.h5')  
    
Filter Visualization (0th Layer)
=========================

The filters of the 0th convolutional layer in CNNs (applied to the raw input images) are often "human-interpretable" and have patterns that are easy to correlate with patterns of the input images. Both Model 1 and Model 2 learn the same number of filters (256) in the same manner via K-Means for the 0th layer. Examine a visualization of these filters in the grid below:

![filters_unsup](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Unsupervised-Learning/Images/filters_unsup.png)

*Note: Many patches appear to have patterns from the higher-intensity, brightly yellow-colored areas of the spectrogram containing a right whale upcall. Note, however, that other patches also appear to have be monochromatic and duller-colored - more representative of spectrograms with ambient noise. This is a product of the process used to train the filters via K-Means: Equal number of samples from the positive class (right whalle upcall) and negative class (ambient noise) were given to the algorithm to learn centroids, resulting in patches representative of both types of images. Including samples from both classes, as opposed to just including samples from the positive class, was found to boost classifier performance.* 

Results of Training for Model 1
=========================

Model 1 was trained for 17 epochs and a batch size of 100 on a training set of 84000 audio files (42000 vertically-enhanced spectrograms and 42000 horizontally-enhanced spectrograms). Training took approximately 4 hours on a Tesla K80 GPU (via FloydHub Cloud Service). The test set consisted of 10000 audio files (5000 vertically-enhanced spectrograms and 5000 horizontally-enhanced spectrograms). The loss and accuracy of the training set, and ROC-AUC score of the test set, are evaluated by Keras for every epoch during training and depicted below. The final ROC-AUC score for the training set after 17 epochs was found to be 94.14%, while the ROC-AUC score for the test set was found to be 93.13%.

- ROC-AUC Score vs Epoch (Graph)

![AUC-Epoch_ModelUnsup1](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Unsupervised-Learning/Images/Model-1/AUC-Epoch_ModelUnsup1.png)

- ROC-AUC Score vs Epoch (Table)

| Epoch                 | Loss        | Accuracy    | ROC-AUC     | 
|-----------------------|-------------|-------------|-------------|
| 1/17                  | 0.2438      | 0.9219      | 0.9020      | 
| 2/17                  | 0.2095      | 0.9326      | 0.9150      | 
| 3/17                  | 0.2040      | 0.9340      | 0.9144      | 
| 4/17                  | 0.2018      | 0.9346      | 0.9078      | 
| 5/17                  | 0.1998      | 0.9351      | 0.9269      | 
| 6/17                  | 0.1969      | 0.9360      | 0.9134      | 
| 7/17                  | 0.1959      | 0.9360      | 0.8939      | 
| 8/17                  | 0.1934      | 0.9362      | 0.9277      | 
| 9/17                  | 0.1939      | 0.9361      | 0.9044      | 
| 10/17                 | 0.1918      | 0.9365      | 0.9256      | 
| 11/17                 | 0.1931      | 0.9360      | 0.9320      | 
| 12/17                 | 0.1904      | 0.9363      | 0.9226      | 
| 13/17                 | 0.1897      | 0.9370      | 0.9127      | 
| 14/17                 | 0.1875      | 0.9362      | 0.9307      | 
| 15/17                 | 0.1898      | 0.9363      | 0.9278      | 
| 16/17                 | 0.1895      | 0.9367      | 0.9316      | 
| 17/17                 | 0.1874      | 0.9370      | 0.9044      | 

**Test ROC_AUC Score = 0.9313**

ROC Curves for Model 1
------

- Training Set ROC Curve vs Test Set ROC Curve
![ROC_ModelUnsup1_BP](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Unsupervised-Learning/Images/Model-1/ROC_ModelUnsup1_BP.png)

*Note: Predictions on the test set are made using the union of the predictions on the vertically-enhanced spectrograms and horizontally-enhanced spectrograms (BP=Both Predictions).*

- Test Set ROC Curves

![ROC_ModelUnsup1_TestOnly](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Unsupervised-Learning/Images/Model-1/ROC_ModelUnsup1_TestOnly.png)

*Note: The three curves represent predictions only on the vertically-enhanced spectrograms in the test set (VP=Vertically-Enhanced Predictions, predictions only on the horizontally-enhanced spectrograms in the test set (HP=Horizontally-Enhanced Predictions), and the union of the predictions on both types of images (BP=Both Predictions).*

- Training Set ROC Curve vs Test Set ROC Curves

![ROC_ModelUnsup1_All](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Unsupervised-Learning/Images/Model-1/ROC_ModelUnsup1_All.png)

Results of Training for Model 2
=========================

Model 2 was trained for 16 epochs and a batch size of 100 on a training set of 84000 audio files (42000 vertically-enhanced spectrograms and 42000 horizontally-enhanced spectrograms). Training took approximately 50 minutes on a Tesla K80 GPU (via FloydHub Cloud Service). The test set consisted of 10000 audio files (5000 vertically-enhanced spectrograms and 5000 horizontally-enhanced spectrograms). The loss and accuracy of the training set, and ROC-AUC score of the test set, are evaluated by Keras for every epoch during training and depicted below. The final ROC-AUC score for the training set after 16 epochs was found to be 96.07%, while the ROC-AUC score for the test set was found to be 95.97%.

- ROC-AUC Score vs Epoch (Graph)

![AUC-Epoch_ModelUnsup2](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Unsupervised-Learning/Images/Model-2/AUC-Epoch_ModelUnsup2.png)

- ROC-AUC Score vs Epoch (Table)

| Epoch                 | Loss        | Accuracy    | ROC-AUC     | 
|-----------------------|-------------|-------------|-------------|
| 1/16                  | 0.2313      | 0.9210      | 0.9354      | 
| 2/16                  | 0.1953      | 0.9303      | 0.9370      | 
| 3/16                  | 0.1870      | 0.9314      | 0.9420      | 
| 4/16                  | 0.1802      | 0.9330      | 0.9439      | 
| 5/16                  | 0.1768      | 0.9339      | 0.9368      | 
| 6/16                  | 0.1728      | 0.9339      | 0.9405      | 
| 7/16                  | 0.1720      | 0.9339      | 0.9419      | 
| 8/16                  | 0.1710      | 0.9344      | 0.9472      | 
| 9/16                  | 0.1686      | 0.9349      | 0.9383      | 
| 10/16                 | 0.1661      | 0.9357      | 0.9491      | 
| 11/16                 | 0.1650      | 0.9364      | 0.9375      | 
| 12/16                 | 0.1636      | 0.9378      | 0.9476      | 
| 13/16                 | 0.1623      | 0.9423      | 0.9395      | 
| 14/16                 | 0.1597      | 0.9414      | 0.9437      | 
| 15/16                 | 0.1594      | 0.9421      | 0.9433      | 
| 16/16                 | 0.1592      | 0.9423      | 0.9411      | 

**Test ROC_AUC Score = 0.9507**

ROC Curves for Model 2
------

- Training Set ROC Curve vs Test Set ROC Curve
![ROC_ModelUnsup2_BP](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Unsupervised-Learning/Images/Model-2/ROC_ModelUnsup2_BP.png)

*Note: Predictions on the test set are made using the union of the predictions on the vertically-enhanced spectrograms and horizontally-enhanced spectrograms (BP=Both Predictions).*

- Test Set ROC Curves

![ROC_ModelUnsup2_TestOnly](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Unsupervised-Learning/Images/Model-2/ROC_ModelUnsup2_TestOnly.png)

*Note: The three curves represent predictions only on the vertically-enhanced spectrograms in the test set (VP=Vertically-Enhanced Predictions, predictions only on the horizontally-enhanced spectrograms in the test set (HP=Horizontally-Enhanced Predictions), and the union of the predictions on both types of images (BP=Both Predictions).*

- Training Set ROC Curve vs Test Set ROC Curves

![ROC_ModelUnsup2_All](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Unsupervised-Learning/Images/Model-2/ROC_ModelUnsup2_All.png)

All Models: ROC-AUC Scores vs Epoch
=========================

![AUC-Epoch_All](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Unsupervised-Learning/Images/AUC-Epoch_All.png)

*Note: The three curves represent the ROC-AUC scores vs epoch for the supervised CNN, the unsupervised CNN using energy-correlated receptive field grouping, and the unsupervised CNN using 1x1 convolution dimensionality reduction, respectively.*

References
=========================

Coates A., Ng A.Y. (2012) [Learning Feature Representations with K-Means.](https://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf) In: Montavon G., Orr G.B., Müller KR. (eds) Neural Networks: Tricks of the Trade. Lecture Notes in Computer Science, vol 7700. Springer, Berlin, Heidelberg

Coates A., Ng A.Y. [Selecting receptive fields in deep networks.](http://robotics.stanford.edu/~ang/papers/nips11-SelectingReceptiveFields.pdf) In: Shawe-Taylor, J., Zemel, R., Bartlett, P., Pereira, F., Weinberger, K. (eds.) Advances in Neural Information Processing Systems 24, pp. 2528–2536. Curran Associates, Inc. (2011)

Salamon J, Bello JP, Farnsworth A, Robbins M, Keen S, Klinck H, et al. (2016) [Towards the Automatic Classification of Avian Flight Calls for Bioacoustic Monitoring.](http://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0166866&type=printable) PLoS ONE 11(11): e0166866. doi:10.1371/journal.pone.0166866

Lin M., Chen Q., Yan S. [Network In Network.](https://arxiv.org/pdf/1312.4400.pdf) *arXiv preprint arXiv:1312.4400*, 2014

Knyazev B., Barth E., Martinetz T. [Recursive Autoconvolution for Unsupervised Learning of Convolutional Neural Networks.](https://arxiv.org/pdf/1606.00611.pdf) *arXiv preprint arXiv:1606.00611*, 2017
