Whale Convolutional Neural Network (Supervised)
=========================

Convolutional Neural Network to Recognize Right Whale Upcalls (via Standard Backpropagation)

The  purpose of this application was to detect right whales from hydrophone recordings by detecting the unique temporal evolution and spectral features of their characteristic upcalls. The dataset was made publicly available for the Kaggle challenge accompanying the Workshop on Machine Learning for Bioacoustics at ICML 2013. Concretely, the dataset consisted of 47841 2-second-long audio clips from recordings taken over four days that were annotated by an expert as either 1 (positive class indicating presence of a right whale upcall) or 0 (negative class indicating ambient noise and absence of an upcall). Following the precedent set by other state-of-the-art deep learning models designed for similar classification tasks, the 2-second audio clips were converted into spectrograms and features extracted from these images were used by a convolutional neural network to make predictions. It was found that an FFT size of 256 and overlap of 75% yielded an optimal balance of time and frequency resolution. 

Uses the dataset and labels constructed for [Kaggle 2013 ICML Whale Challenge](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux)

The training dataset for [Kaggle 2013 ICML Whale Challenge](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux) can be accessed [here](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/data) by selecting "train_2.zip" and clicking "Download"

With the competition having concluded, my goal was to discover alternative methodologies that could present advantages over the winning submissions. As a baseline to compare with future experimental work, I developed a Convolutional Neural Network (CNN) with hyperparameters based on those used by one of the leading competitors, Jure Zbontar.

The winning competitor, Nick Kridler, also used a pre-processing step to improve the horizontal (temporal dimension) and vertical (frequency dimension) contrast of the spectrogram images that proved crucial to improving results. The pre-processing step I implememted is based on the method used by Nick and is described below:

Pre-Processing
=========================

Both kinds of contrast enhancement were achieved by creating two moving average filters (denoted filter A and filter B), with the only difference between them being length: filter A had a length of 3 and filter B had a length of 32. (These values were determined through experimentation to yield spectrograms with the best time-frequency resolution). 

Below are a set of three images depicting the effect of moving average filters of different lengths on an input signal.
The original input signal is depicted below:

![filter_input](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/filter_input.png)

Moving average filter of length 3 applied: 
![filter_3](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/filter_3.png)

Moving average filter of length 51 applied: 
![filter_51](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/filter_51.png)

As demonstrated by the set of three images, moving average filters of greater length result in a higher degree of smoothing of the original input signal

For the temporal dimension, the following steps were then taken: first, each row of the spectrogram (representing the change in one frequency of the spectrogram over time) was convolved separately with filter A and with filter B. Since filter B had a much longer length than filter A, the values of the output of the convolution operation with filter B represented global averages for neighborhoods of pixels, while the output of the convolution with filter A represented local averages for adjacent pixels. Contrast enhancement was achieved by subtracting out these local averages from their corresponding rows, thereby emphasizing more significant temporal changes. 

For the frequency domain, the procedure is nearly identical: each column of the spectrogram (representing the power distribution of frequencies for one time step) is convolved separately with filter A and with filter B, the output of the convolution operation with filter A is subtracted from the output of the convolution with filter B, and the actual local averages subtracted from each corresponding row, thereby attaining contrast enhancement and emphasizing more significant differences within the frequency power distribution for each time step. 

This contrast enhancement pre-processing was a critically important step, not only because it facilitated feature extraction for the deep-learning model, but also because it performed what is known as “data augmentation.” Since supervised learning models, especially those that suffer from high variance, require large amounts of labeled data to learn an optimal function mapping from input to output and this labeled data can be very time and resource-expensive to generate (due to requiring an expert human annotator), researchers often artificially “augment” the size of their given training sets to alleviate this issue. 

![data_augment](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/data_augment.png)

If the task, for example, is to classify words from a recording of speech, noise could be randomly added to the time series values of each sample. Since noise would not change what words were said, only the difficulty with which they are detected, each new sample would have the same label as the original sample, but act as a new source of information for the deep learning model to learn the relationship between input features and output class predictions. Alternatively, if the task is to classify images of cats, the images in the dataset could be translated, rotated, scaled, etc., so that transformed samples act as new sources of information, while retaining the same label as the samples they were derived from. Two advantages are therefore gained in both of these scenarios at no additional cost: the models become more robust to noise/translations/rotations/scaling, and no expert annotator is needed to provide new labels. 

The same advantages apply to the data augmentation achieved through contrast enhancement pre-processing: two new kinds of spectrogram images are produced from one original sample, effectively doubling the number of samples in the dataset: 47841 original spectrogram samples are transformed into 47841 frequency domain contrast-enhanced images and 47841 temporal domain contrast-enhanced images, yielding a new augmented dataset with 95682 spectrogram images. (After the pre-processing steps, the spectrograms are each of shape (50,59,1), i.e. there are 50 frequency bins, 59 time steps, and one channel with the magnitude of values in the matrix interpreted as colors in the image). 

Results of Pre-Processing
=========================

Pre-Processing involved generation of spectrograms with vertically-enhanced contrast and horizontally-enhanced contrast for input to the CNN. Below is an example of an audio file with right whale upcall present before and after pre-processing.

- Spectrogram of an audio file with right whale upcall (original before pre-processing):
![upcall_original](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Upcall/upcall_original%5B93%5D.png)

*Note: You can see a faint diagonal upward trajectory in the spectrogram from about 0.50 to 1.25 seconds that indicates the 
presence of the right whale upcall. This feature was enhanced in the following two images.*

- Spectrogram of the original audio file with vertically-enhanced contrast via the "SlidingWindowV" function in "whale_cnn.py":
![upcall_v_enhanced](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Upcall/upcall_v_enhanced%5B93%5D.png)

*Note: The vertically-enhanced contrast emphasizes the upcall feature in the frequency domain, removing extreme values and 
local means to facilitate classification.*

- Spectrogram of the original audio file with horizontally-enhanced contrast via the "SlidingWindowH" function in "whale_cnn.py":
![upcall_h_enhanced](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Upcall/upcall_h_enhanced%5B93%5D.png)

*Note: The horizontally-enhanced contrast emphasizes the upcall feature in the temporal domain and offers a different 
perspective of the same spectrogram for input to the classifier.*

For comparison, below is an example of an audio file with right whale upcall absent (solely ambient noise) before and after pre-processing.

- Spectrogram of an audio file with right whale upcall absent (original before pre-processing):
![ambient1_original](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Ambient1/ambient1_original%5B0%5D.png)

*Note: At first glance with no pre-processing, there are no distinguishable features in the image, only a background
of low frequency ambient ocean noise.*

- Spectrogram of the original audio file with vertically-enhanced contrast via the "SlidingWindowV" function in "whale_cnn.py":
![ambient1_v_enhanced](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Ambient1/ambient1_v_enhanced%5B0%5D.png)

*Note: The vertically-enhanced contrast removes extreme values and local means to facilitate classification. A small noise
feature from 0.8 to 1.0 seconds is emphasized. By comparison with the spectrograms of ambient noise (2 of 2), which empahsize a different shape of noise feature, there are a large variety of noise feature shapes that the classifier must learn to distinguish from features characteristic of right whale upcalls.*

- Spectrogram of the original audio file with horizontally-enhanced contrast via the "SlidingWindowH" function in "whale_cnn.py":
![ambient1_h_enhanced](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Ambient1/ambient1_h_enhanced%5B0%5D.png)

*Note: The horizontally-enhanced contrast emphasizes the noise feature in the temporal domain and offers a different 
perspective of the same spectrogram for input to the classifier.*

For further comparison, below is another example of an audio with right whale upcall absent (solely ambient noise) before and after pre-processing.

- Spectrogram of an audio file with right whale upcall absent (original before pre-processing):
![ambient2_original](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Ambient2/ambient2_original%5B1%5D.png)

*Note: At first glance with no pre-processing, there are no distinguishable features in the image, only a background
of low frequency ambient ocean noise.*

- Spectrogram of the original audio file with vertically-enhanced contrast via the "SlidingWindowV" function in "whale_cnn.py":
![ambient2_v_enhanced](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Ambient2/ambient2_v_enhanced%5B1%5D.png)

*Note: The vertically-enhanced contrast removes extreme values and local means to facilitate classification. A small noise
feature from 0.50 to 1.50 seconds is emphasized. By comparison with the spectrograms of ambient noise (1 of 2), which emphasize a different shape of noise feature, there are a large variety of noise feature shapes that the classifier must learn to distinguish from features characteristic of right whale upcalls.*

- Spectrogram of the original audio file with horizontally-enhanced contrast via the "SlidingWindowH" function in "whale_cnn.py":
![ambient2_h_enhanced](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Ambient2/ambient2_h_enhanced%5B1%5D.png)

*Note: The horizontally-enhanced contrast emphasizes the noise feature in the temporal domain and offers a different 
perspective of the same spectrogram for input to the classifier.*

Convolutional Neural Network
=========================

A Convolutional Neural Network (CNN) was used to learn an optimal function mapping from input (spectrogram images) to output (class predictions). These models consist of "convolutional layers" which implement the convolution operation displayed graphically below:

![convolution_gif](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/convolution_gif.gif)

Each convolutional layer consists of K convolutional filters - the graphic above depicts the result of the convolution operation between an input image and one of these filters: the filter is superimposed over a smaller patch of the image and a "matrix dot product" (elementwise multiplication and summation) is computed between the filter and patch. The result of the matrix dot product (a scalar) is stored in the relevant location of the output matrix, after which the filter is slid over to the next patch and the process is repeated. 

The filters themselves can be thought of as feature extractors learned through backpropagation and gradient descent - during training, the neural network will learn more optimal sets of filters capable of extracting more informative features from the input images. 

An additional layer implemented in CNNs that assists in the feature extraction process is known as a "max pooling" layer. These layers provide not only downsampling (reducing the height and width dimensions to lower the number of required parameters in the model) but also translation invariance. The max pooling operation itself can be considered similar to a convolutional layer in the sense that a "filter" is slid over an input image. The filter, however, does not perform a matrix dot product - instead, the maximum in each patch of the input image is determined and that maximum value is the value that is stored in the relevant location of the output matrix. 

The picture below displays a classic CNN architecture, in which the convolutional and max pooling layers function as (translation invariant) feature extractors and the fully-connected layers that follow function as the classifier: 

![cnn](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/cnn.jpeg)

Training Procedure
=========================

- A spectrogram was first taken of every audio file in the training set 
- The spectrogram's contrast was enhanced both horizontally (temporal dimension) and vertically (frequency dimension) by removing extreme values and implementing a sliding mean. 
- All spectrograms were resized to (50x59x1) to ensure uniformity in input shape and facilitate input to the CNN 
- The CNN model's hyperparameters were optimized through 3-Fold Cross Validation via GridSearchCV
- The CNN was fit using the new, double-length, contrast-enhanced training set
- The test set consisted of ~10% of the dataset unseen by the CNN model. For prediction, a vertically-enhanced and horizontally-enhanced spectrogram was produced for each audio file and fed into the CNN. A "union" approach was taken, i.e. if the vertically-enhanced input yielded 1 and the horizontally-enhanced input yielded 0, the final predicted label was 1. 
- Due to the unbalance between the sizes of the positive class (right whale upcall) and negative class (ambient noise), with the negative class being significantly larger, accuracy is not as useful a metric for model evaluation. (For example, a model that consistently predicted the negative class would yield a high accuracy, but fail to ever predict the positive class). The Receiver Operating Characteristic (ROC) curve was instead chosen for evaluation, being a measure of the true positive rate vs false positive rate as the discrimination threshold of the binary classifier is varied. The area under the curve (AUC) is a single number metric of a binary classifier's ROC curve and it is this ROC-AUC score that is used for evaluation of the CNN model. 

Filter Visualization (0th Layer)
=========================

The filters of the 0th convolutional layer in CNNs (applied to the raw input images) are often "human-interpretable" and have patterns that are easy to correlate with patterns of the input images.

![filters_sup](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/filters_sup.png)

*Note: Many patches appear to have patterns from the higher-intensity, brightly yellow-colored areas of the spectrogram containing a right whale upcall.*

Results of Tuning
=========================

Hyperparameter Optimization was conducted using GridSearchCV's default 3-Fold Cross Validation to determine an optimum combination of hyperparameters for the CNN.

As an example, the number of neurons in the fully-connected layer ("fc1" in the diagram below) was tuned using this method in tuning.py. The printed results were as follows, with the best accuracy associated with 200 neurons. This is the number of neurons used in the final model architecture. 

| Neurons               | Mean Accuracy  | Std(Accuracy) | 
|-----------------------|----------------|---------------|
| 100                   | 0.9611         | 0.004162      | 
| 125                   | 0.9620         | 0.004944      | 
| 150                   | 0.9583         | 0.003391      | 
| 175                   | 0.9615         | 0.005162      | 
| **200**               | **0.9633**     | **0.003427**  | 
| 225                   | 0.9581         | 0.008134      | 
| 250                   | 0.9603         | 0.001154      | 

**Best: Neurons = 200, Mean = 0.9633, Std = 0.003427**


The final model architecture was based off of the results of hyperparameter optimization, as well as the approach taken by June Zbontar (5th in the Competition) in [The Marinexplore and Cornell University Whale Detection Challenge.](https://www.kaggle.com/c/whale-detection-challenge) The final tuned architecture is depicted below: 

![cnn_architecture](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/cnn_architecture.png)

Results of Training
=========================

The CNN model was trained for 14 epochs and a batch size of 100 on a training set of 84000 audio files (42000 vertically-enhanced spectrograms and 42000 horizontally-enhanced spectrograms). Training took approximately 7 minutes on a Tesla K80 GPU (via FloydHub Cloud Service). The test set consisted of 10000 audio files (5000 vertically-enhanced spectrograms and 5000 horizontally-enhanced spectrograms). The loss and accuracy of the training set, and ROC-AUC score of the test set were evaluated by Keras for every epoch during training and depicted below. The final ROC-AUC score for the training set after 14 epochs was found to be 98.50%, while the ROC-AUC score for the test set was found to be 98.29%.

**Test ROC-AUC Score = 0.9829**

ROC Curves
------

- Training Set ROC Curve vs Test Set ROC Curve
![ROC_ModelSup_BP](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Graphs/ROC_ModelSup_BP.png)

*Note: Predictions on the test set are made using the union of the predictions on the vertically-enhanced spectrograms and horizontally-enhanced spectrograms (BP=Both Predictions).*

- Test Set ROC Curves

![ROC_ModelSup_TestOnly](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Graphs/ROC_ModelSup_TestOnly.png)

*Note: The three curves represent predictions only on the vertically-enhanced spectrograms in the test set (VP=Vertically-Enhanced Predictions, predictions only on the horizontally-enhanced spectrograms in the test set (HP=Horizontally-Enhanced Predictions), and the union of the predictions on both types of images (BP=Both Predictions).*

- Training Set ROC Curve vs Test Set ROC Curves

![ROC_ModelSup_All](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Graphs/ROC_ModelSup_All.png)

Modules and Installation Instructions
=========================

**"Standard" Modules Used (Included in Anaconda Distribution):** numpy, matplotlib, pylab, glob, aifc, os

If necessary, these modules can also be installed via PyPI. For example, for the "numpy" module: 

        pip install numpy

**Additional Modules used:** skimage, sklearn, keras

skimage and sklearn can be installed via PyPI. For example, for the "scikit-image" module:

        pip install scikit-image

For Keras, follow the instructions given in the [documentation](https://keras.io/#installation). Specifically, install the TensorFlow backend and, of the optional dependencies, install HDF5 and h5py if you would like to load and save your Keras models. 

Correct Usage
=========================

My trained CNN model architecture and weights are saved in the "whale_cnn.h5" file. This file can be loaded using the command:

    loaded_model = load_model('whale_cnn.h5')  
    
Note: "load_model" is a function from keras.models. 
With this model loaded, you can follow the procedure as described in training.py to predict the label of a new audio file that may or may not contain a right whale upcall. 
Note: Currently, code is not streamlined to predict the label of a new audio file not originating from train_2.zip (i.e. a new audio file from the user). A future implementation will most likely use sklearn's Pipeline to streamline this prediction process by automatically taking an input audio file, producing the vertically-enhanced and horizontally-enhanced spectrograms, feeding them into the CNN, and unioning the predicted labels to produce the final predicted label. 

If you would like to replicate the process I used to train the CNN model, perform the following:
First, download the training set "train_2.zip" from [here](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/data) to the desired directory on your computer.
Then run:

    python training.py 
    
This trains the weights of the CNN model on the dataset. With the "model" variable referring to the sklearn-wrapped Keras model, this trained model can be saved to your computer using the command:

    model.model.save('whale_cnn.h5')  
    
Note: Since "model" has an sklearn wrapper, you must use model.model.save instead of model.save (as you would for a normal Keras model) to save it to your computer. 

Code is also provided demonstrating my hyperparameter optimization process via GridSearchCV. If you would like to replicate this procedure to tune the number of neurons in the CNN model's fully-connected layer, first download the training set "train_2.zip" from [here](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/data) if not previously downloaded. 
Then run:

    python tuning.py 
    
This prints the best score and corresponding parameter (best # of neurons) found by GridSearchCV, along with the mean scores and standard deviation of the scores found for all of the other parameters. 
Note: For simplicity, these "scores" produced by GridSearchCV were accuracy-based, which is not ideal metric due to the nature of this particular dataset, as described previously. A future implementation will use the ROC-AUC scores for hyperparameter tuning. 
