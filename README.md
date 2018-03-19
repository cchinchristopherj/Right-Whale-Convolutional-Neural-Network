Whale Convolutional Neural Network 
=========================

Convolutional Neural Network to Recognize Right Whale Upcalls

Uses the dataset and labels constructed for [Kaggle 2013 ICML Whale Challenge](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux)

The training dataset for [Kaggle 2013 ICML Whale Challenge](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux) can be accessed [here](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/data) by selecting "train_2.zip" and clicking "Download"

The Kaggle challenge was to differentiate between short audio recordings of ambient noise vs right whale upcalls. A Convolutional Neural Network approach was chosen, with the final model architecture depicted in the attached "cnn-architecture.png" file:
- A spectrogram was first taken of every audio file in the training set 
- The spectrogram's contrast was enhanced both horizontally (temporal dimension) and vertically (frequency dimension) by removing extreme values and implementing a sliding mean. Performing these two pre-processing steps separately on the original training set resulted in a new dataset twice as long.
- All spectrograms were resized to (64x64x1) to ensure uniformity in input shape and facilitate input to the CNN 
- The CNN model's hyperparameters were optimized through 3-Fold Cross Validation via GridSearchCV
- The CNN was fit using the new, double-length, contrast-enhanced training set (84000 audio files). 
- The test set consisted of 10000 audio files (~10% of the training set size) unseen by the CNN model. For prediction, a vertically-enhanced and horizontally-enhanced spectrogram was produced for each audio file and fed into the CNN. A "union" approach was taken, i.e. if the vertically-enhanced input yielded 1 and the horizontally-enhanced input yielded 0, the final predicted label was 1. 
- Due to the unbalance between the sizes of the positive class (right whale upcall) and negative class (ambient noise), with the negative class being significantly larger, accuracy is not as useful a metric for model evaluation. (For example, a model that consistently predicted the negative class would yield a high accuracy, but fail to ever predict the positive class). The Receiver Operating Characteristic (ROC) curve was instead chosen for evaluation, being a measure of the true positive rate vs false positive rate as the discrimination threshold of the binary classifier is varied. The area under the curve (AUC) is a single number metric of a binary classifier's ROC curve and it is this ROC-AUC score that is used for evaluation of the CNN model. 
- The final ROC-AUC score for the test set after 45 epochs and a batch size of 100 was: **98.25%**

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

Results of Pre-Processing
=========================

Pre-Processing involved generation of spectrograms with vertically-enhanced contrast and horizontally-enhanced contrast for input to the CNN. Below is an example of an audio file with right whale upcall present before and after pre-processing.

- Spectrogram of an audio file with right whale upcall (original before pre-processing):
![upcall_original](https://github.com/cchinchristopherj/Whale-CNN/blob/test-branch/Images/Upcall/upcall_original%5B93%5D.png)

*Note: You can see a faint diagonal upward trajectory in the spectrogram from about 0.50 to 1.25 seconds that indicates the 
presence of the right whale upcall. This feature was enhanced in the following two images.*

- Spectrogram of the original audio file with vertically-enhanced contrast via the "SlidingWindowV" function in "whale_cnn.py":
![upcall_v_enhanced](https://github.com/cchinchristopherj/Whale-CNN/blob/test-branch/Images/Upcall/upcall_v_enhanced%5B93%5D.png)

*Note: The vertically-enhanced contrast emphasizes the upcall feature in the frequency domain, removing extreme values and 
local means to facilitate classification.*

- Spectrogram of the original audio file with horizontally-enhanced contrast via the "SlidingWindowH" function in "whale_cnn.py":
![upcall_h_enhanced](https://github.com/cchinchristopherj/Whale-CNN/blob/test-branch/Images/Upcall/upcall_h_enhanced%5B93%5D.png)

*Note: The horizontally-enhanced contrast emphasizes the upcall feature in the temporal domain and offers a different 
perspective of the same spectrogram for input to the classifier.*

For comparison, below is an example of an audio file with right whale upcall absent (solely ambient noise) before and after pre-processing.

- Spectrogram of an audio file with right whale upcall absent (original before pre-processing):
![ambient1_original](https://github.com/cchinchristopherj/Whale-CNN/blob/test-branch/Images/Ambient1/ambient1_original%5B0%5D.png)

*Note: At first glance with no pre-processing, there are no distinguishable features in the image, only a background
of low frequency ambient ocean noise.*

- Spectrogram of the original audio file with vertically-enhanced contrast via the "SlidingWindowV" function in "whale_cnn.py":
![ambient1_v_enhanced](https://github.com/cchinchristopherj/Whale-CNN/blob/test-branch/Images/Ambient1/ambient1_v_enhanced%5B0%5D.png)

*Note: The vertically-enhanced contrast removes extreme values and local means to facilitate classification. A small noise
feature from 0.8 to 1.0 seconds is emphasized. By comparison with the spectrograms of ambient noise (2 of 2), which empahsize a different shape of noise feature, there are a large variety of noise feature shapes that the classifier must learn to distinguish from features characteristic of right whale upcalls.*

- Spectrogram of the original audio file with horizontally-enhanced contrast via the "SlidingWindowH" function in "whale_cnn.py":
![ambient1_h_enhanced](https://github.com/cchinchristopherj/Whale-CNN/blob/test-branch/Images/Ambient1/ambient1_h_enhanced%5B0%5D.png)

*Note: The horizontally-enhanced contrast emphasizes the noise feature in the temporal domain and offers a different 
perspective of the same spectrogram for input to the classifier.*

For further comparison, below is another example of an audio with right whale upcall absent (solely ambient noise) before and after pre-processing.

- Spectrogram of an audio file with right whale upcall absent (original before pre-processing):
![ambient2_original](https://github.com/cchinchristopherj/Whale-CNN/blob/test-branch/Images/Ambient2/ambient2_original%5B1%5D.png)

*Note: At first glance with no pre-processing, there are no distinguishable features in the image, only a background
of low frequency ambient ocean noise.*

- Spectrogram of the original audio file with vertically-enhanced contrast via the "SlidingWindowV" function in "whale_cnn.py":
![ambient2_v_enhanced](https://github.com/cchinchristopherj/Whale-CNN/blob/test-branch/Images/Ambient2/ambient2_v_enhanced%5B1%5D.png)

*Note: The vertically-enhanced contrast removes extreme values and local means to facilitate classification. A small noise
feature from 0.50 to 1.50 seconds is emphasized. By comparison with the spectrograms of ambient noise (1 of 2), which emphasize a different shape of noise feature, there are a large variety of noise feature shapes that the classifier must learn to distinguish from features characteristic of right whale upcalls.*

- Spectrogram of the original audio file with horizontally-enhanced contrast via the "SlidingWindowH" function in "whale_cnn.py":
![ambient2_h_enhanced](https://github.com/cchinchristopherj/Whale-CNN/blob/test-branch/Images/Ambient2/ambient2_h_enhanced%5B1%5D.png)

*Note: The horizontally-enhanced contrast emphasizes the noise feature in the temporal domain and offers a different 
perspective of the same spectrogram for input to the classifier.*

Results of Tuning
=========================

Hyperparameter Optimization was conducted using GridSearchCV's default 3-Fold Cross Validation in tuning.py 
The printed results were as follows, with the best accuracy associated with 200 neurons in the fully connected layer. This is the number of neurons used in the final model architecture. 

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

Results of Training
=========================

The CNN model was trained for 45 epochs and a batch size of 100 on a training set of 84000 audio files (42000 vertically-enhanced spectrograms and 42000 horizontally-enhanced spectrograms). Training took approximately 8 hours on a MacBook Pro (2017). The test set consisted of 10000 audio files (5000 vertically-enhanced spectrograms and 5000 horizontally-enhanced spectrograms). The loss, accuracy, and ROC-AUC score evaluated by Keras for every epoch during training is depicted below. The final ROC-AUC score for the training set after 45 epochs was found to be 98.63%, while the ROC-AUC score for the test set was found to be 98.25%.

| Epoch                 | Loss        | Accuracy    | ROC-AUC     | 
|-----------------------|-------------|-------------|-------------|
| 1/45                  | 0.1761      | 0.9424      | 0.965       | 
| 2/45                  | 0.1199      | 0.9582      | 0.9717      | 
| 3/45                  | 0.1085      | 0.9624      | 0.9768      | 
| 4/45                  | 0.1026      | 0.9635      | 0.9783      | 
| 5/45                  | 0.1003      | 0.9636      | 0.9793      | 
| 6/45                  | 0.0974      | 0.9644      | 0.9811      | 
| 7/45                  | 0.0963      | 0.9651      | 0.9805      | 
| 8/45                  | 0.0955      | 0.9661      | 0.9821      | 
| 9/45                  | 0.0943      | 0.9658      | 0.9822      | 
| 10/45                 | 0.0928      | 0.9665      | 0.9826      | 
| 11/45                 | 0.0902      | 0.9675      | 0.9831      | 
| 12/45                 | 0.0912      | 0.9670      | 0.9828      | 
| 13/45                 | 0.0904      | 0.9671      | 0.9835      | 
| 14/45                 | 0.0895      | 0.9674      | 0.9825      | 
| 15/45                 | 0.0901      | 0.9664      | 0.9841      | 
| 16/45                 | 0.0898      | 0.9673      | 0.9846      | 
| 17/45                 | 0.0878      | 0.9680      | 0.9844      | 
| 18/45                 | 0.0893      | 0.9673      | 0.9847      | 
| 19/45                 | 0.0872      | 0.9681      | 0.9848      | 
| 20/45                 | 0.0865      | 0.9865      | 0.9848      | 
| 21/45                 | 0.0854      | 0.9689      | 0.9837      | 
| 22/45                 | 0.0859      | 0.9685      | 0.9845      | 
| 23/45                 | 0.0853      | 0.9683      | 0.9848      | 
| 24/45                 | 0.0858      | 0.9682      | 0.9851      | 
| 25/45                 | 0.0854      | 0.9690      | 0.985       | 
| 26/45                 | 0.0844      | 0.9690      | 0.985       | 
| 27/45                 | 0.0855      | 0.9692      | 0.9852      | 
| 28/45                 | 0.0849      | 0.9689      | 0.9856      | 
| 29/45                 | 0.0847      | 0.9685      | 0.9855      | 
| 30/45                 | 0.0839      | 0.9688      | 0.9857      | 
| 31/45                 | 0.0847      | 0.9687      | 0.9858      | 
| 32/45                 | 0.0837      | 0.9692      | 0.9856      | 
| 33/45                 | 0.0840      | 0.9689      | 0.9855      | 
| 34/45                 | 0.0850      | 0.9685      | 0.9858      | 
| 35/45                 | 0.0836      | 0.9692      | 0.9861      | 
| 36/45                 | 0.0837      | 0.9693      | 0.9857      | 
| 37/45                 | 0.0829      | 0.9687      | 0.9861      | 
| 38/45                 | 0.0831      | 0.9696      | 0.9862      | 
| 39/45                 | 0.0832      | 0.9690      | 0.9862      | 
| 40/45                 | 0.0828      | 0.9691      | 0.9865      | 
| 41/45                 | 0.0816      | 0.9703      | 0.9864      | 
| 42/45                 | 0.0829      | 0.9692      | 0.9862      | 
| 43/45                 | 0.0823      | 0.9695      | 0.986       | 
| 44/45                 | 0.0831      | 0.9691      | 0.9863      | 
| 45/45                 | 0.0815      | 0.9697      | 0.9863      | 

**Test ROC_AUC Score = 0.9825**
