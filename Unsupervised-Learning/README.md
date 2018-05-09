Whale Convolutional Neural Network (Unsupervised)
=========================

Convolutional Neural Network to Recognize Right Whale Upcalls via Filters Learned Through K-Means

Uses the dataset and labels constructed for [Kaggle 2013 ICML Whale Challenge](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux)

The training dataset for [Kaggle 2013 ICML Whale Challenge](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux) can be accessed [here](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/data) by selecting "train_2.zip" and clicking "Download"

The final tuned model architecture is as depicted below: 

![cnn_architecture](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Unsupervised-Learning/Images/cnn_architecture_unsup.png)

Results of Training
=========================

The CNN model was trained for 16 epochs and a batch size of 100 on a training set of 84000 audio files (42000 vertically-enhanced spectrograms and 42000 horizontally-enhanced spectrograms). Training took approximately 50 minutes on a Tesla K80 GPU (via FloydHub Cloud Service). The test set consisted of 10000 audio files (5000 vertically-enhanced spectrograms and 5000 horizontally-enhanced spectrograms). The loss and accuracy of the training set, and ROC-AUC score of the test set, are evaluated by Keras for every epoch during training and depicted below. The final ROC-AUC score for the training set after 16 epochs was found to be 94.91%, while the ROC-AUC score for the test set was found to be 94.91%.

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

**Test ROC_AUC Score = 0.9491**

