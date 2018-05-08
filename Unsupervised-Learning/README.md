Whale Convolutional Neural Network (Unsupervised)
=========================

Convolutional Neural Network to Recognize Right Whale Upcalls via Filters Learned Through K-Means

Uses the dataset and labels constructed for [Kaggle 2013 ICML Whale Challenge](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux)

The training dataset for [Kaggle 2013 ICML Whale Challenge](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux) can be accessed [here](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/data) by selecting "train_2.zip" and clicking "Download"

The final tuned model architecture is as depicted below: 

![cnn_architecture](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/cnn_architecture.png)

Results of Training
=========================

The CNN model was trained for 22 epochs and a batch size of 100 on a training set of 84000 audio files (42000 vertically-enhanced spectrograms and 42000 horizontally-enhanced spectrograms). Training took approximately 1 hour 10 minutes on a Tesla K80 GPU (via FloydHub Cloud Service). The test set consisted of 10000 audio files (5000 vertically-enhanced spectrograms and 5000 horizontally-enhanced spectrograms). The loss and accuracy of the training set, and ROC-AUC score of the test set, are evaluated by Keras for every epoch during training and depicted below. The final ROC-AUC score for the training set after 22 epochs was found to be 98.63%, while the ROC-AUC score for the test set was found to be 98.25%.

| Epoch                 | Loss        | Accuracy    | ROC-AUC     | 
|-----------------------|-------------|-------------|-------------|
| 1/22                  | 0.1761      | 0.9424      | 0.965       | 
| 2/22                  | 0.1199      | 0.9582      | 0.9717      | 
| 3/22                  | 0.1085      | 0.9624      | 0.9768      | 
| 4/22                  | 0.1026      | 0.9635      | 0.9783      | 
| 5/22                  | 0.1003      | 0.9636      | 0.9793      | 
| 6/22                  | 0.0974      | 0.9644      | 0.9811      | 
| 7/22                  | 0.0963      | 0.9651      | 0.9805      | 
| 8/22                  | 0.0955      | 0.9661      | 0.9821      | 
| 9/22                  | 0.0943      | 0.9658      | 0.9822      | 
| 10/22                 | 0.0928      | 0.9665      | 0.9826      | 
| 11/22                 | 0.0902      | 0.9675      | 0.9831      | 
| 12/22                 | 0.0912      | 0.9670      | 0.9828      | 
| 13/22                 | 0.0904      | 0.9671      | 0.9835      | 
| 14/22                 | 0.0895      | 0.9674      | 0.9825      | 
| 15/22                 | 0.0901      | 0.9664      | 0.9841      | 
| 16/22                 | 0.0898      | 0.9673      | 0.9846      | 
| 17/22                 | 0.0878      | 0.9680      | 0.9844      | 
| 18/22                 | 0.0893      | 0.9673      | 0.9847      | 
| 19/22                 | 0.0872      | 0.9681      | 0.9848      | 
| 20/22                 | 0.0865      | 0.9865      | 0.9848      | 
| 21/22                 | 0.0854      | 0.9689      | 0.9837      | 
| 22/22                 | 0.0859      | 0.9685      | 0.9845      | 

**Test ROC_AUC Score = 0.9502**

