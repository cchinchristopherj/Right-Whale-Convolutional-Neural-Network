Images Folder
=========================
- "cnn_architecture.png" is an image of the final CNN model architecture tuned through hyperparameter optimization via
sklearn's GridSearchCV
- "filters_sup.png" is a visualization of the convolutional filters learned by the CNN in the 0th layer (layer that receives raw input images)
- "Upcall" is a folder containing spectrogram images of an audio file with right whale upcall present. For comparison,
three spectrogram images are included: spectrogram with no contrast enhancement, spectrogram with vertically-enhanced contrast,
spectrogram with horizontally-enhanced contrast 
- "Ambient1" is a folder containing spectrogram images of an audio file with right whale upcall absent (solely ambient noise).
The three types of spectrogram images are likewise included for comparison.
- "Ambient2" is a folder containing spectrogram images of another audio file with right whale upcall absent (solely ambient noise)
for comparison with "Ambient1"'s images. The three types of spectrogram images are likewise included. 
- "Graphs" is a folder containing ROC curve and AUC vs Epoch graphs from training results
