Spectrograms of Ambient Noise (2 of 2)
=========================
- "[1]" refers to the index of the file in the vector of filenames pulled from the folder in "train_2.zip."
- "ambient2_original[1].png" is a spectrogram of an audio file with right whale upcall (original with no enhancements). 
- *Note: At first glance with no pre-processing, there are no distinguishable features in the image, only a background
of low frequency ambient ocean noise.*
- "ambient2_v_enhanced[1].png" is a spectrogram of the original audio file with vertically-enhanced contrast via the 
"SlidingWindowV" function in "whale_cnn.py"
- *Note: The vertically-enhanced contrast removes extreme values and local means to facilitate classification. A small noise
feature from 0.50 to 1.50 seconds is emphasized. By comparison with the spectrograms of ambient noise (1 of 2), which empahsize 
a different shape of noise feature, there are a large variety of noise feature shapes that the classifier must learn to 
distinguish from features characteristic of right whale upcalls.*
- "ambient2_h_enhanced[1].png" is a spectrogram of the original audio file with horizontally-enhanced contrast via the 
"SlidingWindowH" function in "whale_cnn.py"
- *Note: The horizontally-enhanced contrast emphasizes the noise feature in the temporal domain and offers a different 
perspective of the same spectrogram for input to the classifier.*

