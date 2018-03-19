Spectrograms of Ambient Noise (1 of 2)
=========================
- "[0]" refers to the index of the file in the vector of filenames pulled from the folder in "train_2.zip."
- "ambient1_original[0].png" is a spectrogram of an audio file with right whale upcall (original with no enhancements). 
- *Note: At first glance with no pre-processing, there are no distinguishable features in the image, only a background
of low frequency ambient ocean noise.*
- "ambient1_v_enhanced[0].png" is a spectrogram of the original audio file with vertically-enhanced contrast via the 
"SlidingWindowV" function in "whale_cnn.py"
- *Note: The vertically-enhanced contrast removes extreme values and local means to facilitate classification. A small 
feature from 0.8 to 1.0 seconds is emphasized, which the classifier will learn to distinguish from features characteristic
of right whale upcalls.*
- "ambient1_h_enhanced[0].png" is a spectrogram of the original audio file with horizontally-enhanced contrast via the 
"SlidingWindowH" function in "whale_cnn.py"
- *Note: The horizontally-enhanced contrast emphasizes the noise feature in the temporal domain and offers a different 
perspective of the same spectrogram for input to the classifier.*
