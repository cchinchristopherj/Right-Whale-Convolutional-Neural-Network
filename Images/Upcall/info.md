Spectrograms of Right Whale Upcall
=========================
- "[93]" refers to the index of the file in the vector of filenames pulled from the folder in "train_2.zip."
- "upcall_original[93].png" is a spectrogram of an audio file with right whale upcall (original with no enhancements). 
- *Note: You can see a faint diagonal upward trajectory in the spectrogram from about 0.50 to 1.25 seconds that indicates the 
presence of the right whale upcall. This feature was enhanced in the following two images.*
- "upcall_v_enhanced[93].png" is a spectrogram of the original audio file with vertically-enhanced contrast via the 
"SlidingWindowV" function in "whale_cnn.py"
- *Note: The vertically-enhanced contrast emphasizes the upcall feature in the frequency domain, removing extreme values and 
local means to facilitate classification.*
- "upcall_h_enhanced[93].png" is a spectrogram of the original audio file with horizontally-enhanced contrast via the 
"SlidingWindowH" function in "whale_cnn.py"
- *Note: The horizontally-enhanced contrast emphasizes the upcall feature in the temporal domain and offers a different 
perspective of the same spectrogram for input to the classifier.*
