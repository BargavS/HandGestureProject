extract_frame function
extracts the frames for training images /traindata
stores the frames in /framedata/Gesturename/filename.png - 17 subdirectories
extract_features function 
extracts training features from /framedata/ and averages the feature vectors adn stores in feature[]
cosine_similarity function extracts frame, sends frame to feature extract and stores the feature vector inside testfeature[]
compares test_feature vector with 17 feature vectors using cosine_similarity in Keras
the gesture label corresponding to the minimum is stored in csv.
The same is repeated for 51 test videos 
