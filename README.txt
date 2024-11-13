Introduction
The problem statement given for the particular project is on
recognising and predicting the test images using a cnn model.
As a solution to approach this problem project,
implementation of cosine similarity between the training
features and the test features is being carried out.

Frame Extraction train Images:
The part 1 of this project consisted of obtaining training
videos from an Android application and uploading it to a
flask server. The obtained images are used for training the
model. The function extract_features(folder_name) in
main.py, is used to extract the the middle frame as png files.
The frameExtractor class contains a function to extract the
frames which accepts videos(.mp4) as inputs. The train data
directory is iterated and each gesture training video is passed
on to the frameExtractorobject.frameExtractor() function and
the returned image as png file is stored within each gesture
directory as explained in the below code.

Feature Extraction Training images:
The extract_features function in main.py is used for extracting
features from training images. In the function, the frame data
directory is iterated and within each gesture directory inside
frame data, the feature vector is obtained by passing each
image to function featureExtractor.extract_feature(img). This
image is read using cv.imread() as grayscale and sent to the
function. The returned feature vector is appended to an array.
When the gesture directory iteration ends, the features are
averaged over and stored in features array.

Frame and Feature Extraction test images:
The cosine_difference function iterates over test videos
provided, extracts the frame for each video using the similar
function as in training data, sends this frame image to the
feature extraction function and gets the feature vector. This
feature vector is compared with the 17 gesture training
averaged feature vectors using cosine Similarity and the
corresponding gesture for which the cosine is minimum is
considered as the predicted label. This label is stored in
results.csv

