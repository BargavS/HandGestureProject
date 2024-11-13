# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021
@author:
"""
import cv2
import numpy as np
import os
import tensorflow as tf
import frameextractor
import handshape_feature_extractor
import csv
## import the handfeature extractor class

feature = []
gesture_label = {
  "0": "0",
  "1": "1",
  "2": "2",
"3": "3",
  "4": "4",
  "5": "5",
"6": "6",
  "7": "7",
  "8": "8",
"9": "9",
  "DecreaseFanSpeed": "10",
  "FanOff": "12",
"FanOn": "11",
  "IncreaseFanSpeed": "13",
  "LightOff": "14",
"LightOn": "15",
  "SetThermo": "16",
}

featureExtractor = handshape_feature_extractor.HandShapeFeatureExtractor.get_instance()

def extract_frame(folder_name):
    video_folder = folder_name

    count = 1
    for video_file in os.listdir(video_folder):
        count = count + 1
        mp4_file_path = os.path.join(video_folder, video_file)
        if video_file.endswith('.mp4'):
                frame_extract_path = os.path.join("framedata", video_file.split("_")[0])
                frameextractor.frameExtractor(mp4_file_path, frame_extract_path, count)

def extract_features(folder_name):
    frame_folder = folder_name
    for gesture_folder in os.listdir(frame_folder):
        temparr = []
        for image_file in os.listdir(os.path.join(frame_folder,gesture_folder)):
            image_path = os.path.join(folder_name, gesture_folder, image_file)
            img3 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            print("imread-grayscale")
            temparr.append(featureExtractor.extract_feature(img3))
            feature_array = np.array(temparr)
        av_array = np.average(feature_array, axis=0)
        feature.append([av_array, gesture_folder])


def cosine_difference(folder_name):
    with open('Results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        video_folder = folder_name
        count = 1

        for video_file in sorted(os.listdir(video_folder)):
            mp4_file_path = os.path.join(folder_name, video_file)
            if video_file.endswith('.mp4'):
                frame_extract_path = "testframedata"
                frameextractor.frameExtractor(mp4_file_path, frame_extract_path, count)
                for frame_file in os.listdir('testframedata'):
                    cosine_similarity = 5

                    frame_path = os.path.join('testframedata', frame_file)
                    img3 = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                    print("imread-grayscale")
                    testfeature = featureExtractor.extract_feature(img3)
                    gesture = ""
                    for gesture_feature in feature:
                        cosine_sim_test_train = tf.keras.losses.cosine_similarity(testfeature, gesture_feature[0], axis=1)
                        if cosine_sim_test_train.numpy()[0]<cosine_similarity:
                            cosine_similarity = cosine_sim_test_train.numpy()[0]
                            gesture = gesture_feature[1]
                    print("actual:",video_file,"Predicted:",gesture)
                    writer.writerow({gesture_label[gesture]})


# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
extract_frame('traindata')
extract_features('framedata')
# Extract the middle frame of each gesture video

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================

cosine_difference('test')
# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
