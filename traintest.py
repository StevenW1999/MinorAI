import cv2
import dlib
import random
import numpy as np
import glob
from shutil import copyfile
import math
import pandas as pd
from jupyterlab_widgets import data
import os
from sklearn.neighbors import KNeighborsClassifier

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotions = ['anger', 'contempt', 'happy', 'sadness']


def get_landmarks(image):
    detections = detector(image, 1)
    landmarks = []
    for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(image, d)  # Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1, 68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        landmarks.append({'face_x': xlist, 'face_y': ylist})

    return landmarks


faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")


def detect_face(img_path):
    frame = cv2.imread(img_path)  # Open image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

    face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                    flags=cv2.CASCADE_SCALE_IMAGE)
    face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
    face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

    # Go over detected faces, stop at first detected face, return empty if no face.
    if len(face) == 1:
        facefeatures = face
    elif len(face_two) == 1:
        facefeatures = face_two
    elif len(face_three) == 1:
        facefeatures = face_three
    elif len(face_four) == 1:
        facefeatures = face_four
    else:
        facefeatures = ""

    for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
        if facefeatures == "":
            pass
        else:
            print("face found in file: %s" % img_path)
            gray = gray[y:y + h, x:x + w]  # Cut the frame to size

    return gray


happy_vec = get_landmarks(detect_face('emotions/happy/happy.jpg'))
anger_vec = get_landmarks(detect_face('emotions/anger/angry.jpg'))
contempt_vec = get_landmarks(detect_face('emotions/contempt/contempt.jpg'))
sadness_vec = get_landmarks(detect_face('emotions/sadness/sad.jpg'))

emotion_data = pd.Series((happy_vec, anger_vec, contempt_vec, sadness_vec), index=['happy', 'anger', 'contempt', 'sadness'])
print(emotion_data[0])


def order_data(emotion):
    files = glob.glob("CK+48\\%s\\*" % emotion)
    filenumber = 0
    for f in files:
        name = emotion + '_' + str(filenumber)
        try:
            out = detect_face(f)  # Resize face so all images have same size
            cv2.imwrite("datasets\\%s.jpg" % name, out)  # Write image
        except:
            pass  # If error, pass file
        filenumber += 1  # Increment image number


# for emotion in emotions:
#     order_data(emotion)

def get_files():  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob('datasets/*.jpg')
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
    return training, prediction


def make_sets():
    training_data = []
    test_data = []

    training, prediction = get_files()
    # Append data to training and prediction list
    for item in training:
        try:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            training_data.append(gray)
            # block raising an exception
        except:
            pass  # doing nothing on exception
    # append image array to training data list

    for item in prediction:
        try:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            test_data.append(gray)
        # block raising an exception
        except:
            pass  # doing nothing on exception
        # repeat above process for prediction set
    return training_data, test_data


training_d, test_d = make_sets()


def train(t_files):
    for f in t_files:
            landmarks = get_landmarks(f)
            # for emo in emotion_data:
            #     if np.allclose(landmarks, emo, 10, 10):
            #         emotion_data[emo] = (emotion_data[emo] + 100) / 2
            #     else:
            #         pass


train(training_d)

for emo in emotion_data:
    print(emotion_data[emo])

# print(emotion_data[0])
# def run_recognizer():
#     training_data, prediction_data = make_sets()
#
#     print("training fisher face classifier")
#     print("size of training set is:", len(training_labels), "images")
#     fishface.train(training_data, np.asarray(training_labels))
#
#     print("predicting classification set")
#     cnt = 0
#     ncorrect = 0
#     incorrect = 0
#     for image in prediction_data:
#         pred, conf = fishface.predict(image)
#         if pred == prediction_labels[cnt]:
#             ncorrect += 1
#             cnt += 1
#         else:
#             cv2.imwrite("dataset\\difficult\\%s_%s_%s.jpg" % (emotions[prediction_labels[cnt]], emotions[pred], cnt),
#                         image)  # <-- this one is new
#             incorrect += 1
#             cnt += 1
#     return ((100 * ncorrect) / (ncorrect + incorrect))
#
#
# # Now run it
# metascore = []
# for i in range(0, 10):
#     correct = run_recognizer()
#     print("got", correct, "percent correct!")
#     metascore.append(correct)
#
# print("\n\nend score:", np.mean(metascore), "percent correct!")
