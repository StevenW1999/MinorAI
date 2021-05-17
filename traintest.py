import cv2
import dlib
import random
import numpy as np
import glob
from shutil import copyfile
import math

from jupyterlab_widgets import data
from sklearn.svm import SVC

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks(image):
    detections = detector(image, 1)
    for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(image, d)  # Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1, 68):  # Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist)  # Find both coordinates of centre of gravity
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]  # Calculate distance centre <-> other points in both axes
        ycentral = [(y - ymean) for y in ylist]

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x) * 360) / (2 * math.pi))

        data['landmarks_vectorised'] = landmarks_vectorised
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"


emotions = ["neutral", "anger", "happy", "sadness"]  # Define emotion order
participants = glob.glob("source_emotion\\*")  # Returns a list of all folders with participant numbers

for x in participants:
    part = "%s" % x[-4:]  # store current participant number
    for sessions in glob.glob("%s\\*" % x):  # Store list of sessions for current participant
        for files in glob.glob("%s\\*" % sessions):
            current_session = files[20:-30]
            file = open(files, 'r')

            emotion = int(
                float(file.readline()))  # emotions are encoded as a float, readline as float, then convert to integer.

            sourcefile_emotion = glob.glob("source_images\\%s\\%s\\*" % (part, current_session))[
                -1]  # get path for last image in sequence, which contains the emotion
            sourcefile_neutral = glob.glob("source_images\\%s\\%s\\*" % (part, current_session))[
                0]  # do same for neutral image

            dest_neut = "sorted_set\\neutral\\%s" % sourcefile_neutral[25:]  # Generate path to put neutral image
            dest_emot = "sorted_set\\%s\\%s" % (
            emotions[emotion], sourcefile_emotion[25:])  # Do same for emotion containing image

            copyfile(sourcefile_neutral, dest_neut)  # Copy file
            copyfile(sourcefile_emotion, dest_emot)  # Copy file

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

emotions = ["neutral", "anger", "happy", "sadness"]  # Define emotions


def detect_faces(emotion):
    t_files = glob.glob("sorted_set\\%s\\*" % emotion)  # Get list of all images with emotion

    filenumber = 0
    for f in t_files:
        frame = cv2.imread(f)  # Open image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

        # Detect face using 4 different classifiers
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

        # Cut and save face
        for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
            print("face found in file: %s" % f)
            gray = gray[y:y + h, x:x + w]  # Cut the frame to size

            try:
                out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
                cv2.imwrite("dataset\\%s\\%s.jpg" % (emotion, filenumber), out)  # Write image
            except:
                pass  # If error, pass file
        filenumber += 1  # Increment image number


for emotion in emotions:
    detect_faces(emotion)  # Call functiona

# data = {}
# emotions = ["neutral", "anger", "happy", "sadness"]
# fishface = cv2.face.FisherFaceRecognizer_create() #createFisherFaceRecognizer() #Initialize fisher face classifier
#
# def get_files(emotion):  # Define function to get file list, randomly shuffle it and split 80/20
#     a_files = glob.glob("dataset\\%s\\*" % emotion)
#     random.shuffle(a_files)
#     training = a_files[:int(len(a_files) * 0.8)]  # get first 80% of file list
#     prediction = a_files[-int(len(a_files) * 0.2):]  # get last 20% of file list
#     return training, prediction

#
# def make_sets():
#     training_data = []
#     training_labels = []
#     prediction_data = []
#     prediction_labels = []
#     for nemotion in emotions:
#         training, prediction = get_files(nemotion)
#         # Append data to training and prediction list, and generate labels 0-7
#         for item in training:
#             image = cv2.imread(item)  # open image
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
#             training_data.append(gray)  # append image array to training data list
#             training_labels.append(emotions.index(nemotion))
#
#         for item in prediction:  # repeat above process for prediction set
#             image = cv2.imread(item)
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             prediction_data.append(gray)
#             prediction_labels.append(emotions.index(nemotion))
#
#     return training_data, training_labels, prediction_data, prediction_labels
#
# def run_recognizer():
#     training_data, training_labels, prediction_data, prediction_labels = make_sets()
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