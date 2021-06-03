import math
import data as data
import cv2
import dlib
import time
from collections import Counter
import pandas as pd
from PIL import Image
import numpy as np

#import detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotions = ['anger', 'contempt', 'happy', 'sadness']
faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def detect_face(img_path):
    if img_path == 'video':
        cap = cv2.VideoCapture(0)
        while True:
            k = cv2.waitKey(1)
            ret, img = cap.read()
            # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceDet.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                gray = gray[y:y + h, x:x + w]
                # get_landmarks(img)

            cv2.imshow('video', img)

            if k == 32:
                try:
                    gray = cv2.resize(gray, (350, 350))
                    print(predict(gray, 5))
                except:
                    print('error')
            if k == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        return 'close'

    else:
            frame = cv2.imread(img_path)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
            if len(face) == 1:
                facefeatures = face
            else:
                facefeatures = ""

            for (x, y, w, h) in facefeatures:
                if facefeatures == "":
                    print("no face found in file: %s" % img_path)
                else:
                    gray = gray[y:y + h, x:x + w]

            detections = detector(gray, 1)
            for k, d in enumerate(detections):
                shape = predictor(gray, d)
                for i in range(1, 68):
                    cv2.circle(gray, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), thickness=1)
            gray = cv2.resize(gray, (48, 48))
            cv2.imwrite(img_path, gray)


def train(t_data, label):
    for f, b in zip(t_data, label):
        try:
            data.emotion_data.loc[-1] = [f, b]
            data.emotion_data.index = data.emotion_data.index + 1
            data.emotion_data = data.emotion_data.sort_index()
        except:
            print('error in "' + b)


def euclidean_distance(row1, row2):
    distance_x = 0.0
    distance_x += (row1 - row2)**2
    return sum(sum(sum(np.sqrt(distance_x))))


def predict(face, k):
    distance_d = {'distance': [], 'emotion': ''}
    df = pd.DataFrame(data=distance_d)
    for i, r in data.emotion_data.iterrows():
        distance = euclidean_distance(data.emotion_data.iloc[i]['pixels'], face)
        df.loc[-1] = [distance, data.emotion_data.iloc[i]['emotion']]
        df.index = df.index + 1
        df = df.sort_index()
    df2 = df.sort_values(by=['distance'], ascending=True, axis=0)[:k]
    counter = Counter(df2['emotion'])
    prediction = counter.most_common()[0][0]
    return prediction


def test(p_data, label):
    correct = 0
    incorrect = 0
    for f, b in zip(p_data, label):
        predict_face = predict(f, 5)
        if b == predict_face:
            correct += 1
        else:
            incorrect += 1
    accuracy = (1 / (correct + incorrect)) * correct
    print('correct: ' + str(correct) + '\n' + 'incorrect: ' + str(incorrect) + '\n' + 'accuracy: ' + str(accuracy))
