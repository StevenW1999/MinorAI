import math
import data as data
import cv2
import dlib
import time
from collections import Counter
import pandas as pd

#import detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotions = ['anger', 'contempt', 'happy', 'sadness']


def get_landmarks(image):
    detections = detector(image, 1)
    landmarks_x = []
    landmarks_y = []
    for k, d in enumerate(detections):
        shape = predictor(image, d)
        xlist = []
        ylist = []
        for i in range(1, 68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), thickness=2)
        landmarks_x = xlist
        landmarks_y = ylist

    return [landmarks_x, landmarks_y]


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
                get_landmarks(img)

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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                gray = gray[y:y + h, x:x + w]
                gray = cv2.resize(gray, (350, 350))

        return gray


def train(t_data):
    for f in t_data:
        try:
            face = detect_face(f)
            landmarks = get_landmarks(face)

            filename = f
            parts = filename.split('_')
            name = parts[0]
            parts2 = name.split('\\')
            emotion = parts2[1]

            data.emotion_data.loc[-1] = [landmarks[0], landmarks[1], emotion]
            data.emotion_data.index = data.emotion_data.index + 1
            data.emotion_data = data.emotion_data.sort_index()
            #
            # new_x = [data.emotion_data[emotion][0][i] + landmarks[0][i] for i in range(len(data.emotion_data[emotion][0]))]
            # new_y = [data.emotion_data[emotion][1][i] + landmarks[1][i] for i in range(len(data.emotion_data[emotion][1]))]
            #
            # new_x_average = [x / 2 for x in new_x]
            # new_y_average = [y / 2 for y in new_y]
            #
            # data.emotion_data[emotion][0] = new_x_average
            # data.emotion_data[emotion][1] = new_y_average
        except:
            print('error in "' + f)


def euclidean_distance(rowx1, rowx2, rowy1, rowy2):
    distance_x = 0.0
    distance_y = 0.0
    for i in range(len(rowx1)-1):
        distance_x += (rowx1[i] - rowx2[i])**2

    for i in range(len(rowy1) - 1):
        distance_y += (rowy1[i] - rowy2[i]) ** 2

    return math.sqrt(distance_x), math.sqrt(distance_y)


def predict(face, k):
    landmarks = get_landmarks(face)
    rowx = landmarks[0]
    rowy = landmarks[1]
    distance_d = {'distance_x': [], 'distance_y': [], 'emotion': ''}
    df = pd.DataFrame(data=distance_d)
    for i, r in data.emotion_data.iterrows():
        try:
            distance = euclidean_distance(data.emotion_data.iloc[i]['x'], rowx, data.emotion_data.iloc[i]['y'], rowy)
            df.loc[-1] = [distance[0], distance[1], data.emotion_data.iloc[i]['emotion']]
            df.index = df.index + 1
            df = df.sort_index()
        except:
            return 'no idea'
    df2 = df.sort_values(by=['distance_x', 'distance_y'], ascending=[True, True], axis=0)[:k]
    counter = Counter(df2['emotion'])
    prediction = counter.most_common()[0][0]
    return prediction


def test(p_data):
    correct = 0
    incorrect = 0
    for f in p_data:
        face = detect_face(f)
        predict_face = predict(face, 5)

        filename = f
        parts = filename.split('_')
        name = parts[0]
        parts2 = name.split('\\')
        emotion = parts2[1]

        if emotion == predict_face:
            correct += 1
        else:
            incorrect += 1
    accuracy = (1 / (correct + incorrect)) * correct
    print('correct: ' + str(correct) + '\n' + 'incorrect: ' + str(incorrect) + '\n' + 'accuracy: ' + str(accuracy))
