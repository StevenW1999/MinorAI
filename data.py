import cv2
import random
import glob
import pandas as pd
import csv
import dlib
from PIL import Image

# function for initial empty dataframe
# def initial_data():
#     # happy_vec = model.get_landmarks(model.detect_face('dataset/happy_2.jpg'))
#     # anger_vec = model.get_landmarks(model.detect_face('dataset/anger_19.jpg'))
#     # contempt_vec = model.get_landmarks(model.detect_face('dataset/contempt_1.jpg'))
#     # sadness_vec = model.get_landmarks(model.detect_face('dataset/sadness_1.jpg'))
#     # t = pd.Series((happy_vec, anger_vec, contempt_vec, sadness_vec), index=['happy', 'anger', 'contempt', 'sadness'])
#     # happy_data = {'x': [], 'y': [], 'emotion': ''}
#     # sad_data = {'x': [], 'y': [], 'emotion': ''}
#     # angry_data = {'x': [], 'y': [], 'emotion': ''}
#     # contempt_data = {'x': [], 'y': [], 'emotion': ''}
#     # happy = pd.DataFrame(data=happy_data)
#     # sad = pd.DataFrame(data=sad_data)
#     # angry = pd.DataFrame(data=angry_data)
#     # contempt = pd.DataFrame(data=contempt_data)
#     # t = pd.Series((happy, sad, angry, contempt), index=['happy', 'anger', 'contempt', 'sadness'])
#
#     data = {'pixels': [], 'emotion': ''}
#     t = pd.DataFrame(data=data)
#     return t



#import detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotions = ['anger', 'contempt', 'happy', 'sadness']

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# emotion_data = initial_data()
def detect_face(img_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    emotions = ['anger', 'contempt', 'happy', 'sadness']
    faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
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
    # for k, d in enumerate(detections):
    #     shape = predictor(gray, d)
        # for i in range(1, 68):
            # cv2.circle(gray, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), thickness=1)
    gray = cv2.resize(gray, (48, 48))
    cv2.imwrite(img_path, gray)
    return gray

#function to get all files in the emotion folder and put them in a dataset folder with format [emotion] + _ + number +.jpg
def order_data(emotion):
    files = glob.glob("CK+48\\%s\\*" % emotion)
    nr = 0
    for f in files:
        name = emotion + '_' + str(nr)
        try:
            out = detect_face(f)
            cv2.imwrite("dataset3\\%s.jpg" % name, out)
        except:
            pass
        nr += 1

# function to split datset in 75% training 25% test

def get_files():
    files = glob.glob('dataset2/*.jpg')
    random.shuffle(files)
    training = files[:int(len(files) * 0.75)]
    prediction = files[-int(len(files) * 0.25):]
    return training, prediction


def write_csv():
    files = glob.glob('dataset3/*.jpg')
    random.shuffle(files)
    with open('dataset2.csv', mode='w') as csv_file:
        fieldnames = ['pixels', 'emotion']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for f in files:
            filename = f
            parts = filename.split('_')
            name = parts[0]
            parts2 = name.split('\\')
            label = parts2[1]
            # detect_face(f)
            pixels = Image.open(f)
            pixels = list(pixels.getdata())
            string_ints = [str(x) for x in pixels]
            str_of_ints = " ".join(string_ints)
            writer.writerow({'pixels': str_of_ints, 'emotion': label})
