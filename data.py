import cv2
import random
import glob
import pandas as pd
import model


def initial_data():
    happy_vec = model.get_landmarks(model.detect_face('dataset/happy_2.jpg'))
    anger_vec = model.get_landmarks(model.detect_face('dataset/anger_19.jpg'))
    contempt_vec = model.get_landmarks(model.detect_face('dataset/contempt_1.jpg'))
    sadness_vec = model.get_landmarks(model.detect_face('dataset/sadness_1.jpg'))
    t = pd.Series((happy_vec, anger_vec, contempt_vec, sadness_vec), index=['happy', 'anger', 'contempt', 'sadness'])
    return t


emotion_data = initial_data()


def order_data(emotion):
    files = glob.glob("CK+48\\%s\\*" % emotion)
    filenumber = 0
    for f in files:
        name = emotion + '_' + str(filenumber)
        try:
            out = model.detect_face(f)  # Resize face so all images have same size
            cv2.imwrite("datasets\\%s.jpg" % name, out)  # Write image
        except:
            pass  # If error, pass file
        filenumber += 1  # Increment image number


# for emotion in emotions:
#     order_data(emotion)

def get_files():  # Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob('dataset/*.jpg')
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
    return training, prediction


