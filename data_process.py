import pandas as pd
import numpy as np
import warnings
import cv2
warnings.filterwarnings("ignore")

# data process function


def process_data(csv, type):
    df = pd.read_csv(csv)

    # get all pixels to convert it to a 48*48 dimensional array
    image_size = (48, 48)
    pixels = df['pixels'].tolist()
    width, height = 48, 48
    faces = []

    # this one gets all pixels but returns it as a single dimensional array, needed for some algorithms
    # the pixels is a string with all the pixels so to make it useful,
    # we split the whole single string of pixels per pixel
    # and convert eacht one of them into an int and combine them all into a single list per datapoint
    pixels2 = df["pixels"].str.split()
    count = 0
    for i in pixels2:
        ci = 0
        for b in i:
            i[ci] = (((int(b) / 255) - 0.5) * 2.0)
            ci += 1
        pixels2[count] = i
        count += 1
    x2 = pixels2.tolist()

    # here we reshape the pixels into a 48*48 dimensional array because our images are 48*48
    # and add those "translated pixels" to the faces array
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'),
                          image_size)
        faces.append(
            face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)

    # one hot encode the emotions
    emotions = pd.get_dummies(df['emotion'])

    # non encoded emotions
    non_encoded_emotions = df['emotion']

    # divide all pixels by 255 to get a value between -1 and 1
    x = faces.astype('float32')
    x = x / 255.0

    x = x - 0.5
    x = x * 2.0

    d_x = x

    y = emotions
    y_ne = non_encoded_emotions

    if type == "CNN":
        return d_x, y, y_ne
    if type == "KNN":
        return d_x, y_ne
    if type == "OTHER":
        return x2, y_ne