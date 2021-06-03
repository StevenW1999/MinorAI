import pandas as pd
import numpy as np
import warnings
import cv2
import data
import model
from PIL import Image
import dlib
warnings.filterwarnings("ignore")


# data.get_files_CNN()
df = pd.read_csv("dataset.csv")
print(df.head())

#
# img = 'dataset2/anger_18.jpg'
# pixels = Image.open(img)
# pixels = list(pixels.getdata())
# print(pixels)
#
# detect_face(img)
# data.get_files_CNN()

# image_size = (48, 48)
#
#
# w = 48
# h = 48
#
# faces = []
# for x in pixels:
#     for pixel_sequence in x:
#
#         face = np.asarray(pixel_sequence).reshape(w, h)  # converting the list to numpy array in size of 48*48
#         face = cv2.resize(face.astype('uint8'), image_size)  # resize the image to have 48 cols (width) and 48 rows (height)
#         faces.append(face.astype('float32'))  # makes the list of each images of 48*48 and their pixels in numpyarray form
#
# faces = np.asarray(faces)  # converting the list into numpy array
# faces = np.expand_dims(faces, -1)  # Expand the shape of an array -1=last dimension => means color space
# emotions = pd.get_dummies(training_labels).to_numpy()  # doing the one hot encoding type on emotions

# print(faces[0])