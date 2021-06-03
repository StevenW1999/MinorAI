import pandas as pd
import numpy as np
import warnings
import cv2
import data
import model
from PIL import Image
import dlib
warnings.filterwarnings("ignore")

# for emotion in model.emotions:
#     data.order_data(emotion)

data.get_files_CNN()
df = pd.read_csv("dataset.csv")
print(df.head())
image_size = (48, 48)
pixels = df['pixels'].tolist()  # Converting the relevant column element into a list for each row
width, height = 48, 48
faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]  # Splitting the string by space character as a list
    face = np.asarray(face).reshape(width, height)  # converting the list to numpy array in size of 48*48
    face = cv2.resize(face.astype('uint8'), image_size)  # resize the image to have 48 cols (width) and 48 rows (height)
    faces.append(face.astype('float32'))  # makes the list of each images of 48*48 and their pixels in numpyarray form

faces = np.asarray(faces)  # converting the list into numpy array
faces = np.expand_dims(faces, -1)  # Expand the shape of an array -1=last dimension => means color space
emotions = pd.get_dummies(df['emotion']).to_numpy()  # doing the one hot encoding type on emotions

print(faces[0])


print(faces.shape)
print(faces[0].ndim)
print(type(faces))
print(emotions[0]) #Emotion after preprocessing
print(emotions.shape)
print(emotions.ndim)
print(type(emotions))

x = faces.astype('float32')
x = x / 255.0 #Dividing the pixels by 255 for normalization  => range(0,1)

# Scaling the pixels value in range(-1,1)
x = x - 0.5
x = x * 2.0
print(x[0])
print(x.min(),x.max())

num_samples, num_classes = emotions.shape

num_samples = len(x)
num_train_samples = int((1 - 0.2)*num_samples)

# Traning data
train_x = x[:num_train_samples]
train_y = emotions[:num_train_samples]

# Validation data
val_x = x[num_train_samples:]
val_y = emotions[num_train_samples:]

train_data = (train_x, train_y)
val_data = (val_x, val_y)
