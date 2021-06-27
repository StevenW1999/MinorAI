import pandas as pd
import numpy as np
import warnings
import cv2
import data
from PIL import Image
import dlib
import data_process
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
warnings.filterwarnings("ignore")


x,y, a = data_process.process_data('dataset.csv', "CNN")

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
val_data = (X_test,y_test)

input_shape=(48, 48, 1)
num_classes = 4

model_1 = Sequential()
model_1.add(Conv2D(32, (5, 5), strides = (1,1), padding='same', input_shape=input_shape))
model_1.add(Activation('relu'))
model_1.add(Conv2D(32, (5, 5), strides = (1,1)))
model_1.add(Activation('relu'))
model_1.add(MaxPooling2D(pool_size=(2, 2)))
model_1.add(Dropout(0.25))
model_1.add(Flatten())
model_1.add(Dense(512))
model_1.add(Activation('relu'))
model_1.add(Dropout(0.5))
model_1.add(Dense(num_classes))
model_1.add(Activation('softmax'))

model_1.summary()

data_generator = ImageDataGenerator(
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        horizontal_flip=True)
batch_size = 32
opt = RMSprop(lr=0.0005, decay=1e-6)

model_1.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

hist_model_1 = model_1.fit_generator(data_generator.flow(X_train, y_train,
                                            batch_size),
                        epochs=20, verbose=1,validation_data =val_data)

model_1.evaluate(X_test, y_test, verbose=1)

#
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#
# faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# cap = cv2.VideoCapture(0)
#
# while True:
#     string = 'PRESS SPACE TO TAKE A PICTURE'
#     k = cv2.waitKey(1)
#     ret, img = cap.read()
#     # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     # img = cv2.flip(img, 1)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = faceDet.detectMultiScale(gray, 1.3, 5)
#     for (x, y, w, h) in faces:
#         rec = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         gray = gray[y:y + h, x:x + w]
#
#
#     cv2.imshow('video', img)
#     if k == 32:
#         try:
#             gray = cv2.resize(gray, (48, 48))
#             cv2.imwrite("liveCapture/" +
#                         'LIVECAPTURE' + ".jpg", gray)
#             break
#         except:
#             print('error')
#     if k == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()

pixels = Image.open('liveCapture/LIVECAPTURE.jpg')
pixels = list(pixels.getdata())

image_size = (48, 48)
width, height = 48, 48

faces = []

face = np.asarray(pixels).reshape(width, height)
face = cv2.resize(face.astype('uint8'),image_size)
faces.append(face.astype('float32'))
faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)

x = faces.astype('float32')
x = x / 255.0

x = x - 0.5
x = x * 2.0

y_pred = model_1.predict_classes(x)

mood = ''
if y_pred == [0]:
    mood = 'anger'
if y_pred == [1]:
    mood = 'contempt'
if y_pred == [2]:
    mood = 'happy'
if y_pred == [3]:
    mood = 'sadness'

print(y_pred)
print(mood)