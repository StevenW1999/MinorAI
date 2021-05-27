import data as data
import cv2
import model as model


for emotion in model.emotions:
    data.order_data(emotion)


training_data, test_data = data.get_files()

model.train(training_data)
model.test(test_data)

model.detect_face('video')
