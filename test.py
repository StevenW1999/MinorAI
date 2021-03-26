import cv2
import dlib

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

hog = dlib.get_frontal_face_detector()

face_landmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = hog(gray)
    for face in faces:

        landmarks = face_landmark(gray, face)

        def left_brow():
            for n in range(18, 22):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img, (x, y), 1, (0, 255, 255), 1)

        def right_brow():
            for n in range(23, 27):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img, (x, y), 1, (0, 255, 255), 1)

        def eyes():
            for n in range(37, 48):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img, (x, y), 1, (0, 255, 255), 1)

        def mouth():
            for n in range(49, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img, (x, y), 1, (0, 255, 255), 1)

        left_brow()
        right_brow()
        eyes()
        mouth()

    cv2.imshow('img', img)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
