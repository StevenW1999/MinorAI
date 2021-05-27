import cv2
import time
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks(image):
    detections = detector(image, 1)
    for k, d in enumerate(detections):  # For all detected face instances individually
        shape = predictor(image, d)  # Draw Facial Landmarks with the predictor class
        for i in range(1, 68):  # Store X and Y coordinates in two lists
            cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), thickness=2)




def cap(emotion):
    cam = cv2.VideoCapture(0)

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("\nInitializing face capture for emotion: "+emotion+" Look the camera and wait ...")
    time.sleep(5.0)
    print("\n please look with the emotion: " + emotion + " at the camera and press SPACEBAR to take a picture.")
    print("A total of 30 pictures are needed")

    count = 0
    while True:
        k = cv2.waitKey(1)
        ret, img = cam.read()
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            get_landmarks(img)
            if k == 32:
                cv2.imwrite("dataset/" + emotion + '_' +
                            str(count) + ".jpg", gray[y:y + h, x:x + w])
                print('picture :' + str(count) + ' of 30 taken')
                count += 1
            cv2.imshow('image', img)
        if k == 27:
            break
        elif count >= 30:
            break

    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
    return 'done'
